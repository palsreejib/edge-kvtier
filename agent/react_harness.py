"""
ReAct trace harness for edge-kvtier (deliverable M-3).

Start the server first (one slot, so every step of a task hits the same KV cache):
  llama-server.exe -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 16384 -np 1 -fa 1 -ngl 99 -dev Vulkan1 --port 8080
  (CPU: drop -ngl/-dev and add -t 16. RTX 4060: -dev Vulkan0.)

Run:  python react_harness.py --platform igpu --backend vulkan --out traces/2026-11-23_igpu.csv

One CSV row per agent step, in the agreed schema:
  task_id, step, platform, backend, ctx_tokens, cached_tokens, new_tokens, out_tokens,
  prefill_ms, decode_ms, tool_name, tool_gap_ms, success
Plus a raw JSONL log of every request/response, so nothing is lost if the schema changes.
"""
import argparse, csv, json, re, time
import requests

SERVER = "http://127.0.0.1:8080" # got from the llama.cpp server; documentation
MAX_STEPS = 16          # hard cap so a confused 3B model can't loop forever
N_PREDICT = 256         # cap on tokens per step

pages = {"page_title": "",
        "page_text": "",
        "last_keyword": "",
        "position": 0} # common dict for search and lookup to share found page info

# ---------------------------------------------------------------------------
# 1. Tools. Each takes a string argument and returns a string observation.
#    Mix fast and slow tools on purpose: M3's gap predictor needs different
#    gap lengths per tool. Keep outputs a few hundred to ~1,500 tokens so the
#    context grows toward ~8k by step 10 (the proposal's 3,000 -> 7,800 loop).
# ---------------------------------------------------------------------------
# def tool_calc(arg):            # fast (~ms)
#     # TODO: safe arithmetic eval (e.g. ast-based), return the number as a string
#     raise NotImplementedError

def search(arg):     
    url = 'https://en.wikipedia.org/w/api.php'
    headers = {"User-Agent":  "Friday/1.1 (https://starkindustries.com)"} # Change to a real address when it works! - follow MediaWiki
    params = {"action": "query",
              "format": "json",
              "formatversion": 2,
              "explaintext": 1,
              "redirects": 1,
              "prop": "extracts",
              "titles": arg,
              }
    r = requests.get(url=url, headers=headers, params=params, timeout=5)
    r.raise_for_status()

    inter = r.json()
    page = inter["query"]["pages"][0]

    if page.get("missing") or page.get("invalid"):
        params = {"action": "query",
                    "list": "search",
                    "srlimit": 5, #Default=10
                    "srsearch": arg,
                    "format": "json",
                    "formatversion": 2}
        r = requests.get(url=url, headers=headers, params = params, timeout=5)
        r.raise_for_status()
        results = r.json()["query"]["search"]

    else:
        text = page["extract"]

def lookup(arg):       # medium, large output: this is what makes the context grow
    # TODO: return file contents, truncated to ~1,500 tokens (~6,000 chars)
    raise NotImplementedError

# def tool_run_python(arg):      # slow and variable: subprocess with a timeout (e.g. run pytest)
#     # TODO: subprocess.run([...], capture_output=True, timeout=60); return stdout/stderr tail
#     raise NotImplementedError

TOOLS = {
    # "calc": tool_calc,
    "search_docs": search,
    "read_file": lookup,
    # "run_python": tool_run_python,
}

# ---------------------------------------------------------------------------
# 2. Prompt. Built once per task, then ONLY APPENDED TO. Never rebuild it from a
#    message list: re-rendering can re-tokenise the boundary and break the prefix
#    cache, which would corrupt cached_tokens.
# ---------------------------------------------------------------------------
SYSTEM = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an agent that solves tasks by calling tools. Use exactly this format:
Thought: <reasoning>
Action: <tool name>
Action Input: <argument>
Observation: <tool result, written by the system>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I know the answer.
Final Answer: <answer>

Tools:
- search: searches local documents, returns snippets
- lookup: returns the contents of a file

TODO: add 1-2 full worked examples here. A 3B model needs them to follow the format.
<|eot_id|>"""

def start_prompt(task_text):
    return (SYSTEM
            + "<|start_header_id|>user<|end_header_id|>\n\n" + task_text + "<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>\n\nThought:")

# ---------------------------------------------------------------------------
# 3. One LLM step against llama-server's native /completion endpoint.
# ---------------------------------------------------------------------------
def llm_step(prompt):
    body = { # body depends entirely on the llama.cpp server; check documentation for why this format
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0.0,          # deterministic: same tokens on every platform
        "seed": 42,
        "cache_prompt": True,        # reuse the prefix (this is what B1 models)
        "id_slot": 0,
        "stop": ["Observation:", "<|eot_id|>"],   # model must not invent tool results
    }
    r = requests.post(f"{SERVER}/completion", json=body, timeout=600)
    r.raise_for_status()
    return r.json()

def parse_action(text):
    """Returns (tool_name, arg) or ("finish", answer) or (None, None) if unparseable."""
    m = re.search(r"Final Answer:\s*(.*)", text, re.S)
    if m:
        return "finish", m.group(1).strip()
    m = re.search(r"Action:\s*(\S+)\s*Action Input:\s*(.*)", text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None

# ---------------------------------------------------------------------------
# 4. The loop. tool_gap_ms on row k = time from step k's response arriving to
#    step k+1's request being sent (tool execution + prompt building).
# ---------------------------------------------------------------------------
def run_task(task, platform, backend, log, kv_type):
    prompt = start_prompt(task["text"])
    rows = []
    for step in range(1, MAX_STEPS + 1):
        resp = llm_step(prompt)
        t_resp = time.perf_counter()
        log.write(json.dumps({"task_id": task["id"], "step": step, "resp": resp}) + "\n")

        tm = resp["timings"]
        # Check these once by printing resp on your build (b11889). cache_n = reused prompt
        # tokens, prompt_n = newly prefilled tokens, predicted_n = generated tokens.
        row = {
            "task_id": task["id"], "step": step, "platform": platform, "backend": backend,
            "cached_tokens": tm["cache_n"],
            "new_tokens": tm["prompt_n"],
            "ctx_tokens": tm["cache_n"] + tm["prompt_n"],
            "out_tokens": tm["predicted_n"],
            "prefill_ms": round(tm["prompt_ms"], 2),
            "decode_ms": round(tm["predicted_ms"], 2),
            "kv_type": kv_type,
        }

        text = resp["content"]
        prompt += text                      # append exactly what the model produced
        tool, arg = parse_action(text)

        if tool == "finish" or tool is None or step == MAX_STEPS:
            row.update(tool_name="finish" if tool == "finish" else "abort", tool_gap_ms=0)
            rows.append(row)
            answer = arg if tool == "finish" else None
            return rows, answer

        fn = TOOLS.get(tool)
        try:
            obs = fn(arg) if fn else f"Unknown tool '{tool}'. Use one of {list(TOOLS)}."
        except Exception as e:
            obs = f"Error: {e}"
        prompt += f"\nObservation: {obs}\nThought:"
        t_next = time.perf_counter()        # next request is sent right after this

        row.update(tool_name=tool, tool_gap_ms=round((t_next - t_resp) * 1000, 2))
        rows.append(row)
    return rows, None

# ---------------------------------------------------------------------------
# 5. Tasks: ~10 tasks that each need ~10 tool calls. Multi-hop over local files
#    works well (find a number in one file, another in a second, compute, verify).
# ---------------------------------------------------------------------------
TASKS = [
    # TODO: {"id": "t01", "text": "...", "answer": "..."},
]

def check(answer, expected):
    # TODO: loose match (numbers within tolerance, or substring)
    return answer is not None and expected.lower() in answer.lower()

FIELDS = ["task_id", "step", "platform", "backend", "kv_type", "ctx_tokens", "cached_tokens", "new_tokens",
          "out_tokens", "prefill_ms", "decode_ms", "tool_name", "tool_gap_ms", "success"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)   # cpu | igpu | rtx4060
    ap.add_argument("--backend", required=True)    # cpu | vulkan
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-type", default="q4_0")
    a = ap.parse_args()

    with open(a.out, "w", newline="") as f, open(a.out + ".jsonl", "w") as log:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for task in TASKS:
            # Clear the slot so task N doesn't reuse task N-1's KV beyond the shared system prompt.
            # TODO: decide; simplest is to restart nothing and accept the system-prompt reuse
            # (that's realistic). Note the choice in the run log.
            rows, answer = run_task(task, a.platform, a.backend, a.kv_type, log)
            ok = int(check(answer, task["answer"]))
            for r in rows:
                r["success"] = ok          # task-level outcome, repeated on every row
                w.writerow(r)
            f.flush()
            print(f"{task['id']}: {len(rows)} steps, success={ok}, final ctx={rows[-1]['ctx_tokens']}")
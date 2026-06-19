#!/usr/bin/env python3
"""glm-bench-client.py LABEL BASE_URL OUTDIR

Run a small general battery against an OpenAI-compatible chat endpoint, recording
each answer, token usage, wall time, and decode tok/s.  Used to compare DeepSeek
V4 Flash (prod, port 8085) with GLM-5.2 Q2 / Q4.  Stdlib only (urllib)."""
import json, sys, time, urllib.request, urllib.error

LABEL, BASE_URL, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
import os
TEMP = float(os.environ.get("BENCH_TEMP", "0"))
REASONING = os.environ.get("BENCH_REASONING", "none")  # none|high|max
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "200"))    # raise for reasoning (think+answer)

# General battery: factual, factual+complex (the "hard" prompt), reasoning, code,
# and a categorization-style prompt from the user's domain.
BATTERY = [
    ("factual_short",  "List the first five planets from the Sun."),
    ("factual_complex","List the first five planets from the Sun and give one interesting fact about each one."),
    ("reasoning_math", "A train goes 240 km in 3 hours, then 180 km in 2 hours. What is its average speed for the whole trip? Give the number and a one-line explanation."),
    ("code",           "Write a Python function longest_increasing_run(nums) that returns the length of the longest strictly increasing run in a list of integers."),
    ("categorize",     "Categorize this bank transaction into one spending category. Reply with just the category name.\n\"TRADER JOE'S #482 LOS ANGELES CA $63.18\""),
]

def call(content):
    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAXTOK, "temperature": TEMP, "top_p": 0.95,
        "reasoning_effort": REASONING,
    }).encode()
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    return {
        "wall_s": round(dt, 2),
        "completion_tokens": ct,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "tok_per_s": round(ct / dt, 2) if dt > 0 else 0,
        "finish": d["choices"][0].get("finish_reason"),
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content", ""),
    }

summary = []
for name, prompt in BATTERY:
    try:
        res = call(prompt)
    except Exception as e:
        res = {"error": str(e)}
    with open(f"{OUTDIR}/{LABEL}.{name}.json", "w") as f:
        json.dump({"prompt": prompt, **res}, f, indent=2)
    if "error" in res:
        line = f"  {name:16s} ERROR {res['error']}"
    else:
        line = (f"  {name:16s} {res['completion_tokens']:4d} tok  {res['wall_s']:6.1f}s  "
                f"{res['tok_per_s']:5.1f} tok/s  finish={res['finish']}")
    summary.append(line)
    print(line, flush=True)

with open(f"{OUTDIR}/{LABEL}.summary.txt", "w") as f:
    f.write(f"=== {LABEL} ({BASE_URL}) ===\n" + "\n".join(summary) + "\n")

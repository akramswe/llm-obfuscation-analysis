import os
import subprocess
import json
from datetime import datetime
from groq import Groq, RateLimitError, APIStatusError

# ===================== CONFIG =====================
PROGRAMS_DIR = "programs"
OBFUSCATED_DIR = "obfuscated"
RESULTS_DIR = "results"

MODEL_NAME = "openai/gpt-oss-20b"
TARGET_FUNCTION = "main"
RUNS = 1
MAX_CHARS = 6000

GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

GROUND_TRUTH = {
    "anagram.c": "anagram",
    "armstrong.c": "armstrong",
}

# ===================== SETUP ======================
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(OBFUSCATED_DIR, "flatten"), exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

# ===================== TIGRESS ====================
def run_tigress_flatten(input_c, output_c):
    cmd = [
        "tigress",
        "--Transform=Flatten",
        "--Functions=main",
        input_c,
        f"--out={output_c}",
    ]
    try:
        subprocess.run(cmd, check=True)
        return os.path.exists(output_c)
    except subprocess.CalledProcessError:
        return False

# ===================== CODE PREP ==================
def strip_comments(code):
    return "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("//")
    )

def trim_to_entry(code, max_chars):
    for key in [
        "FUNCTION-DEF main",
        "_TIG_FN_",
        "int main"
    ]:
        idx = code.find(key)
        if idx != -1:
            return code[idx:idx + max_chars]
    return code[-max_chars:]

def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
    return text.strip()

# ===================== GROQ =======================
def query_groq(code):
    code = strip_comments(code)
    code = trim_to_entry(code, MAX_CHARS)

    prompt = f"""
You are an expert reverse engineer analyzing C code obfuscated using
Tigress CONTROL-FLOW FLATTENING only.

Important assumptions:
- Semantics are preserved exactly.
- No virtualization, no opaque predicates.
- Flattening introduces an artificial dispatcher/state variable.

Your task:
1) Identify the dispatcher variable.
2) Reconstruct the original control flow.
3) Identify the underlying algorithm implemented by main().

Rules:
- Ignore dispatcher mechanics.
- Focus on data flow, arithmetic, comparisons, and I/O.
- Do NOT guess.
- If the algorithm is not uniquely identifiable, output "cannot determine".

Output JSON ONLY:

{{
  "algorithm": string,
  "confidence": number,
  "dispatcher_variable": string,
  "key_operations": [string],
  "reconstructed_logic": string,
  "why_not_other_algorithms": string
}}

Decision threshold:
- confidence must be >= 0.85 to name an algorithm.

Analyze the following flattened C code:

<<<CODE
{code}
CODE
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except (RateLimitError, APIStatusError) as e:
        print("Groq API error:", e)
        exit(1)

# ===================== CLASSIFIER =================
def classify_response(resp_text, truth):
    try:
        resp_text = clean_json(resp_text)
        data = json.loads(resp_text)
    except json.JSONDecodeError:
        return "unknown", "Invalid JSON output"

    algo = data.get("algorithm", "").lower()
    conf = data.get("confidence", 0.0)

    # normalize algorithm names
    if "armstrong" in algo:
        algo = "armstrong"
    if "anagram" in algo:
        algo = "anagram"

    if algo == truth and conf >= 0.85:
        return "correct", "Correct with high confidence"

    if algo == "cannot determine" or conf < 0.85:
        return "unknown", "Insufficient confidence"

    return "incorrect", f"Incorrect identification: {algo}"

# ===================== MAIN =======================
def main():
    results = []

    for prog in os.listdir(PROGRAMS_DIR):
        if not prog.endswith(".c"):
            continue

        in_path = os.path.join(PROGRAMS_DIR, prog)
        out_name = prog.replace(".c", "_flatten.c")
        out_path = os.path.join(OBFUSCATED_DIR, "flatten", out_name)

        if not run_tigress_flatten(in_path, out_path):
            print(f"Flatten failed for {prog}")
            continue

        with open(out_path, "r") as f:
            code = f.read()

        entry = {
            "program": prog,
            "ground_truth": GROUND_TRUTH.get(prog, "unknown"),
            "runs": [],
        }

        resp = query_groq(code)
        classification, comment = classify_response(
            resp, entry["ground_truth"]
        )

        entry["runs"].append({
            "run": 1,
            "llm_raw_output": resp,
            "classification": classification,
            "comment": comment,
        })

        entry["summary"] = {
            "correct": 1 if classification == "correct" else 0,
            "incorrect": 1 if classification == "incorrect" else 0,
            "unknown": 1 if classification == "unknown" else 0,
            "consistency": "high" if classification == "correct" else "low",
        }

        results.append(entry)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    final_output = {
        "experiment_metadata": {
            "model": MODEL_NAME,
            "runs_per_program": RUNS,
            "obfuscation": "Flatten",
            "timestamp": timestamp,
        },
        "results": results,
    }

    out_file = os.path.join(
        RESULTS_DIR, f"results_flatten_{timestamp}.json"
    )
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {out_file}")

# ===================== RUN ========================
if __name__ == "__main__":
    main()

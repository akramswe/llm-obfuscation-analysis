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
RUNS = 2
MAX_CHARS = 6000

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

GROUND_TRUTH = {
    "anagram.c": "anagram",
    "armstrong.c": "armstrong",
}

def normalize_algorithm(name):
    name = name.lower()
    if "anagram" in name:
        return "anagram"
    if "armstrong" in name:
        return "armstrong"
    if "fibonacci" in name:
        return "fibonacci"
    return name

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
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
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
        return "unknown", "Invalid JSON output", 0.0

    # algo = data.get("algorithm", "").lower()
    algo = normalize_algorithm(data.get("algorithm", ""))

    conf = data.get("confidence", 0.0)

    if truth == "unknown":
        return "unknown", "No ground truth available", conf

    if algo == truth and conf >= 0.85:
    # if conf >= 0.85:
        return "correct", "Correct with high confidence", conf

    if algo == "cannot determine" or conf < 0.85:
        return "unknown", "Insufficient confidence", conf

    return "incorrect", f"Incorrect identification: {algo}", conf

# ===================== VERIFICATION =====================
def is_algorithmic(parsed):
    if not parsed:
        return False

    # algo = parsed.get("algorithm", "").lower()
    algo = normalize_algorithm(parsed.get("algorithm", ""))
    ops = " ".join(parsed.get("key_operations", [])).lower()

    # Explicitly reject trivial programs
    trivial_keywords = [
        "hello world",
        "print",
        "printf only",
        "no computation",
    ]

    if any(k in algo for k in trivial_keywords):
        return False

    # Must have at least one computation-related concept
    computation_keywords = [
        "loop", "recursion", "compute", "calculate",
        "compare", "sum", "count", "sort", "search",
        "check", "evaluate"
    ]

    return any(k in ops for k in computation_keywords)

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
            "obfuscated_file": out_path.replace("\\", "/"),
            "ground_truth": GROUND_TRUTH.get(prog, "unknown"),
            "runs": [],
        }

        resp = query_groq(code)
        classification, comment, confidence = classify_response(
            resp, entry["ground_truth"]
        )
        verified = confidence >= 0.85 and classification == "correct"

        parsed = None
        try:
            parsed = json.loads(clean_json(resp))
        except Exception:
            pass

        algorithmic = is_algorithmic(parsed)
        verified = (
                algorithmic and
                confidence >= 0.85 and
                classification == "correct"
        )

        if verified:
            consistency = "high"
        elif confidence >= 0.85 and algorithmic:
            consistency = "medium"
        else:
            consistency = "low"

        entry["runs"].append({
            "run": 1,
            "llm_raw_output": resp,
            "llm_parsed_output": parsed,
            "classification": classification,
            "confidence": confidence,
            "verified": verified,
            "comment": comment,
        })

        entry["summary"] = {
            "correct": "Yes" if classification == "correct" else "No",
            "incorrect": "Yes" if classification == "incorrect" else "No",
            "unknown": "Yes" if classification == "unknown" else "No",
            "algorithmic": "Yes" if algorithmic else "No",
            "verified": "Yes" if verified else "No",
            "consistency": consistency,
            "overall_classification": classification,
        }

        results.append(entry)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    now = datetime.now()

    final_output = {
        "experiment_metadata": {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "model": MODEL_NAME,
            "obfuscation": "Flatten",
            "runs_per_program": RUNS,
            "auto_verify": True,
            "environment": "x86_64 Linux GCC",
            "consistency_definition": {
                "high": "Algorithm is correct and verified (ground truth match)",
                "medium": "Algorithm identified with high confidence, but not fully verified",
                "low": "Low confidence, wrong, non-algorithmic, or invalid"
            }
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

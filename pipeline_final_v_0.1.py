import os
import subprocess
import json
import requests
import time
import re
from datetime import datetime
from groq import Groq, RateLimitError, APIStatusError
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x


# ===================== CONFIG =====================
PROGRAMS_DIR = "programs"
OBFUSCATED_DIR = "obfuscated"
RESULTS_DIR = "results"
GROUND_TRUTH_CACHE = "ground_truth_cache.json"

MODEL_NAME = "openai/gpt-oss-20b"
# TARGET_FUNCTION = "main"
RUNS = 1
MAX_CHARS = 6000
MAX_RETRIES = 3
RETRY_DELAY_SEC = 60
MIN_CONFIDENCE = 0.6

# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

def print_experiment_header(repo_info, num_files):
    print("\n" + "=" * 60)
    print("🧪  EXPERIMENT: Algorithm Identification under Obfuscation")
    print("=" * 60)
    print()
    print("▶ Obfuscation Technique : Tigress Control-Flow Flattening")
    print("▶ Benchmark Suite       : tum-i4/obfuscation-benchmarks")
    print(f"▶ Benchmark Path        : {repo_info['path']}")
    print(f"▶ Pinned Commit         : {repo_info['commit']}")
    print(f"▶ Total Programs        : {num_files}")
    print()
    print(f"▶ Analysis Model        : {MODEL_NAME} (Groq API)")
    print(f"▶ Runs per Program      : {RUNS}")
    print("▶ Verification Mode     : Automatic (Ground Truth Matching)")
    print()
    print("-" * 60)
    print("📦 Step 1: Ground Truth Preparation")
    print("-" * 60)


# ===================== GROUND TRUTH =====================
# GROUND_TRUTH = {
#     "anagram.c": "anagram",
#     "armstrong.c": "armstrong",
# }

def generate_ground_truth_cache(files, repo_info):
    ground_truth = {}

    for filename in files:
        base_name = filename.replace(".c", "").lower()
        algo = normalize_algorithm(base_name)

        # ALWAYS add — this is a benchmark
        ground_truth[filename] = {
            "algorithm": algo,
            "source": repo_info
        }

    return ground_truth

def fetch_github_file_list_at_commit(repo, commit, path):
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={commit}"
    # response = requests.get(url)
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return [item["name"] for item in data if item["type"] == "file"]


    # print("Files at pinned commit:")
    # for f in files:
    #     print(" -", f)

def normalize_algorithm(name: str) -> str:
    name = name.lower().strip()

    # ---------- String / text algorithms ----------
    if "anagram" in name:
        return "anagram"
    if "compare string" in name or "comparestrings" in name:
        return "comparestrings"
    if "concat" in name:
        return "concatstrings"
    if "palindrome" in name:
        return "palindrome"
    if "reverse" in name:
        return "reverse"
    if "vowel" in name:
        return "vowels"
    if "string to ascii" in name or "stringtoascii" in name:
        return "stringtoascii"
    if "print initial" in name or "printinitials" in name:
        return "printinitials"

    # ---------- Numeric / math algorithms ----------
    if "armstrong" in name:
        return "armstrong"
    if "factorial" in name:
        return "factorial"
    if "fibonacci" in name or name == "fib":
        return "fibonacci"
    if "prime" in name:
        return "prime"
    if "perfect" in name:
        return "perfect"
    if "gcd" in name or "greatest common divisor" in name:
        return "gcd"
    if "lcm" in name or "least common multiple" in name:
        return "lcm"
    if "sum" in name and "rec" in name:
        return "sumrec"
    if "min max" in name or "minmaxarray" in name:
        return "minmaxarray"
    if "multiplication table" in name or "multtable" in name:
        return "multtable"
    if "number to words" in name or "numtowords" in name:
        return "numtowords"
    if "random" in name:
        return "random"
    if "sum of cubes" in name or "digit power" in name:
        return "armstrong"

    # ---------- Searching algorithms ----------
    if "binary search" in name or "binarysearch" in name:
        return "binarysearch"
    if "linear search" in name or "linearsearch" in name:
        return "linearsearch"

    # ---------- Sorting algorithms ----------
    if "bubble sort" in name or "bubblesort" in name:
        return "bubblesort"
    if "insertion sort" in name or "insertionsort" in name:
        return "insertionsort"
    if "selection sort" in name or "selectionsort" in name:
        return "selectionsort"
    if "merge sort" in name or "mergesort" in name:
        return "mergesort"
    if "quick sort" in name or "quicksort" in name:
        return "quicksort"

    # ---------- Conversion algorithms ----------
    if "decimal to binary" in name or "decimaltobinary" in name:
        return "decimaltobinary"
    if "decimal to hex" in name or "decimaltohex" in name:
        return "decimaltohex"
    if "decimal to octal" in name or "decimaltooctal" in name:
        return "decimaltooctal"
    if "roman numeral" in name or "romannumerals" in name:
        return "romannumerals"
    if "binary addition" in name or "binary add" in name or "binaryadd" in name:
        return "binaryadd"
    if "binary multiplication" in name or "binary mult" in name or "binarymult" in name:
        return "binarymult"

    # ---------- Misc / control-flow / IO ----------
    if "floyd" in name:
        return "floyd"
    if "frequency" in name:
        return "frequency"
    if "file" in name or "iofile" in name:
        return "iofile"
    if "pyramid" in name:
        return "pyramid"

    # ---------- Fallback ----------
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
        "--Functions=*",
        "--Transform=CleanUp",
        "--CleanUpKinds=names,annotations,randomize,compress,noExterns,removeUnusedFunctions,removeLabelsAsValues",
        input_c,
        f"--out={output_c}",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # IMPORTANT: noMain removes main(), warnings are expected
    # Success criterion = output file exists
    return os.path.exists(output_c)

# ===================== CODE PREP ==================
def strip_comments(code):
    return "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("//")
    )

# def trim_to_entry(code, max_chars):
#     for key in [
#         "FUNCTION-DEF main",
#         "_TIG_FN_",
#         "int main"
#
#     ]:
#         idx = code.find(key)
#         if idx != -1:
#             return code[idx:idx + max_chars]
#     return code[-max_chars:]

def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

def strip_tigress_header(code: str) -> str:
    """
    Remove ONLY the initial Tigress banner comment.
    Do NOT remove comments globally.
    """
    if code.lstrip().startswith("/*"):
        end = code.find("*/")
        if end != -1:
            return code[end + 2:]
    return code


def extract_semantic_features(code: str):
    features = {
        "has_mod_10": False,
        "has_div_10": False,
        "has_digit_cube": False,
        "has_accumulator": False,
        "has_nested_loops": False,
        "has_swap": False,
        "has_adjacent_compare": False,
    }

    if re.search(r"%\s*10", code):
        features["has_mod_10"] = True

    if re.search(r"/=\s*10", code) or re.search(r"/\s*10", code):
        features["has_div_10"] = True

    if re.search(r"digit\s*\*\s*digit\s*\*\s*digit", code) or \
       re.search(r"pow\s*\(\s*digit\s*,\s*3", code):
        features["has_digit_cube"] = True

    if re.search(r"sum\s*\+=", code):
        features["has_accumulator"] = True

    if len(re.findall(r"for\s*\(", code)) >= 2 or \
       len(re.findall(r"while\s*\(", code)) >= 2:
        features["has_nested_loops"] = True

    if re.search(
        r"tmp\s*=\s*\w+.*?\w+\s*=\s*\w+.*?\w+\s*=\s*tmp",
        code,
        re.DOTALL
    ):
        features["has_swap"] = True

    if re.search(r"\w+\s*\[\s*\w+\s*\]\s*>\s*\w+\s*\[\s*\w+\s*\]", code):
        features["has_adjacent_compare"] = True

    return features

# ===================== GROQ =======================

def find_logic_start(code: str):
    """
    Prefer slicing near loops and switches (algorithmic core),
    not helper functions like __bswap.
    """
    markers = [
        "for (",
        "while (",
        "do {",
        "switch (",
        "case ",
    ]

    positions = [code.find(m) for m in markers if code.find(m) != -1]

    if positions:
        return max(0, min(positions) - 2000)

    # fallback: near first Tigress function
    tig = code.find("_TIG")
    if tig != -1:
        return max(0, tig - 2000)

    # final fallback
    return max(0, len(code) - MAX_CHARS)


def looks_truncated(code: str) -> bool:
    markers = ["_TIG", "switch", "while", "for", "goto"]
    return not any(m in code for m in markers)

def extract_relevant_slices(code: str):
    slices = []

    # 1) Slice near loops (algorithm core)
    loop_markers = ["for (", "while (", "do {"]
    for m in loop_markers:
        idx = code.find(m)
        if idx != -1:
            start = max(0, idx - 2000)
            slices.append(code[start:start + MAX_CHARS])
            break

    # 2) Slice near first Tigress function
    tig = code.find("_TIG")
    if tig != -1:
        start = max(0, tig - 2000)
        slices.append(code[start:start + MAX_CHARS])

    # 3) Slice from the end (bubble sort often lives here)
    slices.append(code[-MAX_CHARS:])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in slices:
        h = hash(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)

    return "\n\n/* ==== NEXT SLICE ==== */\n\n".join(unique)


def query_groq(code):
    # 🔥 KEEP ORIGINAL CODE FOR FEATURES
    full_code = strip_tigress_header(code)

    # 🔥 FEATURES MUST COME FROM FULL FILE
    full_features = extract_semantic_features(full_code)

    # 🔥 SLICES ONLY FOR LLM CONTEXT
    sliced_code = extract_relevant_slices(full_code)

    features = full_features

    prompt = f"""
You are an expert reverse engineer analyzing C code obfuscated using
Tigress CONTROL-FLOW FLATTENING only.

Important assumptions:
- Semantics are preserved exactly.
- No virtualization, no opaque predicates.
- Flattening introduces an artificial dispatcher/state variable.

Semantic feature analysis (reliable signals extracted statically):
{json.dumps(features, indent=2)}

Your task:
1) Identify the dispatcher variable.
2) Reconstruct the original control flow.
3) Identify the underlying algorithm implemented by the program logic. The original main() function may have been removed or renamed by Tigress CleanUp.



Rules:
- Ignore dispatcher mechanics.
- Focus on data flow, arithmetic, comparisons, and I/O.
- Base your decision on dominant data-flow patterns.
- If multiple algorithms are possible, choose the most likely one and lower confidence.
- Ignore helper functions such as __bswap*, __uint*_identity unless they are central.
- Focus on loops, arithmetic on input values, digit extraction, comparisons, and array indexing.

Feature interpretation hints:
- mod 10 + div 10 + digit cube + accumulator ⇒ Armstrong number
- nested loops + adjacent array comparison + swap ⇒ Bubble Sort
- nested loops + multiplication only ⇒ Multiplication Table
- string length checks or character array indexing ⇒ String algorithm (Anagram / Palindrome)



Output JSON ONLY:

{{
  "algorithm": string,
  "confidence": number,
  "dispatcher_variable": string,
  "key_operations": [string],
  "reconstructed_logic": string,
  "why_not_other_algorithms": string
}}

Analyze the following flattened C code:

<<<CODE
{sliced_code}
CODE
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except RateLimitError as e:
        msg = str(e)

        # Try to extract suggested wait time (if provided)
        wait_match = re.search(r"try again in ([0-9m\.s]+)", msg)
        wait_time = wait_match.group(1) if wait_match else "up to 24 hours"

        print("\n[Groq] ⚠️ DAILY TOKEN QUOTA EXCEEDED")
        print(f"[Groq] Model: {MODEL_NAME}")
        print(f"[Groq] Tokens per day limit reached.")
        print(f"[Groq] Suggested retry time: {wait_time}")
        print("[Groq] Please retry after token reset (typically 24 hours).")
        print("[Groq] Experiment stopped safely.\n")

        exit(1)

    except APIStatusError as e:
        print("\n[Groq] ⚠️ API status error")
        print(e)
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

    if algo == truth and conf >= MIN_CONFIDENCE:
        return "correct", "Correct with sufficient confidence", conf

    # hard reject multtable unless ground truth says so
    if algo == "multtable" and truth != "multtable":
        return "incorrect", "False positive multiplication table", conf

    if algo == "cannot determine" or conf < MIN_CONFIDENCE:
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
        "check", "evaluate",
        "modulo", "digit", "power", "accumulate",
        "multiply", "divide", "remainder"
    ]

    return any(k in ops for k in computation_keywords)

def load_or_create_ground_truth(files, repo_info, force=False):
    if os.path.exists(GROUND_TRUTH_CACHE) and not force:
        with open(GROUND_TRUTH_CACHE, "r") as f:
            print("Loaded cached ground truth.")
            return json.load(f)

    print("Creating ground truth cache from GitHub benchmark...")
    gt = generate_ground_truth_cache(files, repo_info)

    with open(GROUND_TRUTH_CACHE, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"Ground truth JSON file created successfully: {GROUND_TRUTH_CACHE}")
    print(f"Ground truth entries: {len(gt)}")

    return gt

# ===================== MAIN =======================
def main():
    results = []

    repo_info = {
        "repo": "tum-i4/obfuscation-benchmarks",
        "commit": "d11452ffb3ec7418a462f65d4034f9f1474136c8",
        "path": "basic-algorithms"
    }

    print("\n==================== EXPERIMENT START ====================")
    print("Transform: Tigress CONTROL-FLOW FLATTENING")
    print(f"Model: {MODEL_NAME}")
    print(f"Runs per program: {RUNS}")
    print("Benchmark: tum-i4/obfuscation-benchmarks (pinned commit)\n")

    # --- Ground truth from GitHub (cached to ground_truth_cache.json) ---
    print("📥 Fetching file list from GitHub benchmark...")
    files = fetch_github_file_list_at_commit(
        repo_info["repo"],
        repo_info["commit"],
        repo_info["path"]
    )

    print_experiment_header(repo_info, len(files))

    GROUND_TRUTH = load_or_create_ground_truth(files, repo_info, force=True)
    print(f"✅ Ground truth covers: {len(GROUND_TRUTH)} files\n")
    # --- Collect local programs to analyze ---
    programs = [p for p in os.listdir(PROGRAMS_DIR) if p.endswith(".c")]
    programs.sort()
    if not programs:
        print(f"⚠️  No .c files found in {PROGRAMS_DIR}")
        return

    print(f"📁 Local programs found: {len(programs)}\n")

    # --- Progress bar over programs ---
    for prog in tqdm(programs, desc="Analyzing programs", unit="file"):
        in_path = os.path.join(PROGRAMS_DIR, prog)
        out_name = prog.replace(".c", "_flatten.c")
        out_path = os.path.join(OBFUSCATED_DIR, "flatten", out_name)

        print(f"\n🧩 Program: {prog}")
        print("   🔐 Applying Tigress flattening...")

        if not run_tigress_flatten(in_path, out_path):
            print(f"   ❌ Flatten failed for {prog}")
            continue

        print("   ✅ Flattening done")
        print("   📄 Loading obfuscated code...")

        with open(out_path, "r") as f:
            code = f.read()

        gt = GROUND_TRUTH.get(prog, "unknown")
        truth_algo = gt["algorithm"] if isinstance(gt, dict) else gt

        entry = {
            "program": prog,
            "obfuscated_file": out_path.replace("\\", "/"),
            "ground_truth": gt,
            "runs": [],
        }

        # --- Runs loop (with mini progress text) ---
        for run_idx in range(1, RUNS + 1):
            print(f"   🤖 Querying Groq LLM... (run {run_idx}/{RUNS})")
            resp = query_groq(code)

            classification, comment, confidence = classify_response(resp, truth_algo)

            parsed = None
            try:
                parsed = json.loads(clean_json(resp))
            except Exception:
                pass

            algorithmic = is_algorithmic(parsed)
            verified = (
                    algorithmic and
                    confidence >= MIN_CONFIDENCE and
                    classification == "correct"
            )

            entry["runs"].append({
                "run": run_idx,
                "llm_raw_output": resp,
                "llm_parsed_output": parsed,
                "classification": classification,
                "confidence": confidence,
                "verified": verified,
                "comment": comment,
            })

        best_run = max(entry["runs"],key=lambda r: (r["confidence"] if isinstance(r["confidence"], (int, float)) else 0.0))
        best_classification = best_run["classification"]
        best_algorithmic = is_algorithmic(best_run["llm_parsed_output"])
        best_verified = best_run["verified"]

        if best_verified:
            consistency = "high"
        elif best_run["confidence"] >= MIN_CONFIDENCE and best_algorithmic:
            consistency = "medium"
        else:
            consistency = "low"


        best_algo = normalize_algorithm(
            best_run["llm_parsed_output"].get("algorithm", "")
            if best_run["llm_parsed_output"] else ""
        )
        truth_algo_norm = normalize_algorithm(truth_algo)
        conf = best_run["confidence"]

        with open(out_path, "r") as f:
            code = f.read()



        entry["summary"] = {
            "correct": "Yes" if best_classification == "correct" else "No",
            "incorrect": "Yes" if best_classification == "incorrect" else "No",
            "unknown": "Yes" if best_classification == "unknown" else "No",
            "algorithmic": "Yes" if best_algorithmic else "No",
            "verified": "Yes" if best_verified else "No",
            "consistency": consistency,
            "overall_classification": best_classification,
        }
        print( f"   📊 Result: classification={best_classification}, confidence={best_run['confidence']}, consistency={consistency}")
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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(RESULTS_DIR, f"results_flatten_{timestamp}.json")
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n==================== EXPERIMENT DONE ====================")
    print(f"✅ Results saved to: {out_file}\n")

# ===================== RUN ========================
if __name__ == "__main__":
    main()

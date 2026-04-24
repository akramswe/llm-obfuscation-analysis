import os
import json
import time
import re
import random
from datetime import datetime
from typing import Optional, Tuple
# from groq import Groq, RateLimitError, APIStatusError
from openai import OpenAI, RateLimitError, APIStatusError

# ===================== CONFIG =====================

MODELS = [
    # "openai/gpt-oss-20b" for groq
    # "openrouter/free"
    # "openrouter/hunter-alpha"
    # "xiaomi/mimo-v2-pro"
    #  "openai/gpt-5.4"
    # "nvidia/nemotron-3-super-120b-a12b:free"
    # "meta-llama/llama-3.1-8b-instruct",
    # "openai/gpt-4.1",
    # "qwen/qwen2.5-7b-instruct",
    # "meta-llama/llama-3.1-8b-instruct"
    # "qwen/qwen3.6-plus"
    "tencent/hy3-preview:free"
]

# MODELS = [
#     "openai/gpt-oss-20b",
#     "openai/gpt-oss-120b",
#     "llama-3.3-70b-versatile",
#     "llama-3.1-8b-instant",
#     "qwen/qwen3-32b"
# ]

CONSTRUCT_TYPES = [
    "direct",
    # "function_pointer"
]

# CONSTRUCT_TYPES = [
#     "direct",
#     "function_pointer",
#     "pointer_array",
#     "struct_pointer",
#     "switch_dispatch"
# ]

CALL_CHAIN_SIZES_BY_CONSTRUCT = {
    "direct": [1000],
    # "direct": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # "function_pointer": [100]
}

# CALL_CHAIN_SIZES_BY_CONSTRUCT = {
#     "direct": [100, 300, 500, 700, 900, 1100],
#     "function_pointer": [100, 200, 300, 400, 500],
#     "pointer_array": [100, 200, 300, 400, 500],
#     "struct_pointer": [100, 200, 300, 400, 500],
#     "switch_dispatch": [100, 200, 300, 400, 500],
# }

TEMPERATURE = 0.0
DELAY_BETWEEN_REQUESTS = 30
# DELAY_BETWEEN_REQUESTS = 0
BASE_RESULTS_DIR = "results_terminal"
TPM_SOFT_LIMIT = 12000
SHOW_DEBUG_LOGS = False
PREVIEW_LINES = 5
MAX_DISPLAY_CHARS = 160
MAX_REQUEST_ATTEMPTS = 5
REASONING_EFFORT = "none"
EXCLUDE_REASONING_TOKENS = True
PROMPT_VERSION = "v2_terminal_only"
SCHEMA_VERSION = "1.0"
EXPERIMENT_NAME = "terminal_function_identification"
PROVIDER_NAME = "openrouter"
MODEL_ROUTING_NOTE = "openrouter/free is a routing model; actual backend models may vary per request and are recorded in summary.json"
FAILURE_REASON_LABELS = {
    "prompt_too_large": "Prompt exceeds provider/model limits",
    "daily_token_limit": "Provider daily token quota reached",
    "insufficient_credits": "Insufficient credits for this request",
    "rate_limit": "Rate-limited by provider",
    "api_error": "API error from provider",
    "empty_response": "Model returned empty output",
    "malformed_output": "Output format violation (non-call text or malformed lines)",
    "underestimate": "Model output shorter than expected depth",
    "wrong_terminal_line": "Final line is incorrect (expected END line mismatch)",
    "exact_match_failure": "Output mismatch despite similar structure",
}

# # Use environment variable only
# GROQ_API_KEY = "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"
# # GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise RuntimeError("Please set GROQ_API_KEY as an environment variable.")

OPENROUTER_API_KEY = "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"

# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY as an environment variable.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        # Optional headers. You can remove them if you want.
        "X-OpenRouter-Title": "call-chain-experiment",
    },
)

# ===================== CODE GENERATORS =====================

def generate_direct_call_chain(n: int):
    rng = random.Random(n)
    order = list(range(1, n + 1))
    rng.shuffle(order)

    code = "#include <stdio.h>\n\n"

    for i in range(n - 1):
        code += f"void func{order[i]}(void){{ func{order[i+1]}(); }}\n"
    code += f"void func{order[-1]}(void){{ printf(\"End\\n\"); }}\n"
    code += f"int main(void){{ func{order[0]}(); return 0; }}\n"

    expected_lines = [
        f"func{order[i]} calls func{order[i+1]}" for i in range(n - 1)
    ]
    expected_lines.append(f"func{order[-1]} calls END")

    return code, expected_lines

def generate_function_pointer_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"

    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    for i in range(1, n):
        code += f"void (*fp{i})(void);\n"
    code += "\n"

    for i in range(1, n):
        code += f"void func{i}(void){{ fp{i}(); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    code += "int main(void){\n"
    for i in range(1, n):
        code += f"    fp{i} = func{i+1};\n"
    code += "    func1();\n"
    code += "    return 0;\n"
    code += "}\n"
    return code

def generate_pointer_array_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"
    code += f"#define N {n}\n"
    code += "typedef void (*fp_t)(void);\n"
    code += "fp_t fp[N + 1];\n\n"

    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    for i in range(1, n):
        code += f"void func{i}(void){{ fp[{i}](); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    code += "int main(void){\n"
    for i in range(1, n):
        code += f"    fp[{i}] = func{i+1};\n"
    code += f"    fp[{n}] = func{n};\n"
    code += "    func1();\n"
    code += "    return 0;\n"
    code += "}\n"
    return code

def generate_struct_pointer_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"
    code += "typedef void (*fp_t)(void);\n"
    code += "struct Node { fp_t next; };\n"
    code += f"struct Node nodes[{n + 1}];\n\n"

    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    for i in range(1, n):
        code += f"void func{i}(void){{ nodes[{i}].next(); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    code += "int main(void){\n"
    for i in range(1, n):
        code += f"    nodes[{i}].next = func{i+1};\n"
    code += f"    nodes[{n}].next = func{n};\n"
    code += "    func1();\n"
    code += "    return 0;\n"
    code += "}\n"
    return code

def generate_switch_dispatch_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"

    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    code += "void dispatch(int id){\n"
    code += "    switch(id){\n"
    for i in range(1, n):
        code += f"        case {i}: func{i+1}(); break;\n"
    code += "        default: break;\n"
    code += "    }\n"
    code += "}\n\n"

    for i in range(1, n):
        code += f"void func{i}(void){{ dispatch({i}); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    code += "int main(void){\n"
    code += "    func1();\n"
    code += "    return 0;\n"
    code += "}\n"
    return code

def generate_code(n: int, construct: str):
    generators = {
        "direct": generate_direct_call_chain,
        "function_pointer": generate_function_pointer_chain,
        "pointer_array": generate_pointer_array_chain,
        "struct_pointer": generate_struct_pointer_chain,
        "switch_dispatch": generate_switch_dispatch_chain,
    }
    if construct not in generators:
        raise ValueError(f"Unknown construct type: {construct}")
    generated = generators[construct](n)

    # direct returns (code, expected_lines); other constructs currently return code.
    if isinstance(generated, tuple) and len(generated) == 2:
        return generated

    code = generated
    expected_lines = [f"func{i} calls func{i+1}" for i in range(1, n)]
    expected_lines.append(f"func{n} calls END")
    return code, expected_lines

# ===================== PROMPT =====================

def build_prompt(code: str, depth: int) -> str:
    return f"""You are analyzing C code.

Your task is to reconstruct the COMPLETE linear function call chain starting from main().

Strict instructions:
- Follow the function calls step by step until the program terminates.
- Output EXACTLY one call per line.
- Use EXACT format: funcX calls funcY
- Even if the code calls printf("End\n"), you MUST output: funcX calls END
- NEVER output printf or any library function name
- Do NOT skip any function.
- Do NOT jump ahead.
- Do NOT repeat lines.
- Do NOT include main.
- Do NOT include explanations, comments, numbering, or code blocks.
- Output ONLY the call chain.
- Output must contain EXACTLY {depth} lines.
- If you are unsure, continue the sequence step-by-step logically without skipping or stopping early.
- NEVER return empty output.
- NEVER stop early.
- You MUST output ALL {depth} lines.

Example:
func10 calls func3
func3 calls func88
func88 calls END

C CODE:
<<<CODE
{code}
CODE
"""
# - The final line MUST be: func{depth} calls END
# ===================== ESTIMATION =====================

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def normalize_response(response: Optional[str]) -> str:
    if not response:
        return ""

    text = response

    # --- Fix missing spaces ---
    text = re.sub(r'(func\d+)calls', r'\1 calls', text)
    text = re.sub(r'calls(func\d+|END)', r'calls \1', text)

    # Fix models that output printf("End\n") instead of END
    text = re.sub(
        r'func(\d+)\s+calls\s+printf\("End\\n"\)',
        r'func\1 calls END',
        text
    )

    # --- Fix merged lines like:
    # func402 calls func624func624 calls func888
    text = re.sub(
        r'(func\d+\s+calls\s+func\d+)(?=func\d+\s+calls)',
        r'\1\n',
        text
    )

    # --- Normalize lines ---
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)

# ===================== EVALUATION =====================

def classify_failure(response: Optional[str], expected_lines: list[str], error: Optional[str]) -> Optional[str]:
    if error is not None:
        error_lower = error.lower()

        if "prompt_too_large" in error_lower:
            return "prompt_too_large"
        if "daily_token_limit" in error_lower or "tokens per day" in error_lower:
            return "daily_token_limit"
        if "insufficient_credits" in error_lower:
            return "insufficient_credits"
        if "rate_limit" in error_lower:
            return "rate_limit"
        return "api_error"

    if verify_response(response, expected_lines):
        return None

    if response is None or not response.strip():
        return "empty_response"

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return "empty_response"

    if any(not re.fullmatch(r"func\d+\s+calls\s+(func\d+|END)", ln) for ln in lines):
        return "malformed_output"

    last_line = lines[-1]

    if " calls " not in last_line:
        return "malformed_output"

    if len(lines) != len(expected_lines):
        return "underestimate"

    if last_line != expected_lines[-1]:
        return "wrong_terminal_line"

    return "exact_match_failure"

def verify_response(response: Optional[str], expected_lines: list[str]) -> bool:
    if not response:
        return False

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    return lines == expected_lines

# ===================== API =====================

def compute_max_tokens(depth: int) -> int:
    # Output is exactly `depth` lines in form "funcX calls funcY".
    # Keep enough room for deep chains (e.g., depth=1000) while avoiding runaway costs.
    return max(512, min(8000, depth * 8))

def call_with_retries(client,model: str,prompt: str,depth: int,retries: int = 5 ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[str]]:
    last_err = None

    for attempt in range(retries):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=compute_max_tokens(depth),
                extra_body={
                    "reasoning": {
                        "effort": REASONING_EFFORT,
                        "exclude": EXCLUDE_REASONING_TOKENS
                    }
                },
            )
            latency = time.perf_counter() - start
            text = resp.choices[0].message.content.strip() if resp.choices[0].message.content else ""
            # 🚨 Reject empty or too short responses
            if not text:
                print("⚠️ Empty response from API.")
                raise Exception("Empty response")

            lines = text.splitlines()
            if len(lines) < 5:  # basic sanity check
                print("⚠️ Too short response from API.")
                raise Exception("Too short")

            actual_model = getattr(resp, "model", None)
            return text, None, None, latency, actual_model

        except RateLimitError as e:
            msg = str(e)
            last_err = msg
            if SHOW_DEBUG_LOGS:
                print(f"DEBUG RateLimitError attempt {attempt + 1}/{retries}: {msg}")

            err_lower = msg.lower()

            if "tokens per day" in err_lower:
                return None, "daily_token_limit", msg, None, None

            if "request too large" in err_lower or ("limit" in err_lower and "requested" in err_lower):
                return None, "prompt_too_large", msg, None, None

            if attempt < retries - 1:
                sleep_time = (2 ** attempt) * 10
                # sleep_time = 5
                if SHOW_DEBUG_LOGS:
                    print(f"DEBUG sleeping {sleep_time}s before retry...")
                time.sleep(sleep_time)
            else:
                return None, "rate_limit", msg, None, None

        except APIStatusError as e:
            msg = str(e)
            if SHOW_DEBUG_LOGS:
                print(f"DEBUG APIStatusError attempt {attempt + 1}/{retries}: {msg}")

            msg_lower = msg.lower()

            if "requires more credits" in msg_lower or "can only afford" in msg_lower:
                return None, "insufficient_credits", msg, None, None

            if "tokens per day" in msg_lower:
                return None, "daily_token_limit", msg, None, None

            if "request too large" in msg_lower or ("limit" in msg_lower and "requested" in msg_lower):
                return None, "prompt_too_large", msg, None, None

            if "rate_limit_exceeded" in msg_lower or "rate limit" in msg_lower:
                if attempt < retries - 1:
                    sleep_time = (2 ** attempt) * 10
                    if SHOW_DEBUG_LOGS:
                        print(f"DEBUG sleeping {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                    continue
                return None, "rate_limit", msg, None, None

            return None, "api_error", msg, None, None

        except Exception as e:
            msg = str(e)
            if SHOW_DEBUG_LOGS:
                print(f"DEBUG UnexpectedError attempt {attempt + 1}/{retries}: {msg}")
            return None, "api_error", msg, None, None

    return None, "rate_limit" if last_err else "api_error", last_err, None, None

# ===================== UTIL =====================

def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def shorten_for_display(value: Optional[str], max_chars: int = MAX_DISPLAY_CHARS) -> str:
    if not value:
        return "None"
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."

def format_failure_reason(reason: Optional[str]) -> str:
    if not reason:
        return "none"
    return FAILURE_REASON_LABELS.get(reason, f"Unknown failure (see raw error details): {reason}")

def explain_retry_reason(err_type: Optional[str], err_msg: Optional[str]) -> str:
    msg = (err_msg or "").lower()
    if "empty response" in msg:
        return "the previous attempt returned an empty response"
    if "too short" in msg:
        return "the previous attempt returned output that was too short"
    if err_type:
        return format_failure_reason(err_type)
    return "the previous attempt returned empty/invalid output"

def print_sample_report(
    run_id: str,
    model: str,
    construct: str,
    depth: int,
    est_prompt_tokens: int,
    err_type: Optional[str],
    actual_model: Optional[str],
    latency: Optional[float],
    success: bool,
    last_call: Optional[str],
    predicted_last_func: Optional[str],
    failure_reason: Optional[str],
    failure_analysis: dict,
    normalized_text: str,
) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print("=" * 60)
    print(f"Run ID   : {run_id}")
    print(f"Model    : {model}")
    print(f"Construct: {construct}")
    print(f"Depth    : {depth}")
    print("=" * 60)
    print()
    print(f"[{ts}] Request finished")
    print(f"Prompt tokens (est): {est_prompt_tokens}")
    print(f"API status         : {'success' if err_type is None else err_type}")
    print(f"Requested model    : {model}")
    print(f"Actual model       : {actual_model}")
    print(f"Latency            : {latency:.2f}s" if latency is not None else "Latency            : n/a")
    print()
    print("Evaluation")
    all_lines = [ln.strip() for ln in normalized_text.splitlines() if ln.strip()]
    call_lines = [ln for ln in all_lines if re.fullmatch(r"func\d+\s+calls\s+(func\d+|END)", ln)]
    non_call_lines = len(all_lines) - len(call_lines)
    display_last_call = call_lines[-1] if call_lines else last_call
    display_predicted_func = predicted_last_func if predicted_last_func and re.fullmatch(r"func\d+", predicted_last_func) else None

    print(f"- Success          : {'YES' if success else 'NO'}")
    print(f"- Last call        : {shorten_for_display(display_last_call)}")
    print(f"- Predicted func   : {display_predicted_func if display_predicted_func else 'None'}")
    print(f"- Failure reason   : {format_failure_reason(None if success else failure_reason)}")
    print(f"- Correct lines    : {failure_analysis.get('last_correct_line', 0)}/{depth}")
    if non_call_lines > 0:
        print(f"- Non-call lines   : {non_call_lines}")
    print()

    if call_lines:
        print(f"Preview (valid call-chain lines, first {min(PREVIEW_LINES, len(call_lines))})")
        for i, line in enumerate(call_lines[:PREVIEW_LINES], start=1):
            print(f"{i}. {line}")
        if len(call_lines) > PREVIEW_LINES:
            print("...")
            print(f"{len(call_lines)}. {call_lines[-1]}")
        if non_call_lines > 0:
            print(f"(Skipped {non_call_lines} non-call lines in preview)")
        print()
    elif all_lines:
        print("Preview")
        print("No valid call-chain lines found in model output.")
        print()

# ===================== RESPONSE PARSING =====================

def extract_last_line(response: Optional[str]) -> Optional[str]:
    if response is None or not response.strip():
        return None

    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if not lines:
        return None

    return lines[-1]

def extract_predicted_func(response: Optional[str]) -> Optional[str]:
    last_line = extract_last_line(response)
    if not last_line or " calls " not in last_line:
        return None
    return last_line.split(" calls ")[0].strip()

def extract_last_call(response: Optional[str]) -> Optional[str]:
    if response is None or not response.strip():
        return None
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]

def analyze_failure_point(response: Optional[str], expected_lines: list[str], error: Optional[str] = None):
    if error is not None:
        return {
            "first_wrong_line": None,
            "last_correct_line": 0,
            "last_correct_call": None,
            "failure_started_at": None,
            "failure_reason": error
        }

    if response is None or not response.strip():
        return {
            "first_wrong_line": 1,
            "last_correct_line": 0,
            "last_correct_call": None,
            "failure_started_at": expected_lines[0].split(" calls ")[0] if expected_lines else None,
            "failure_reason": "empty_response"
        }

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    last_correct_line = 0
    last_correct_call = None

    for i, expected in enumerate(expected_lines, start=1):
        if i > len(lines):
            return {
                "first_wrong_line": i,
                "last_correct_line": last_correct_line,
                "last_correct_call": last_correct_call,
                "failure_started_at": expected.split(" calls ")[0],
                "failure_reason": "truncated_or_stopped_early"
            }

        if lines[i - 1] != expected:
            return {
                "first_wrong_line": i,
                "last_correct_line": last_correct_line,
                "last_correct_call": last_correct_call,
                "failure_started_at": expected.split(" calls ")[0],
                "failure_reason": "incorrect_line"
            }

        last_correct_line = i
        last_correct_call = lines[i - 1]

    return {
        "first_wrong_line": None,
        "last_correct_line": len(expected_lines),
        "last_correct_call": lines[-1] if lines else None,
        "failure_started_at": None,
        "failure_reason": None
    }

def explain_success(response: Optional[str], expected_lines: list[str]) -> Optional[str]:
    if verify_response(response, expected_lines):
        return "exact_match"
    return None


def explain_output_failure(response: Optional[str], expected_lines: list[str]) -> Optional[str]:
    if response is None or not response.strip():
        return "empty_output"

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    for i, expected in enumerate(expected_lines, start=1):
        if i > len(lines):
            return f"stopped_early_at_line_{i}"

        actual = lines[i - 1]

        if actual != expected:
            if " calls " not in actual:
                return f"malformed_line_{i}"

            left, right = actual.split(" calls ", 1)
            expected_left, expected_right = expected.split(" calls ", 1)

            if right != expected_right:
                return f"wrong_callee_line_{i}_expected_{expected_right}_got_{right}"

            if left != expected_left:
                return f"wrong_caller_line_{i}"

            return f"incorrect_line_{i}"

    if len(lines) > len(expected_lines):
        return "too_many_lines"

    return None

# ===================== EXPERIMENT =====================

def run_experiment():
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_root = os.path.join(BASE_RESULTS_DIR, f"run_{run_id}")
    raw_root = os.path.join(run_root, "raw")
    summaries_root = os.path.join(run_root, "summaries")

    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(summaries_root, exist_ok=True)

    config = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": EXPERIMENT_NAME,
        "models": MODELS,
        "note": MODEL_ROUTING_NOTE,
        "construct_types": CONSTRUCT_TYPES,
        "depths_by_construct": CALL_CHAIN_SIZES_BY_CONSTRUCT,
        "temperature": TEMPERATURE,
        "delay_between_requests": DELAY_BETWEEN_REQUESTS,
        "tpm_soft_limit": TPM_SOFT_LIMIT,
        "prompt_version": PROMPT_VERSION,
    }
    save_json(os.path.join(run_root, "config.json"), config)

    # client = Groq(api_key=GROQ_API_KEY)

    flat_results = []
    summary = []
    total_tasks = sum(len(CALL_CHAIN_SIZES_BY_CONSTRUCT[c]) for c in CONSTRUCT_TYPES) * len(MODELS)
    current_task = 0

    stop_all = False

    for model in MODELS:
        print(f"\n===== MODEL: {model} =====")
        model_dir = os.path.join(raw_root, model.replace("/", "_").replace(":", "_"))
        os.makedirs(model_dir, exist_ok=True)

        for construct in CONSTRUCT_TYPES:
            print(f"--- Construct: {construct} ---")
            construct_dir = os.path.join(model_dir, construct)
            os.makedirs(construct_dir, exist_ok=True)

            max_success_depth = 0
            first_failure_depth = None
            tested_depths = 0
            successes = 0

            depths = CALL_CHAIN_SIZES_BY_CONSTRUCT[construct]

            for depth in depths:
                tested_depths += 1
                current_task += 1
                print(f"\nProgress: {current_task}/{total_tasks} | testing depth {depth}")

                code, expected_lines = generate_code(depth, construct)
                prompt = build_prompt(code, depth)
                est_prompt_tokens = approx_tokens(prompt)
                print(
                    f"Code generation successful: created call-chain test with {depth} functions "
                    f"and {len(expected_lines)} expected call lines."
                )
                print("Prompt prepared successfully. Ready to call API.")

                text = None
                err_type = None
                err_msg = None
                latency = None
                actual_model = None

                if est_prompt_tokens > TPM_SOFT_LIMIT:
                    print(
                        f"WARNING: estimated prompt tokens {est_prompt_tokens} exceed soft limit {TPM_SOFT_LIMIT}, sending anyway..."
                    )

                print("Sending request to API...")
                # text, err_type, err_msg, latency, actual_model = call_with_retries(client, model, prompt, retries=5)
                for attempt_idx in range(1, MAX_REQUEST_ATTEMPTS + 1):  # retry until valid
                    print(
                        f"Attempt {attempt_idx}/{MAX_REQUEST_ATTEMPTS}: "
                        f"requesting call-chain output from model={model}, construct={construct}, depth={depth}"
                    )
                    text, err_type, err_msg, latency, actual_model = call_with_retries(client, model, prompt, depth, retries=1)
                    if text and text.strip():
                        break
                    if err_type in {"insufficient_credits", "prompt_too_large", "daily_token_limit"}:
                        print(
                            "Stopping retries due to non-retriable error: "
                            f"{format_failure_reason(err_type)}"
                        )
                        break
                    if attempt_idx < MAX_REQUEST_ATTEMPTS:
                        reason_text = explain_retry_reason(err_type, err_msg)
                        print(
                            f"Retrying ({attempt_idx + 1}/{MAX_REQUEST_ATTEMPTS}) because "
                            f"{reason_text}."
                        )
                print("Response received.")

                if text is None:
                    text = ""

                normalized_text = normalize_response(text)
                normalized_lines = [ln.strip() for ln in normalized_text.splitlines() if ln.strip()]
                line_count = len(normalized_lines)
                ended_with_end = bool(normalized_lines and re.fullmatch(r"func\d+\s+calls\s+END", normalized_lines[-1]))

                success = verify_response(normalized_text, expected_lines)
                success_reason = explain_success(normalized_text, expected_lines) if success else None

                detailed_output_reason = (
                    explain_output_failure(normalized_text, expected_lines)
                    if not success and err_type is None
                    else None
                )
                failure_reason = classify_failure(normalized_text, expected_lines, err_type)
                last_call = extract_last_call(normalized_text)
                predicted_last_func = extract_predicted_func(normalized_text)
                failure_analysis = analyze_failure_point(normalized_text, expected_lines, err_type)

                if SHOW_DEBUG_LOGS:
                    print(f"DEBUG: est_prompt_tokens={est_prompt_tokens}, err_type={err_type}, err_msg={err_msg}")
                    print(f"DEBUG: requested_model={model}, actual_model={actual_model}")
                    print(f"DEBUG: raw_text={repr(text[:200])}")
                    print(f"DEBUG: normalized_text={repr(normalized_text[:200])}")
                    print(f"DEBUG: last_call={last_call}")
                    print(f"DEBUG: predicted_last_func={predicted_last_func}")
                    print(f"DEBUG: failure_reason={None if success else failure_reason}")
                    print(f"DEBUG: failure_analysis={failure_analysis}")

                if success:
                    successes += 1
                    max_success_depth = depth
                elif first_failure_depth is None:
                    first_failure_depth = depth

                expected_last_func = expected_lines[-1].split(" calls ")[0]

                if success:
                    failure_type = None
                    failure_explanation = "Successful execution"
                elif err_type is not None:
                    failure_type = "api_failure"
                    failure_explanation = f"API failure occurred: {err_type}"
                else:
                    failure_type = "output_failure"
                    if failure_analysis.get("failure_reason") == "truncated_or_stopped_early":
                        failure_explanation = (
                            f"Model stopped early after line {failure_analysis.get('last_correct_line')}"
                        )
                    elif failure_analysis.get("failure_reason") == "incorrect_line":
                        failure_explanation = (
                            f"Model produced an incorrect line starting at line {failure_analysis.get('first_wrong_line')}"
                        )
                    elif failure_reason == "malformed_output":
                        failure_explanation = "Model produced malformed output"
                    elif failure_reason == "empty_response":
                        failure_explanation = "Model returned an empty response"
                    else:
                        failure_explanation = "Model output did not match the expected call chain"

                error_info = {
                    "api_reason": "none" if err_type is None else err_type,
                    "api_message": "API call successful" if err_type is None else err_msg
                }

                result_data = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "experiment": {
                        "name": EXPERIMENT_NAME,
                        "task": "Reconstruct the complete linear function call chain from main in generated C code",
                        "output_format": "funcX calls funcY per line"
                    },
                    "model": {
                        "provider": PROVIDER_NAME,
                        "requested_name": model,
                        "actual_name": actual_model,
                        "temperature": TEMPERATURE
                    },
                    "sample": {
                        "construct": construct,
                        "depth": depth,
                        "code_chars": len(code),
                        "estimated_prompt_tokens": est_prompt_tokens
                    },
                    "request": {
                        "prompt_version": PROMPT_VERSION,
                        "prompt_chars": len(prompt)
                    },
                    "response": {
                        "text": text,
                        "normalized_text": normalized_text,
                        "line_count": line_count,
                        "ended_with_end": ended_with_end,
                        "response_chars": len(text),
                        "response_tokens_est": approx_tokens(text) if text else 0,
                        "latency_seconds": latency
                    },
                    "evaluation": {
                        "expected_depth": depth,
                        "expected_last_func": expected_last_func,
                        "predicted_last_func": predicted_last_func,
                        "success": success,
                        # "success_reason": success_reason,
                        "result": "success" if success else "failure",
                        "success_reason": success_reason,
                        "failure_type": failure_type,
                        "failure_reason": None if success else failure_reason,
                        "failure_reason_readable": None if success else format_failure_reason(failure_reason),
                        # "failure_explanation": failure_explanation,
                        "failure_explanation": None if success else failure_explanation,
                        "detailed_output_reason": detailed_output_reason,
                        "failure_analysis": failure_analysis,
                        "first_wrong_line": failure_analysis.get("first_wrong_line"),
                        "last_correct_line": failure_analysis.get("last_correct_line", 0)
                    },
                    "error": error_info
                }

                out_path = os.path.join(construct_dir, f"depth_{depth}.json")
                save_json(out_path, result_data)
                flat_results.append(result_data)

                print_sample_report(
                    run_id=run_id,
                    model=model,
                    construct=construct,
                    depth=depth,
                    est_prompt_tokens=est_prompt_tokens,
                    err_type=err_type,
                    actual_model=actual_model,
                    latency=latency,
                    success=success,
                    last_call=last_call,
                    predicted_last_func=predicted_last_func,
                    failure_reason=failure_reason,
                    failure_analysis=failure_analysis,
                    normalized_text=normalized_text,
                )

                if err_type == "daily_token_limit":
                    stop_all = True
                    break

                print(f"Saved to: {out_path}")
                print(f"Next request in {DELAY_BETWEEN_REQUESTS}s...\n")
                time.sleep(DELAY_BETWEEN_REQUESTS)

            summary.append({
                "model": model,
                "construct": construct,
                "tested_depths": tested_depths,
                "successes": successes,
                "success_rate": successes / tested_depths if tested_depths else 0.0,
                "max_success_depth": max_success_depth,
                "first_failure_depth": first_failure_depth
            })

            if stop_all:
                break

        if stop_all:
            break

    def build_clean_summary(run_id: str, summary: list, flat_results: list):
        total_tests = len(flat_results)
        total_successes = sum(1 for r in flat_results if r["evaluation"]["success"])
        total_failures = total_tests - total_successes

        models_summary = {}
        construct_summary = {}

        for r in flat_results:
            requested_model = r["model"]["requested_name"]
            actual_model = r["model"]["actual_name"]
            construct = r["sample"]["construct"]
            depth = r["sample"]["depth"]
            success = r["evaluation"]["success"]
            success_reason = r["evaluation"].get("success_reason")
            failure_type = r["evaluation"]["failure_type"]
            failure_reason = r["evaluation"].get("failure_reason")
            failure_explanation = r["evaluation"].get("failure_explanation")
            failure_analysis = r["evaluation"]["failure_analysis"]
            response_info = r.get("response", {})
            error_info = r.get("error", {})

            if requested_model not in models_summary:
                models_summary[requested_model] = {
                    "requested_model": requested_model,
                    "actual_models_seen": [],
                    "num_tests": 0,
                    "num_successes": 0,
                    "success_rate": 0.0,
                    "samples": [],
                    "constructs": {}
                }

            model_entry = models_summary[requested_model]
            model_entry["num_tests"] += 1
            if success:
                model_entry["num_successes"] += 1

            if actual_model and actual_model not in model_entry["actual_models_seen"]:
                model_entry["actual_models_seen"].append(actual_model)

            if construct not in model_entry["constructs"]:
                model_entry["constructs"][construct] = {
                    "num_tests": 0,
                    "num_successes": 0,
                    "success_rate": 0.0,
                    "max_success_depth_tested": 0,
                    "first_failure_depth": None,
                    "actual_models_seen": []
                }

            construct_entry = model_entry["constructs"][construct]
            construct_entry["num_tests"] += 1

            if success:
                construct_entry["num_successes"] += 1
                construct_entry["max_success_depth_tested"] = max(construct_entry["max_success_depth_tested"], depth)
            elif construct_entry["first_failure_depth"] is None:
                construct_entry["first_failure_depth"] = depth

            if actual_model and actual_model not in construct_entry["actual_models_seen"]:
                construct_entry["actual_models_seen"].append(actual_model)

            if construct not in construct_summary:
                construct_summary[construct] = {
                    "num_tests": 0,
                    "num_successes": 0,
                    "success_rate": 0.0
                }
            construct_summary[construct]["num_tests"] += 1
            if success:
                construct_summary[construct]["num_successes"] += 1

            sample_entry = {
                "construct": construct,
                "depth": depth,
                "requested_model": requested_model,
                "actual_model": actual_model,
                "is_exact_match": success,
                "result": "success" if success else "failure",
                "line_count": response_info.get("line_count"),
                "ended_with_end": response_info.get("ended_with_end"),
                "api_reason": error_info.get("api_reason"),
                "api_message": error_info.get("api_message"),
            }
            if success:
                sample_entry["success_reason"] = success_reason
            else:
                sample_entry["failure_type"] = failure_type
                sample_entry["failure_reason"] = failure_reason
                sample_entry["failure_reason_readable"] = r["evaluation"].get("failure_reason_readable")
                sample_entry["failure_explanation"] = failure_explanation
                sample_entry["failure_point"] = {
                    "first_wrong_line": failure_analysis.get("first_wrong_line"),
                    "last_correct_line": failure_analysis.get("last_correct_line"),
                    "last_correct_call": failure_analysis.get("last_correct_call"),
                    "failure_started_at": failure_analysis.get("failure_started_at"),
                    "failure_reason": failure_analysis.get("failure_reason")
                }
            model_entry["samples"].append(sample_entry)

        for requested_model, model_entry in models_summary.items():
            model_entry["success_rate"] = round(
                model_entry["num_successes"] / model_entry["num_tests"], 3
            ) if model_entry["num_tests"] else 0.0

            for construct, construct_entry in model_entry["constructs"].items():
                construct_entry["success_rate"] = round(
                    construct_entry["num_successes"] / construct_entry["num_tests"], 3
                ) if construct_entry["num_tests"] else 0.0

            model_entry["samples"].sort(key=lambda x: (x["construct"], x["depth"]))

        for construct, entry in construct_summary.items():
            entry["success_rate"] = round(
                entry["num_successes"] / entry["num_tests"], 3
            ) if entry["num_tests"] else 0.0

        return {
            "meta": {
                "run_id": run_id,
                "experiment": EXPERIMENT_NAME,
                "provider": PROVIDER_NAME,
                "timestamp": datetime.now().isoformat()
            },
            "schema_hint": {
                "overall": "Run-level metrics across all tested models and constructs.",
                "models": "Per-model breakdown with concise sample-level outcomes.",
                "samples": "Each sample is one (model, construct, depth) evaluation."
            },
            "overall": {
                "total_tests": total_tests,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "success_rate": round(total_successes / total_tests, 3) if total_tests else 0.0,
                "construct_summary": construct_summary
            },
            "models": models_summary
        }

    clean_summary = build_clean_summary(run_id, summary, flat_results)

    save_json(os.path.join(summaries_root, "summary.json"), clean_summary)

    print("\nExperiment complete.")
    print(f"Results saved under: {run_root}")

if __name__ == "__main__":
    run_experiment()

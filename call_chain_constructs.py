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
    "openrouter/free"
    # "openrouter/hunter-alpha"
    # "xiaomi/mimo-v2-pro"
    #  "openai/gpt-5.4"
    # "nvidia/nemotron-3-super-120b-a12b:free",
    # "meta-llama/llama-3.1-8b-instruct",
    # "qwen/qwen2.5-7b-instruct"
]

# MODELS = [
#     "openai/gpt-oss-20b",
#     "openai/gpt-oss-120b",
#     "llama-3.3-70b-versatile",
#     "llama-3.1-8b-instant",
#     "qwen/qwen3-32b"
# ]

CONSTRUCT_TYPES = [
    "direct"
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
    "direct": [100, 1000]
    # "function_pointer": [100],
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
BASE_RESULTS_DIR = "results_terminal"
TPM_SOFT_LIMIT = 12000
PROMPT_VERSION = "v2_terminal_only"
SCHEMA_VERSION = "1.0"
EXPERIMENT_NAME = "terminal_function_identification"
PROVIDER_NAME = "openrouter"
MODEL_ROUTING_NOTE = "openrouter/free is a routing model; actual backend models may vary per request and are recorded in summary.json"

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
    return generators[construct](n)

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

def call_with_retries(client,model: str,prompt: str,retries: int = 5 ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[str]]:
    last_err = None

    for attempt in range(retries):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            latency = time.perf_counter() - start
            text = resp.choices[0].message.content.strip() if resp.choices[0].message.content else ""
            actual_model = getattr(resp, "model", None)
            return text, None, None, latency, actual_model

        except RateLimitError as e:
            msg = str(e)
            last_err = msg
            print(f"DEBUG RateLimitError attempt {attempt + 1}/{retries}: {msg}")

            err_lower = msg.lower()

            if "tokens per day" in err_lower:
                return None, "daily_token_limit", msg, None, None

            if "request too large" in err_lower or ("limit" in err_lower and "requested" in err_lower):
                return None, "prompt_too_large", msg, None, None

            if attempt < retries - 1:
                sleep_time = (2 ** attempt) * 10
                print(f"DEBUG sleeping {sleep_time}s before retry...")
                time.sleep(sleep_time)
            else:
                return None, "rate_limit", msg, None, None

        except APIStatusError as e:
            msg = str(e)
            print(f"DEBUG APIStatusError attempt {attempt + 1}/{retries}: {msg}")

            msg_lower = msg.lower()

            if "tokens per day" in msg_lower:
                return None, "daily_token_limit", msg, None, None

            if "request too large" in msg_lower or ("limit" in msg_lower and "requested" in msg_lower):
                return None, "prompt_too_large", msg, None, None

            if "rate_limit_exceeded" in msg_lower or "rate limit" in msg_lower:
                if attempt < retries - 1:
                    sleep_time = (2 ** attempt) * 10
                    print(f"DEBUG sleeping {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                    continue
                return None, "rate_limit", msg, None, None

            return None, "api_error", msg, None, None

        except Exception as e:
            msg = str(e)
            print(f"DEBUG UnexpectedError attempt {attempt + 1}/{retries}: {msg}")
            return None, "api_error", msg, None, None

    return None, "rate_limit" if last_err else "api_error", last_err, None, None

# ===================== UTIL =====================

def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
                # print(f"Testing depth {depth}")
                current_task += 1
                print(f"\n📊 Progress: {current_task}/{total_tasks}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing depth {depth}")

                code, expected_lines = generate_code(depth, construct)
                prompt = build_prompt(code, depth)
                est_prompt_tokens = approx_tokens(prompt)

                text = None
                err_type = None
                err_msg = None
                latency = None
                actual_model = None

                if est_prompt_tokens > TPM_SOFT_LIMIT:
                    print(
                        f"WARNING: estimated prompt tokens {est_prompt_tokens} exceed soft limit {TPM_SOFT_LIMIT}, sending anyway..."
                    )

                print("🚀 Sending request to API...")
                print(f"DEBUG: starting API call for depth {depth}")
                text, err_type, err_msg, latency, actual_model = call_with_retries(client, model, prompt, retries=5)
                print(f"DEBUG: finished API call for depth {depth}")
                print("✅ Response received")

                if text is None:
                    text = ""

                normalized_text = normalize_response(text)

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

                expected_func = expected_lines[-1].split(" calls ")[0]

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
                        "response_chars": len(text),
                        "response_tokens_est": approx_tokens(text) if text else 0,
                        "latency_seconds": latency
                    },
                    "evaluation": {
                        "expected_depth": expected_func,
                        "predicted_last_func": predicted_last_func,
                        "success": success,
                        # "success_reason": success_reason,
                        "result": "success" if success else "failure",
                        "success_reason": success_reason,
                        "failure_type": failure_type,
                        "failure_reason": None if success else failure_reason,
                        # "failure_explanation": failure_explanation,
                        "failure_explanation": None if success else failure_explanation,
                        "detailed_output_reason": detailed_output_reason,
                        "failure_analysis": failure_analysis
                    },
                    "error": error_info
                }

                out_path = os.path.join(construct_dir, f"depth_{depth}.json")
                save_json(out_path, result_data)
                flat_results.append(result_data)

                if success:
                    print(f"Result: success=True | last_call={last_call}")
                else:
                    print(f"Result: success=False ({failure_reason}) | expected=randomized_chain, last_call={last_call}")
                if err_type == "daily_token_limit":
                    stop_all = True
                    break

                print(f"⏳ Waiting {DELAY_BETWEEN_REQUESTS}s before next request...\n")
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

        for r in flat_results:
            requested_model = r["model"]["requested_name"]
            actual_model = r["model"]["actual_name"]
            construct = r["sample"]["construct"]
            depth = r["sample"]["depth"]
            success = r["evaluation"]["success"]
            failure_type = r["evaluation"]["failure_type"]
            failure_reason = r["evaluation"].get("failure_reason")
            failure_explanation = r["evaluation"].get("failure_explanation")
            failure_analysis = r["evaluation"]["failure_analysis"]
            error_info = r.get("error", {})

            if requested_model not in models_summary:
                models_summary[requested_model] = {
                    "requested_model": requested_model,
                    "actual_models_seen": [],
                    "total_tests": 0,
                    "total_successes": 0,
                    "success_rate": 0.0,
                    "test_points": [],
                    "constructs": {}
                }

            model_entry = models_summary[requested_model]
            model_entry["total_tests"] += 1
            if success:
                model_entry["total_successes"] += 1

            if actual_model and actual_model not in model_entry["actual_models_seen"]:
                model_entry["actual_models_seen"].append(actual_model)

            if construct not in model_entry["constructs"]:
                model_entry["constructs"][construct] = {
                    "tested_depths": 0,
                    "successes": 0,
                    "success_rate": 0.0,
                    "max_success_depth": 0,
                    "first_failure_depth": None,
                    "actual_models_seen": []
                }

            construct_entry = model_entry["constructs"][construct]
            construct_entry["tested_depths"] += 1

            if success:
                construct_entry["successes"] += 1
                construct_entry["max_success_depth"] = max(construct_entry["max_success_depth"], depth)
            elif construct_entry["first_failure_depth"] is None:
                construct_entry["first_failure_depth"] = depth

            if actual_model and actual_model not in construct_entry["actual_models_seen"]:
                construct_entry["actual_models_seen"].append(actual_model)

            model_entry["test_points"].append({
                "construct": construct,
                "depth": depth,
                "requested_model": requested_model,
                "actual_model": actual_model,
                "result": "success" if success else "failure",
                "success_reason": r["evaluation"].get("success_reason"),
                "failure_type": failure_type,
                "failure_reason": failure_reason,
                "failure_explanation": None if success else failure_explanation,
                "api_reason": error_info.get("api_reason"),
                "api_message": error_info.get("api_message"),
                "failure_point": {
                    "first_wrong_line": failure_analysis.get("first_wrong_line"),
                    "last_correct_line": failure_analysis.get("last_correct_line"),
                    "last_correct_call": failure_analysis.get("last_correct_call"),
                    "failure_started_at": failure_analysis.get("failure_started_at"),
                    "failure_reason": failure_analysis.get("failure_reason")
                }
            })

        for requested_model, model_entry in models_summary.items():
            model_entry["success_rate"] = round(
                model_entry["total_successes"] / model_entry["total_tests"], 3
            ) if model_entry["total_tests"] else 0.0

            for construct, construct_entry in model_entry["constructs"].items():
                construct_entry["success_rate"] = round(
                    construct_entry["successes"] / construct_entry["tested_depths"], 3
                ) if construct_entry["tested_depths"] else 0.0

            model_entry["test_points"].sort(key=lambda x: (x["construct"], x["depth"]))

        return {
            "meta": {
                "run_id": run_id,
                "experiment": EXPERIMENT_NAME,
                "provider": PROVIDER_NAME,
                "timestamp": datetime.now().isoformat()
            },
            "overall": {
                "total_tests": total_tests,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "success_rate": round(total_successes / total_tests, 3) if total_tests else 0.0
            },
            "models": models_summary
        }

    clean_summary = build_clean_summary(run_id, summary, flat_results)

    save_json(os.path.join(summaries_root, "summary.json"), clean_summary)

    print("\nExperiment complete.")
    print(f"Results saved under: {run_root}")

if __name__ == "__main__":
    run_experiment()
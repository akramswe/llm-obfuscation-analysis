import os
import json
import time
import re
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
    "nvidia/nemotron-3-super-120b-a12b:free"
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
]

# CONSTRUCT_TYPES = [
#     "direct",
#     "function_pointer",
#     "pointer_array",
#     "struct_pointer",
#     "switch_dispatch"
# ]

CALL_CHAIN_SIZES_BY_CONSTRUCT = {
    "direct": [100, 300, 1000]
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

def generate_direct_call_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"
    for i in range(1, n):
        code += f"void func{i}(void){{ func{i+1}(); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n"
    code += "int main(void){ func1(); return 0; }\n"
    return code

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

def generate_code(n: int, construct: str) -> str:
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

Instructions:
- Follow the function calls until the program terminates.
- Output EXACTLY one call per line.
- Use the exact format: funcX calls funcY
- The final line must be: func{depth} calls END
- Do NOT include main in the output.
- Do NOT include explanations, comments, numbering, or code blocks.
- Output ONLY the call chain.

Example format:
func1 calls func2
func2 calls func3
func3 calls END

C CODE:
<<<CODE
{code}
CODE
"""

# ===================== ESTIMATION =====================

def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def normalize_response(response: Optional[str]) -> str:
    if not response:
        return ""

    text = response

    # Fix missing space: "func1calls func2" -> "func1 calls func2"
    text = re.sub(r'(func\d+)calls', r'\1 calls', text)
    # Fix missing space after "calls": "func1 callsfunc2" -> "func1 calls func2"
    text = re.sub(r'calls(func\d+|END)', r'calls \1', text)
    # Fix merged lines:
    # "func10 calls func11func11 calls func12"
    # -> "func10 calls func11\nfunc11 calls func12"
    text = re.sub(
        r'(func\d+\s+calls\s+(?:func\d+|END))(func\d+\s+calls\s+)',
        r'\1\n\2',
        text
    )

    # Normalize spaces inside each line
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)

    return "\n".join(lines)

# ===================== EVALUATION =====================

def classify_failure(response: Optional[str], depth: int, error: Optional[str]) -> Optional[str]:
    if error is not None:
        error_lower = error.lower()

        if "prompt_too_large" in error_lower:
            return "prompt_too_large"
        if "daily_token_limit" in error_lower or "tokens per day" in error_lower:
            return "daily_token_limit"
        if "rate_limit" in error_lower:
            return "rate_limit"
        return "api_error"

    if verify_response(response, depth):
        return None

    if response is None or not response.strip():
        return "empty_response"

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return "empty_response"

    last_line = lines[-1]

    if " calls " not in last_line:
        return "malformed_output"

    if last_line != f"func{depth} calls END":
        return "underestimate"

    return "exact_match_failure"

def verify_response(response: Optional[str], depth: int) -> bool:
    if not response:
        return False

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    if len(lines) != depth:
        return False

    for i in range(1, depth):
        if lines[i - 1] != f"func{i} calls func{i+1}":
            return False

    if lines[-1] != f"func{depth} calls END":
        return False

    return True

# ===================== API =====================

def call_with_retries(client, model: str, prompt: str, retries: int = 5) -> Tuple[Optional[str], Optional[str], Optional[float]]:
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
            return text, None, latency

        except RateLimitError as e:
            last_err = str(e)
            print(f"DEBUG RateLimitError attempt {attempt + 1}/{retries}: {last_err}")

            err_lower = last_err.lower()

            if "tokens per day" in err_lower:
                return None, "daily_token_limit", None

            if "request too large" in err_lower or ("limit" in err_lower and "requested" in err_lower):
                return None, "prompt_too_large", None

            if attempt < retries - 1:
                sleep_time = (2 ** attempt) * 10
                print(f"DEBUG sleeping {sleep_time}s before retry...")
                time.sleep(sleep_time)
            else:
                return None, "rate_limit", None

        except APIStatusError as e:
            msg = str(e)
            print(f"DEBUG APIStatusError attempt {attempt + 1}/{retries}: {msg}")

            msg_lower = msg.lower()

            if "tokens per day" in msg_lower:
                return None, "daily_token_limit", None

            if "request too large" in msg_lower or ("limit" in msg_lower and "requested" in msg_lower):
                return None, "prompt_too_large", None

            if "rate_limit_exceeded" in msg_lower or "rate limit" in msg_lower:
                if attempt < retries - 1:
                    sleep_time = (2 ** attempt) * 10
                    print(f"DEBUG sleeping {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                    continue
                return None, "rate_limit", None

            return None, "api_error", None

        except Exception as e:
            msg = str(e)
            print(f"DEBUG UnexpectedError attempt {attempt + 1}/{retries}: {msg}")
            return None, "api_error", None

    return None, "rate_limit" if last_err else "api_error", None

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

def analyze_failure_point(response: Optional[str], depth: int, error: Optional[str] = None):
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
            "failure_started_at": "func1",
            "failure_reason": "empty_response"
        }

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    last_correct_line = 0
    last_correct_call = None

    for i in range(1, depth + 1):
        expected = f"func{i} calls func{i+1}" if i < depth else f"func{depth} calls END"

        if i > len(lines):
            return {
                "first_wrong_line": i,
                "last_correct_line": last_correct_line,
                "last_correct_call": last_correct_call,
                "failure_started_at": f"func{i}",
                "failure_reason": "truncated_or_stopped_early"
            }

        if lines[i - 1] != expected:
            return {
                "first_wrong_line": i,
                "last_correct_line": last_correct_line,
                "last_correct_call": last_correct_call,
                "failure_started_at": f"func{i}",
                "failure_reason": "incorrect_line"
            }

        last_correct_line = i
        last_correct_call = lines[i - 1]

    return {
        "first_wrong_line": None,
        "last_correct_line": depth,
        "last_correct_call": lines[-1] if lines else None,
        "failure_started_at": None,
        "failure_reason": None
    }

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    last_correct_line = 0
    last_correct_call = None

    for i in range(1, depth + 1):
        expected = f"func{i} calls func{i+1}" if i < depth else f"func{depth} calls END"

        if i > len(lines):
            return {
                "first_wrong_line": i,
                "last_correct_line": last_correct_line,
                "last_correct_call": last_correct_call,
                "failure_started_at": f"func{i}"
            }

        if lines[i - 1] != expected:
            return {
                "first_wrong_line": i,
                "last_correct_line": last_correct_line,
                "last_correct_call": last_correct_call,
                "failure_started_at": f"func{i}"
            }

        last_correct_line = i
        last_correct_call = lines[i - 1]

    return {
        "first_wrong_line": None,
        "last_correct_line": depth,
        "last_correct_call": lines[-1] if lines else None,
        "failure_started_at": None
    }

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
                print(f"Testing depth {depth}")

                code = generate_code(depth, construct)
                prompt = build_prompt(code, depth)
                est_prompt_tokens = approx_tokens(prompt)

                text = None
                err = None
                latency = None

                if est_prompt_tokens > TPM_SOFT_LIMIT:
                    print(
                        f"WARNING: estimated prompt tokens {est_prompt_tokens} exceed soft limit {TPM_SOFT_LIMIT}, sending anyway..."
                    )

                print(f"DEBUG: starting API call for depth {depth}")
                text, err, latency = call_with_retries(client, model, prompt, retries=5)
                print(f"DEBUG: finished API call for depth {depth}")

                if text is None:
                    text = ""

                normalized_text = normalize_response(text)

                success = verify_response(normalized_text, depth)
                failure_type = classify_failure(normalized_text, depth, err)
                last_call = extract_last_call(normalized_text)
                predicted_last_func = extract_predicted_func(normalized_text)
                failure_analysis = analyze_failure_point(normalized_text, depth, err)

                print(f"DEBUG: est_prompt_tokens={est_prompt_tokens}, err={err}")
                print(f"DEBUG: raw_text={repr(text[:200])}")
                print(f"DEBUG: normalized_text={repr(normalized_text[:200])}")
                print(f"DEBUG: last_call={last_call}")
                print(f"DEBUG: predicted_last_func={predicted_last_func}")
                print(f"DEBUG: failure_type={None if success else failure_type}")
                print(f"DEBUG: failure_analysis={failure_analysis}")

                if success:
                    successes += 1
                    max_success_depth = depth
                elif first_failure_depth is None:
                    first_failure_depth = depth

                expected_func = f"func{depth}"

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
                        "name": model,
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
                        "failure_type": None if success else failure_type,
                        "failure_analysis": failure_analysis
                    },
                    "error": err
                }

                out_path = os.path.join(construct_dir, f"depth_{depth}.json")
                save_json(out_path, result_data)
                flat_results.append(result_data)

                if success:
                    print(f"Result: success=True | last_call={last_call}")
                else:
                    print(f"Result: success=False ({failure_type}) | expected=func{depth}, last_call={last_call}")

                if err == "daily_token_limit":
                    stop_all = True
                    break

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

    save_json(os.path.join(summaries_root, "flat_results.json"), flat_results)
    save_json(os.path.join(summaries_root, "summary_by_model_construct.json"), summary)

    print("\nExperiment complete.")
    print(f"Results saved under: {run_root}")

if __name__ == "__main__":
    run_experiment()
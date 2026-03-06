import os
import json
import time
from datetime import datetime
from groq import Groq, RateLimitError, APIStatusError

# ===================== CONFIG =====================
MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b"
]

CONSTRUCT_TYPES = [
    "direct",
    "function_pointer",
    "pointer_array",
    "struct_pointer",
    "switch_dispatch"
]

# CALL_CHAIN_SIZES = [100, 300, 500, 700, 900, 1100]
# Different constructs generate different code sizes → use different depth schedules (recommended)
# CALL_CHAIN_SIZES_BY_CONSTRUCT = {
#     "direct": [100, 300, 500, 700, 900, 1000],
#     "function_pointer": [100, 200, 300, 400, 500],
#     "pointer_array": [100, 200, 300, 400, 500],
#     "struct_pointer": [100, 200, 300, 400, 500],
#     "switch_dispatch": [100, 200, 300, 400, 500],
# }

CALL_CHAIN_SIZES_BY_CONSTRUCT = {
    "direct": [100, 300, 500, 700],
    "function_pointer": [100, 200, 300],
    "pointer_array": [100, 200, 300],
    "struct_pointer": [100, 200, 300],
    "switch_dispatch": [100, 200, 300],
}

TEMPERATURE = 0.0
DELAY_BETWEEN_REQUESTS = 15
BASE_RESULTS_DIR = "results_constructs"

# ✅ Use env var ONLY (do not hardcode keys)
GROQ_API_KEY = "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"
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY as environment variable.")

# Groq on-demand TPM limit mentioned in your error: 8000
# Keep headroom under that so you don't hit 413.
TPM_SOFT_LIMIT = 7800

# ===================== GENERATORS =====================

def generate_direct_call_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"
    for i in range(1, n):
        code += f"void func{i}(){{func{i+1}();}}\n"
    code += f"void func{n}(){{printf(\"End\\n\");}}\n"
    code += "int main(){func1();return 0;}\n"
    return code


def generate_function_pointer_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"

    # Forward declarations
    for i in range(1, n + 1):
        code += f"void func{i}();\n"
    code += "\n"

    # Declare function pointers
    for i in range(1, n):
        code += f"void (*fp{i})();\n"
    code += "\n"

    # Define functions
    for i in range(1, n):
        code += f"void func{i}(){{ fp{i}(); }}\n"
    code += f"void func{n}(){{ printf(\"End\\n\"); }}\n\n"

    # Initialize pointers in main
    code += "int main(){\n"
    for i in range(1, n):
        code += f"fp{i} = func{i+1};\n"
    code += "func1();\nreturn 0;\n}\n"

    return code

def generate_pointer_array_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"
    code += f"#define N {n}\n"
    code += "typedef void (*fp_t)(void);\n"
    code += "fp_t fp[N+1];\n\n"

    # Forward declarations
    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    # Define functions: funci calls fp[i]()
    for i in range(1, n):
        code += f"void func{i}(void){{ fp[{i}](); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    # Init pointer array
    code += "int main(void){\n"
    for i in range(1, n):
        code += f"fp[{i}] = func{i+1};\n"
    code += f"fp[{n}] = func{n};\n"  # not used, but keeps array fully valid
    code += "func1();\nreturn 0;\n}\n"
    return code

def generate_struct_pointer_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"
    code += "typedef void (*fp_t)(void);\n"
    code += "struct Node { fp_t next; };\n"
    code += "struct Node nodes[%d+1];\n\n" % n

    # Forward declarations
    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    # Functions call through struct field
    for i in range(1, n):
        code += f"void func{i}(void){{ nodes[{i}].next(); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    # Initialize struct field pointers
    code += "int main(void){\n"
    for i in range(1, n):
        code += f"nodes[{i}].next = func{i+1};\n"
    code += f"nodes[{n}].next = func{n};\n"
    code += "func1();\nreturn 0;\n}\n"
    return code


def generate_switch_dispatch_chain(n: int) -> str:
    code = "#include <stdio.h>\n\n"

    # Forward declarations
    for i in range(1, n + 1):
        code += f"void func{i}(void);\n"
    code += "\n"

    # Dispatcher: id -> call func(id+1)
    code += "void dispatch(int id){\n"
    code += "  switch(id){\n"
    for i in range(1, n):
        # dispatch(i) calls func(i+1)
        code += f"    case {i}: func{i+1}(); break;\n"
    code += "    default: break;\n"
    code += "  }\n"
    code += "}\n\n"

    # Functions call dispatch(i)
    for i in range(1, n):
        code += f"void func{i}(void){{ dispatch({i}); }}\n"
    code += f"void func{n}(void){{ printf(\"End\\n\"); }}\n\n"

    code += "int main(void){\n"
    code += "func1();\nreturn 0;\n}\n"
    return code

def generate_code(n: int, construct: str) -> str:
    if construct == "direct":
        return generate_direct_call_chain(n)
    elif construct == "function_pointer":
        return generate_function_pointer_chain(n)
    elif construct == "pointer_array":
        return generate_pointer_array_chain(n)
    elif construct == "struct_pointer":
        return generate_struct_pointer_chain(n)
    elif construct == "switch_dispatch":
        return generate_switch_dispatch_chain(n)
    else:
        raise ValueError(f"Unknown construct type: {construct}")
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

# ===================== SIZE ESTIMATE =====================

def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4) # Rough estimate: 1 token ~ 4 chars (good enough to avoid 413 spam)

# ===================== VERIFICATION =====================

def verify_response(response: str, depth: int) -> bool:
    if not response:
        return False

    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]

    # Must be exactly depth lines
    if len(lines) != depth:
        return False

    for i in range(1, depth):
        if lines[i - 1] != f"func{i} calls func{i+1}":
            return False

    if lines[-1] != f"func{depth} calls END":
        return False

    return True

# ===================== RETRIES =====================

def call_with_retries(client, model: str, prompt: str, retries: int = 5):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
            )
            return resp.choices[0].message.content.strip(), None
        except RateLimitError as e:
            last_err = str(e)
            time.sleep((2 ** attempt) * 5)  # 5,10,20,40,80s
        except APIStatusError as e:
            return None, str(e)
    return None, last_err

# ===================== EXPERIMENT =====================

def run_experiment():
    summary = []
    client = Groq(api_key=GROQ_API_KEY)  # create once

    for model in MODELS:
        print(f"\n===== MODEL: {model} =====\n")

        model_dir = os.path.join(BASE_RESULTS_DIR, model.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)

        for construct in CONSTRUCT_TYPES:
            print(f"--- Construct: {construct} ---")

            construct_dir = os.path.join(model_dir, construct)
            os.makedirs(construct_dir, exist_ok=True)

            max_success_depth = 0
            depths = CALL_CHAIN_SIZES_BY_CONSTRUCT.get(construct, [100, 300, 500, 700, 900, 1100])

            for depth in depths:
                print(f"Testing depth: {depth}")

                code = generate_code(depth, construct)
                prompt = build_prompt(code, depth)

                # Skip prompts that are likely to exceed the 8000 TPM limit
                est = approx_tokens(prompt)

                if est > TPM_SOFT_LIMIT:
                    print(f"Skipping depth {depth} (~{est} tokens > limit)")
                    text, err = None, "skipped_prompt_too_large"
                    success = False
                else:
                    text, err = call_with_retries(client, model, prompt, retries=5)

                    if err:
                        print("API error:", err)

                        if "tokens per day" in err:
                            print("Daily token limit reached. Stopping further tests.")
                            text = None
                            err = "skipped_tpd_limit"
                            success = False
                        else:
                            success = verify_response(text, depth)
                    else:
                        success = verify_response(text, depth)

                result_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "construct": construct,
                    "depth": depth,
                    "success": success,
                    "estimated_prompt_tokens": est,
                    "response_length": len(text) if text else 0,
                    "error": err,
                    "response_text": text,
                }

                out_file = os.path.join(construct_dir, f"depth_{depth}.json")
                with open(out_file, "w") as f:
                    json.dump(result_data, f, indent=2)

                # Stop the loop if daily token limit reached
                if err == "skipped_tpd_limit":
                    break

                if success:
                    max_success_depth = depth

                print(f"Result: {success}\n")
                time.sleep(DELAY_BETWEEN_REQUESTS)

            summary.append({
                "model": model,
                "construct": construct,
                "max_success_depth": max_success_depth
            })

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    summary_file = os.path.join(BASE_RESULTS_DIR, f"summary_{timestamp}.json")

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===============================")
    print("Experiment complete.")
    print(f"Summary saved to: {summary_file}")
    print("===============================\n")


if __name__ == "__main__":
    run_experiment()
import os
import json
import time
import re
from datetime import datetime
from groq import Groq, RateLimitError, APIStatusError

# ===================== CONFIG =====================

MODELS = [
    "openai/gpt-oss-20b",
    # Add more models here
]

CONSTRUCT_TYPES = [
    "direct",
    "function_pointer",
]

CALL_CHAIN_SIZES = [100, 300, 500, 700, 900, 1100]

TEMPERATURE = 0.0
DELAY_BETWEEN_REQUESTS = 15
BASE_RESULTS_DIR = "results_constructs"

# GROQ_API_KEY = os.environ.get("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")
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY as environment variable.")

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


def generate_code(n: int, construct: str) -> str:
    if construct == "direct":
        return generate_direct_call_chain(n)
    elif construct == "function_pointer":
        return generate_function_pointer_chain(n)
    else:
        raise ValueError("Unknown construct type")

# ===================== PROMPT =====================

def build_prompt(code: str, depth: int) -> str:
    return f"""You are analyzing C code.

Reconstruct the COMPLETE linear function call chain starting from main().

Rules:
- One call per line.
- Format exactly: funcX calls funcY
- Final line must be: func{depth} calls END
- No explanation.

C CODE:
<<<CODE
{code}
CODE
"""

# ===================== VERIFICATION =====================

def verify_response(response: str, depth: int):
    if response is None:
        return False

    pair_pattern = r"func(\d+)\s*calls\s*func(\d+)"
    pairs = [(int(a), int(b)) for a, b in re.findall(pair_pattern, response)]

    if not pairs:
        return False

    pairs.sort(key=lambda x: x[0])

    for a, b in pairs:
        if b != a + 1:
            return False

    if f"func{depth} calls END" not in response:
        return False

    return True

# ===================== EXPERIMENT =====================

def run_experiment():
    summary = []

    for model in MODELS:
        print(f"\n===== MODEL: {model} =====\n")
        client = Groq(api_key=GROQ_API_KEY)

        model_dir = os.path.join(BASE_RESULTS_DIR, model.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)

        for construct in CONSTRUCT_TYPES:
            print(f"--- Construct: {construct} ---")

            construct_dir = os.path.join(model_dir, construct)
            os.makedirs(construct_dir, exist_ok=True)

            max_success_depth = 0

            for depth in CALL_CHAIN_SIZES:
                print(f"Testing depth: {depth}")

                code = generate_code(depth, construct)
                prompt = build_prompt(code, depth)

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                    )
                    text = response.choices[0].message.content.strip()
                except (RateLimitError, APIStatusError) as e:
                    print("API error:", e)
                    text = None

                success = verify_response(text, depth)

                result_data = {
                    "model": model,
                    "construct": construct,
                    "depth": depth,
                    "success": success,
                    "response_length": len(text) if text else 0,
                }

                out_file = os.path.join(
                    construct_dir,
                    f"depth_{depth}.json"
                )

                with open(out_file, "w") as f:
                    json.dump(result_data, f, indent=2)

                if success:
                    max_success_depth = depth

                print(f"Result: {success}\n")
                time.sleep(DELAY_BETWEEN_REQUESTS)

            summary.append({
                "model": model,
                "construct": construct,
                "max_success_depth": max_success_depth
            })

    # Save global summary
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_file = os.path.join(BASE_RESULTS_DIR, f"summary_{timestamp}.json")

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===============================")
    print("Experiment complete.")
    print(f"Summary saved to: {summary_file}")
    print("===============================\n")


if __name__ == "__main__":
    run_experiment()
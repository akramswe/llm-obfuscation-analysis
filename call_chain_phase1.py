import os
import json
import time
import re
from datetime import datetime
from groq import Groq, RateLimitError, APIStatusError

# ===================== CONFIG =====================

MODEL_NAME = "openai/gpt-oss-20b"
GROQ_API_KEY = "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"
# GROQ_API_KEY = os.environ.get("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")

# Keep sizes realistic for Dev tier
CALL_CHAIN_SIZES = [100, 300, 500, 700, 900, 1000]

TEMPERATURE = 0.0
RESULTS_DIR = "results_call_chain_phase1"
DELAY_BETWEEN_REQUESTS = 15  # seconds (avoid TPM burst)

if not GROQ_API_KEY or "GROQ_API_KEY" in GROQ_API_KEY:
    raise RuntimeError("Please set your GROQ_API_KEY.")

os.makedirs(RESULTS_DIR, exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

# ===================== GENERATE CALL CHAIN =====================

def generate_call_chain_code(n: int) -> str:
    """
        Minimal-token C program:
        main -> func1 -> func2 -> ... -> funcN
        No forward declarations (saves tokens).
        """

    code = "#include <stdio.h>\n\n"

    # Only function definitions (no forward declarations)
    for i in range(1, n):
        code += f"void func{i}(){{func{i+1}();}}\n"

    code += f"void func{n}(){{printf(\"End\\n\");}}\n"
    code += "int main(){func1();return 0;}\n"

    return code


# ===================== BUILD PROMPT =====================

def build_prompt(code: str, depth: int) -> str:
    return f"""You are a static program analysis system.

        You are given C source code.
        
        Your task:
        1. Start from main().
        2. Follow the direct function calls.
        3. Reconstruct the FULL linear call chain.
        4. Do NOT skip any function.
        5. Do NOT summarize.
        6. Do NOT explain.
        7. Do NOT add any extra text.
        
        Output format rules (STRICT):
        - Exactly one function call per line.
        - Format each line exactly like this:
          funcX calls funcY
        - The final line must be:
          func{depth} calls END
        - Output must contain exactly {depth} lines.
        - No numbering.
        - No commentary.
        - No blank lines before or after.
        
        C CODE:
        <<<CODE
        {code}
        CODE
        """

# ===================== QUERY LLM =====================

def query_llm(prompt: str):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        print("RateLimitError: hit a rate/token limit. Try increasing delay or reducing sizes.")
        return None

    except APIStatusError as e:
        print("API error:", e)
        return None


# ===================== VERIFICATION =====================

"""
- Accepts minor formatting differences.
    - Requires a continuous sequential chain func1->func2->...->funcN and explicit END.
    - Correctly handles that 'funcN calls END' is not a func-to-func pair.
"""
def verify_response(response: str, depth: int):
    if response is None:
        return False, "No response (API error)"

    # Extract funcX calls funcY pairs
    pair_pattern = r"func(\d+)\s*calls\s*func(\d+)"
    pairs = [(int(a), int(b)) for a, b in re.findall(pair_pattern, response)]

    if not pairs:
        return False, "No valid funcX calls funcY pairs found"

    pairs.sort(key=lambda x: x[0])

    # Must start correctly
    if pairs[0] != (1, 2):
        return False, "Chain does not start with func1 calls func2"

    # Check sequential continuity
    for a, b in pairs:
        if b != a + 1:
            return False, f"Non-sequential call at func{a} -> func{b}"

    # Check for explicit END line
    end_pattern = rf"\bfunc{depth}\s*calls\s*END\b"
    has_end = re.search(end_pattern, response) is not None

    if not has_end:
        last_called = pairs[-1][1]
        return False, f"Chain ended early at func{last_called} (missing END)"

    # Ensure we reached func{depth-1} calls func{depth}
    last_pair_caller = pairs[-1][0]
    if last_pair_caller < depth - 1:
        return False, f"Missing calls near end (last seen func{last_pair_caller})"

    return True, "Full chain reconstructed (ended with END)"

# ===================== MAIN =====================

def main():

    print("\n================ PHASE 1: CALL CHAIN SCALING ================\n")
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {TEMPERATURE}\n")

    results = []

    for depth in CALL_CHAIN_SIZES:

        print(f"Testing depth: {depth}")

        code = generate_call_chain_code(depth)
        prompt = build_prompt(code, depth)

        response = query_llm(prompt)

        success, message = verify_response(response, depth)

        results.append({
            "depth": depth,
            "success": success,
            "message": message,
            "response_length_chars": len(response) if response else 0,
        })

        print(f"Result: {success} | {message}\n")

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(
        RESULTS_DIR,
        f"phase1_results_{timestamp}.json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("===============================================================")
    print(f"Results saved to: {output_file}")
    print("Phase 1 complete.\n")


if __name__ == "__main__":
    main()

import os
import subprocess
import json
from groq import Groq
from datetime import datetime

# ===================== CONFIG =====================
PROGRAMS_DIR = "programs"
OBFUSCATED_DIR = "obfuscated"
RESULTS_DIR = "results"

MODEL_NAME = "openai/gpt-oss-20b"
GROQ_API_KEY = "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"

TARGET_FUNCTION = "main"
RUNS = 5

# ===================== SETUP ======================
os.makedirs(RESULTS_DIR, exist_ok=True)
for d in ["flatten", "virtualize", "opaque"]:
    os.makedirs(os.path.join(OBFUSCATED_DIR, d), exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

# ===================== TIGRESS ====================
def run_tigress(input_c, output_c, transform):
    subprocess.run(
        [
            "tigress",
            f"--Transform={transform}",
            f"--Functions={TARGET_FUNCTION}",
            input_c,
            f"--out={output_c}",
        ],
        check=True,
    )

# ===================== GROQ =======================
def query_groq(code):
    MAX_CHARS = 6000  # safe for Groq TPM limits
    truncated_code = code[:MAX_CHARS]

    prompt = (
        "The following is an obfuscated C program (possibly incomplete). "
        "Based on the visible structure, what algorithm does it implement? "
        "Explain briefly.\n\n"
        + truncated_code
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# ===================== INPUT ======================
def choose_obfuscation():
    print("1 - Flatten")
    print("2 - Virtualize")
    print("3 - Opaque")
    print("4 - All")
    c = input("Choice: ").strip()
    return {
        "1": ["Flatten"],
        "2": ["Virtualize"],
        "3": ["AddOpaque"],
        "4": ["Flatten", "Virtualize", "AddOpaque"],
    }.get(c, [])

def ask_groq():
    return input("Run Groq API? (y/n): ").strip().lower() == "y"

# ===================== MAIN =======================
def main():
    transforms = choose_obfuscation()
    if not transforms:
        return

    run_llm = ask_groq()
    results = []

    for prog in os.listdir(PROGRAMS_DIR):
        if not prog.endswith(".c"):
            continue

        in_path = os.path.join(PROGRAMS_DIR, prog)

        for t in transforms:
            folder = t.lower().replace("addopaque", "opaque")
            out_name = f"{prog.replace('.c','')}_{folder}.c"
            out_path = os.path.join(OBFUSCATED_DIR, folder, out_name)

            run_tigress(in_path, out_path, t)

            entry = {
                "program": prog,
                "obfuscation": t,
                "model": MODEL_NAME,
                "runs": [],
            }

            if run_llm:
                with open(out_path, "r") as f:
                    code = f.read()
                for i in range(1, RUNS + 1):
                    entry["runs"].append(
                        {"run": i, "response": query_groq(code)}
                    )

            results.append(entry)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")


# ===================== RUN ========================
if __name__ == "__main__":
    main()

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
RUNS = 7

# Set in WSL before running:
# export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxx"
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

GROUND_TRUTH = {
    "anagram.c": "anagram",
    "armstrong.c": "armstrong",
}

# ===================== SETUP ======================
os.makedirs(RESULTS_DIR, exist_ok=True)
for d in ["flatten", "virtualize", "opaque"]:
    os.makedirs(os.path.join(OBFUSCATED_DIR, d), exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

# ===================== TIGRESS ====================
# def run_tigress(input_c, output_c, transform):
#     subprocess.run(
#         [
#             "tigress",
#             f"--Transform={transform}",
#             f"--Functions={TARGET_FUNCTION}",
#             input_c,
#             f"--out={output_c}",
#         ],
#         check=True,
#     )

def run_tigress(input_c, output_c, transform):
    cmd = ["tigress"]

    if transform == "AddOpaque":
        pretty_name = "Opaque"
        cmd.extend([
            "--Transform=InitOpaque",
            "--Functions=main",
            "--InitOpaqueStructs=list,array,input,env",
            "--InitOpaqueCount=2",

            "--Transform=AddOpaque",
            "--Functions=main",
            "--AddOpaqueKinds=question",
            "--AddOpaqueCount=5",
        ])
    else:
        pretty_name = transform
        cmd.extend([
            f"--Transform={transform}",
            "--Functions=main",
        ])

    cmd.extend([
        input_c,
        f"--out={output_c}",
    ])

    try:
        subprocess.run(cmd, check=True)

        if os.path.exists(output_c):
            print(f"Created obfuscated file for {pretty_name}.")
            print(f"Location is: {output_c}\n")
            return True
        else:
            print(f"Failed to create obfuscated file for {pretty_name}.\n")
            return False

    except subprocess.CalledProcessError:
        print(f"Failed to create obfuscated file for {pretty_name}.\n")
        return False

# ===================== GROQ =======================
# def query_groq(code):
#     MAX_CHARS = 6000  # keep under token limits
#     code = code[:MAX_CHARS]

    # prompt = (
    #     "The following is an obfuscated C program. "
    #     "What algorithm does it implement? Explain briefly.\n\n"
    #     + code
    # )
    # prompt = (
    #         "The following is an obfuscated C program.\n\n"
    #         "Analyze the control flow, data usage, and arithmetic operations visible in the code.\n"
    #         "Determine whether the underlying algorithm can be identified with confidence.\n\n"
    #         "If the algorithm can be identified, clearly name it.\n"
    #         "If the algorithm cannot be identified with sufficient evidence, respond exactly with:\n"
    #         "cannot determine\n\n"
    #         "Do not guess or speculate.\n"
    #         "Explain your reasoning briefly.\n\n"
    #         + code
    # )
    #
    # response = client.chat.completions.create(
    #     model=MODEL_NAME,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.7,
    # )
    # return response.choices[0].message.content.strip()

def query_groq(code):
    MAX_CHARS = 6000
    code = code[:MAX_CHARS]

    prompt = (
            "The following C program has been obfuscated using control-flow flattening.\n"
            "No functionality has been added or removed.\n\n"
            "From a reverse-engineering perspective, analyze the control flow and data usage.\n"
            "Try to mentally reconstruct the original structure of the program.\n\n"
            "If you can identify the underlying algorithm with reasonable confidence,\n"
            "name the algorithm clearly.\n\n"
            "If the algorithm cannot be identified, respond exactly with:\n"
            "cannot determine\n\n"
            "Do not guess.\n"
            "Explain your reasoning briefly.\n\n"
            + code
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except RateLimitError:
        print("\nGroq API limit reached.")
        print("Your token limit is exhausted.")
        print("Please wait before retrying or upgrade your Groq plan.\n")
        exit(1)

    except APIStatusError as e:
        print("\nGroq API error:")
        print(e)
        exit(1)

    except Exception as e:
        print("\nUnexpected error:")
        print(e)
        exit(1)


# ===================== CLASSIFIER =================
def classify_response(response, truth):
    r = response.lower()
    t = truth.lower()

    if t in r:
        return "correct"
    if "cannot" in r or "not determine" in r or "unknown" in r:
        return "unknown"
    return "incorrect"

def generate_comment(classification, truth):
    if classification == "correct":
        return f"Correct identification of {truth}."
    if classification == "unknown":
        return "Model could not determine the algorithm."
    return f"Confident but incorrect guess; original program is {truth}."

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

    if not run_llm:
        print("\nGroq API disabled. Skipping LLM analysis.\n")
    else:
        print("\nGroq API enabled. Running LLM analysis.\n")


    for prog in os.listdir(PROGRAMS_DIR):
        if not prog.endswith(".c"):
            continue

        in_path = os.path.join(PROGRAMS_DIR, prog)

        for t in transforms:
            folder = t.lower().replace("addopaque", "opaque")
            out_name = f"{prog.replace('.c','')}_{folder}.c"
            out_path = os.path.join(OBFUSCATED_DIR, folder, out_name)

            run_tigress(in_path, out_path, t)

            # success = run_tigress(in_path, out_path, t)
            #
            # if not success:
            #     continue  # skip to next file
            #
            # if not ask_continue():
            #     print("\nObfuscation stopped by user.\n")
            #     return

            entry = {
                "program": prog,
                "ground_truth": GROUND_TRUTH.get(prog, "unknown"),
                "runs": [],
            }

            if run_llm:
                with open(out_path, "r") as f:
                    code = f.read()

                for i in range(1, RUNS + 1):
                    resp = query_groq(code)
                    classification = classify_response(resp, entry["ground_truth"])
                    comment = generate_comment(classification, entry["ground_truth"])

                    entry["runs"].append({
                        "run": i,
                        "llm_answer": resp,
                        "comment": comment,
                        "classification": classification,
                    })

            summary = {"correct": 0, "incorrect": 0, "unknown": 0}
            for r in entry["runs"]:
                summary[r["classification"]] += 1

            summary["consistency"] = (
                "high" if summary["correct"] >= 4 else "low"
            )

            entry["summary"] = summary
            results.append(entry)

    if not run_llm:
        print("\nGroq API disabled.")
        print("No results file created.\n")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    final_output = {
        "experiment_metadata": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "model": MODEL_NAME,
            "runs_per_program": RUNS,
            "obfuscation": transforms,
        },
        "results": results,
    }

    output_file = os.path.join(
        RESULTS_DIR, f"results_{timestamp}.json"
    )

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to {output_file}\n")


# def ask_continue():
#     choice = input("Do you want to continue obfuscation? (y/n): ").strip().lower()
#     return choice == "y"


# ===================== RUN ========================
if __name__ == "__main__":
    main()

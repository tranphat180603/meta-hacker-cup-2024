import json
from unittest.mock import patch
import time
import re
from tqdm import tqdm
import argparse
import time
import sys

from datasets import load_dataset

from model import (
    load_model_and_tokenizer,
    understanding_problem,
    analyze_test_cases,
    get_refine_understanding,
    self_generate_test_cases,
    generate_solution_ideas,
    evaluate_solutions_f,
    generate_python_code,
    request_code_improvement_dte,
    request_code_improvement_dtfc
)

from dataloader import(
    extract_problem_cases_from_hf,
    extract_problem_cases_from_folder
)

from executor import(
    evaluate_generated_code_on_test_cases
)


class Tee:
    def __init__(self, *files):
        self.files = files
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.closed = False

    def write(self, data):
        if not self.closed:
            self.stdout.write(data)
            for f in self.files:
                f.write(data)
            self.flush()

    def flush(self):
        if not self.closed:
            self.stdout.flush()
            for f in self.files:
                f.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.closed:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            for f in self.files:
                if not f.closed:
                    f.close()
            self.closed = True

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full process for solving coding problems.")
    parser.add_argument("--code_iterations", type=int, default=15, help="Number of code improvement iterations.")
    parser.add_argument("--max_num_retry", type=int, default=10, help="Maximum number of retries for model responses.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (equal to the number of GPUs).")
    parser.add_argument("--problem_name", type=str, default=None, help="Specify the name of the problem to solve for hf dataset")
    parser.add_argument("--show_coT", action="store_true", help="Show the Chain of Thought output for debugging.")
    parser.add_argument("--dataset_local_path", type = str, default = "", help = "if specified, open dataset in local machine, problem is formatted the same as online dataset") 
    parser.add_argument("--local_ds_idx", type = int, help = "if specified, solve particular problem in the folder")
    parser.add_argument("--fine_tuned", action="store_true", help="flag to use my fine-tuned version. Currently supporting Qwen 2.5 7B")
    parser.add_argument("--out", type = str, default = "output.txt", help = "log C-O-T to a text file") 
    return parser.parse_args()


# Helper function to clean and parse JSON response
def response_json(response_string):
    # If the response is already a dictionary, return it as is
    if isinstance(response_string, dict):
        return response_string

    # Step 1: Remove 'json', backticks, and any LaTeX-style formatting like \( ... \) or \[ ... \]
    cleaned_response = response_string.replace('json', '').strip('```').strip()

    # Step 2: Remove LaTeX-like math notation \( ... \) or \[ ... \]
    cleaned_response = re.sub(r'\\\(|\\\)', '', cleaned_response)  # Removes \( and \)
    cleaned_response = re.sub(r'\\\[|\\\]', '', cleaned_response)  # Removes \[ and \]

    try:
        # Step 3: Parse the cleaned string into a Python dictionary
        parsed_json = json.loads(cleaned_response)
        return parsed_json
    except json.JSONDecodeError as e:
        # print(f"Error decoding JSON")
        return e  # Return None if parsing fails

# Retry function to retry any function that uses response_json with added try-except for resilience
def retry(func, max_attempts, *args, **kwargs):
    attempts = 0
    result = None

    while attempts < max_attempts:
        print(f"Parsing JSON attempts: #{attempts + 1}")
        try:
            raw_response = func(*args, **kwargs)
            parsed_response = response_json(raw_response)
            
            if parsed_response is not None and isinstance(parsed_response, dict):
                return parsed_response
            else:
                print(f"Error parsing json with this e: {parsed_response}")
        except Exception as e:
            print(f"Error at function retry: {e}")

        attempts += 1
    
    return None  # Return None to signal failure

# Main function to run the process
def run_full_process(model, tokenizer,problem_description, test_input, test_output ,code_iterations=5, max_num_retry=5, show_coT=False):
    try:
        # Step 1: Understand the problem
        understand = retry(understanding_problem, max_num_retry, model, tokenizer, problem_description, show_coT=show_coT)
        if not understand:
            print("Failed parsing JSON for problem understanding.")
            return

        # Step 2: Analyze test cases
        analysis = retry(analyze_test_cases, max_num_retry, model, tokenizer, problem_description, show_coT=show_coT)
        if not analysis:
            print("Failed parsing JSON for test case analysis.")
            return

        # Step 3: Refine understanding
        refine_understanding = retry(
            get_refine_understanding, max_num_retry, 
            model, 
            tokenizer,
            understand['understanding'], 
            analysis, #all new information from the test case analysis
            show_coT=show_coT
        )
        if not refine_understanding:
            print("Failed parsing JSON for refining understanding.")
            return

        # Step 4: Generate AI test cases
        ai_test = retry(
            self_generate_test_cases, 
            max_num_retry, 
            model, 
            tokenizer,
            refine_understanding['refined_problem_understanding'], 
            analysis, 
            show_coT=show_coT
        )
        if not ai_test:
            print("Failed parsing JSON for AI test case generation.")
            return

        # Step 5: Generate solution ideas
        solutions = retry(
            generate_solution_ideas, max_num_retry, 
            model, 
            tokenizer,
            refine_understanding['refined_problem_understanding'], 
            analysis, 
            num_solutions=5, 
            show_coT=show_coT
        )
        if not solutions:
            print("Failed parsing JSON for solution generation.")
            return

        # Step 6: Evaluate solutions
        evaluate_solutions = retry(
            evaluate_solutions_f, 
            max_num_retry, 
            model, 
            tokenizer,
            solutions['solutions'], 
            refine_understanding['refined_problem_understanding'], 
            analysis, 
            refine_understanding['refined_problem_understanding']['difficulty_assessment_update'], 
            show_coT=show_coT
        )
        if not evaluate_solutions:
            print("Failed parsing JSON for solution evaluation.")
            return

        # Step 7: Generate Python code
        code_solution = retry(
            generate_python_code, 
            max_num_retry, 
            model, 
            tokenizer,
            evaluate_solutions['selected_solution'], 
            analysis, 
            show_coT=show_coT
        )
        if not code_solution:
            print("Failed parsing JSON for Python code generation.")
            return

        generated_code = code_solution['solution_code']['code']
        attempts = 0
        best_score = 0
        best_code = generated_code

        # Step 8: Start the code iteration loop
        while attempts < code_iterations:
            # Run the generated code
            score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
                generated_code, test_input=test_input, test_output=test_output
            )
            
            #Fail to run code. Logging data
            if show_coT:
                if failed_cases:
                    print(f"Logic error. Failed cases are: {failed_cases}")
                else:
                    print(f"Execution error: {error}")
                print(f"Code iterations. Attempt #{attempts + 1}/{code_iterations}")

            # If this score is better than the previous best, update the best result
            if score > best_score:
                best_score = score
                best_code = generated_code

            # If we achieve a perfect score, stop and return the best code
            if best_score == 100:
                print(f"Perfect score achieved: {best_score}%")
                return best_code, best_score

            #Fix code
            if failed_cases:  # Handle failed test cases
                new_code = retry(request_code_improvement_dtfc, max_num_retry, model, tokenizer, generated_code, error, analysis ,show_coT=show_coT)
            else:  # Handle execution/runtime errors
                new_code = retry(request_code_improvement_dte, max_num_retry, model, tokenizer, generated_code, error, analysis , show_coT=show_coT)

            new_code = new_code['solution_code']['code'] if new_code else generated_code
            generated_code = new_code
            attempts += 1

        # After max iterations, return the best result so far if it exists
        if best_score > 0:
            return best_code, best_score
        else:
            return 

    except Exception as e:
        print(f"ERROR OCCURRED: {str(e)}")
        return None, 0


def process_problems_sequentially(model, tokenizer, problem_cases, code_iterations, max_num_retry, show_coT):
    total_problems = len(problem_cases)
    error_msg = ""
    for index, problem in enumerate(tqdm(problem_cases, desc="Processing problems", unit="problem")):
        try:
            print(f"\nRunning problem {index + 1}/{total_problems} {problem['name']}")
            problem_description = problem["problem_description"]
            input_data = problem["sample_input"]
            expected_output = problem["sample_output"]
            
            generated_code, best_score = run_full_process(model, tokenizer, problem_description, input_data, expected_output, code_iterations, max_num_retry, show_coT=show_coT)
            
            if best_score > 0:
                result = f"Problem {index + 1}/{total_problems}: {problem['name']}, Score: {best_score}%"
            else:
                result = f"Problem {index + 1}/{total_problems}: {problem['name']}, Failed to find solution after {code_iterations} iterations"

        except Exception as e:
            error_msg = f"ERROR processing problem {index + 1}/{total_problems}: {problem['name']}, Error: {str(e)}"

        with open('results.txt', 'w') as file, Tee(file):
            print(result)
            print(error_msg)

def main():
    #init arguments
    args = parse_args()
    #init model
    base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path = "../adapter/"
    model, tokenizer = load_model_and_tokenizer(base_model_name, adapter_path, lora=args.fine_tuned)
    with open(args.out, 'w') as f, Tee(f):
        if args.fine_tuned:
            print(f"Using model: {base_model_name} with Lora adapter fine_tuned by Phat")
        else:
            print(f"Using model: {base_model_name}")
        # Extract problem cases
        if args.dataset_local_path:  # handle local dataset (folder structured)
            if args.local_ds_idx is not None:
                problem_cases = [problem_cases[args.local_ds_idx]]
                print(f"Processing specific problem: {problem_cases[0]['name']}")
            else:
                problem_cases = extract_problem_cases_from_folder(args.dataset_local_path)
                print(f"Processing all {len(problem_cases)} problems in the folder")
        else:  # handle hf dataset
            ds = load_dataset("hackercupai/hackercup")
            problem_cases = extract_problem_cases_from_hf(ds)
            if args.problem_name:
                problem_cases = [problem for problem in problem_cases if problem['name'].lower() == args.problem_name.lower()]
                if not problem_cases:
                    print(f"No problem found with the name '{args.problem_name}'")
                    return
                print(f"Processing specific problem: {problem_cases[0]['name']}")
            else:
                print(f"Processing all {len(problem_cases)} problems from the dataset")

        # Process problems sequentially
        process_problems_sequentially(model, tokenizer,problem_cases, args.code_iterations, args.max_num_retry, args.show_coT)

        print("All processing finished.")

if __name__ == "__main__":
    main()


# python main.py --code_iterations 15 --max_num_retry 5 --dataset_local_path "contest_data" --show_coT

#python main.py --code_iterations 10 --max_num_retry 5 --dataset_local_path "contest_data" --show_coT --out "output1.txt"

#python main.py --problem_name "cheeseburger_corollary_ch1" --fine_tuned --show_coT
#python main.py --problem_name "cheeseburger_corollary_ch1" --show_coT
#python main.py --dataset_local_path "../contest_data/" --show_coT
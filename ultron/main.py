import json
import time
import argparse
import re
from tqdm import tqdm
import time
import sys

from datasets import load_dataset
from huggingface_hub import login
login("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")



from model import (
    load_model_and_tokenizer,
    load_image_model_and_tokenizer,
    understanding_problem,
    understanding_image,
    analyze_test_cases,
    get_refine_understanding,
    generate_solution_ideas,
    evaluate_solutions_f,
    generate_python_code,
    request_improvement_dte,
    request_improvement_dtfc,
    request_final_improvement
)

from dataloader import(
    extract_problem_cases_from_hf,
    extract_problem_cases_from_folder
)

from executor import(
    run_extracted_code_with_timeout,
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
    parser.add_argument("--code_iterations", type=int, default=20, help="Number of code improvement iterations.")
    parser.add_argument("--max_num_retry", type=int, default=5, help="Maximum number of retries for model responses.")
    parser.add_argument("--num_refinement", type=int, default=10, help="Maximum number of retries for model responses.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for dataset slicing")
    parser.add_argument("--end_index", type=int, default=None, help="End index for dataset slicing")
    parser.add_argument("--problem_name", type=str, default=None, help="Specify the name of the problem to solve for hf dataset")
    parser.add_argument("--show_coT", action="store_true", help="Show the Chain of Thought output for debugging.")
    parser.add_argument("--dataset_local_path", type = str, default = "", help = "if specified, open dataset in local machine, problem is formatted the same as online dataset") 
    parser.add_argument("--local_ds_idx", type = int, help = "if specified, solve particular problem in the folder")
    parser.add_argument("--fine_tuned", action="store_true", help="flag to use my fine-tuned version. Currently supporting Qwen 2.5 7B")
    parser.add_argument("--out", type = str, default = "output.txt", help = "log C-O-T to a text file")
    parser.add_argument("--result_out", type = str, default = "result.txt", help = "log result of processing to a text file")
    parser.add_argument("--model_name", type = str, default = None, help = "model name from hf") 
    return parser.parse_args()


# Helper function to clean and parse JSON response
def response_json(response_string):
    # If the response is already a dictionary, return it as is
    if isinstance(response_string, dict):
        return response_string

    # Step 1: Remove 'json', backticks
    cleaned_response = response_string.replace('json', '').strip('```').strip()
    
    # Step 2: Remove all LaTeX-style math notation
    # Remove \( ... \) notation
    cleaned_response = re.sub(r'\\\([^\)]*\\\)', '', cleaned_response)
    # Remove \[ ... \] notation
    cleaned_response = re.sub(r'\\\[[^\]]*\\\]', '', cleaned_response)
    # Remove other LaTeX commands
    cleaned_response = re.sub(r'\\[a-zA-Z]+\{[^\}]*\}', '', cleaned_response)
    # Remove remaining single LaTeX brackets
    cleaned_response = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', cleaned_response)

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

    while attempts < max_attempts:
        print(f"Parsing JSON attempts: #{attempts + 1}")
        raw_response = func(*args, **kwargs)
        parsed_response = response_json(raw_response)
        
        if parsed_response is not None and isinstance(parsed_response, dict):
            return parsed_response
        elif parsed_response is None:
            print(f"Error parsing json with this e: {parsed_response}")
        else:
            print(f"Error parsing json with this e: {parsed_response}")

        attempts += 1
    
    return None  # Return None to signal failure

# Main function to run the process
def run_full_process(model, tokenizer, img_model, img_tokenizer, problem_description, test_input, test_output, full_input, img_raw ,code_iterations=5, max_num_retry=5, refinement_num=5, show_coT=False):
    # Step 1: Understand the problem
    understand = retry(understanding_problem, max_num_retry, model, tokenizer, problem_description, show_coT=show_coT)
    if not understand:
        print("Failed parsing JSON for problem understanding.")
        return

    img_understanding = ""
    if img_raw:
        img_understanding = retry(understanding_image, max_num_retry, img_model, img_tokenizer, problem_description, img_raw, show_coT=show_coT)
        if not img_understanding:
            print("Failed parsing JSON for image understanding.")
            return
    
    # Track attempts and best results
    attempts = 0
    best_score = 0
    best_code = ""
    final_code = ""
    final_score = 0

    # Track the path we've gone through
    error_history = {}
    failure_history = {}  # Merged dictionary to track both failed cases and solution paths

    # Reflect after every iteration
    reflection = ""

    #track code refinement
    refinement_n = 0

    is_first_iteration = True
    while attempts < code_iterations:
        print(f"Iterate attempt #{attempts + 1}/{code_iterations}")

        # Step 2: Analyze test cases
        analysis = retry(analyze_test_cases, max_num_retry, model, tokenizer, problem_description, reflection, show_coT=show_coT)
        if not analysis:
            print("Failed parsing JSON for test case analysis.")
            return

        # Step 3: Refine understanding
        refine_understanding = retry(
            get_refine_understanding, 
            max_num_retry, 
            model, 
            tokenizer,
            understand, 
            analysis,
            reflection,
            img_understanding if is_first_iteration else "",
            show_coT=show_coT
        )
        if not refine_understanding:
            print("Failed parsing JSON for refining understanding.")
            return

        is_first_iteration = False

        # Step 4: Generate solution ideas
        solutions = retry(
            generate_solution_ideas, max_num_retry, 
            model, 
            tokenizer,
            refine_understanding, 
            analysis, 
            num_solutions=5, 
            show_coT=show_coT
        )
        if not solutions:
            print("Failed parsing JSON for solution generation.")
            return

        # Step 5: Evaluate solutions
        evaluate_solutions = retry(
            evaluate_solutions_f, 
            max_num_retry, 
            model, 
            tokenizer,
            solutions, 
            refine_understanding, 
            analysis, 
            show_coT=show_coT
        )
        if not evaluate_solutions:
            print("Failed parsing JSON for solution evaluation.")
            return

        # Step 6: Generate Python code
        code_solution = retry(
            generate_python_code, 
            max_num_retry, 
            model,
            tokenizer,
            evaluate_solutions, 
            analysis,
            refine_understanding,
            show_coT=show_coT
        )
        if not code_solution:
            print("Failed parsing JSON for Python code generation.")
            return

        generated_code = code_solution['solution_code']['code']
        curr_solution = code_solution['solution_code']['general_formula_update']
        
        if generated_code:
            # Run the generated code
            score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
                generated_code, test_input=test_input, test_output=test_output
            )
        
        # If this score is better than the previous best, update the best result
        if score > best_score:
            best_score = score
            best_code = generated_code

        #fix code
        if error:
            error = str(error)
            error_history[error] = error_history.get(error, 0) + 1
            if show_coT:
                print(f"Execution error: {error} (Occurred {error_history[error]} times)")

            # Separate loop for fixing code errors
            fix_attempts = 0
            while fix_attempts < max_num_retry:
                print(f"Fixing code attempt #{fix_attempts + 1}/{max_num_retry}")
                execution_error = retry(request_improvement_dte, max_num_retry, model, tokenizer, code_solution, error, analysis, error_history, show_coT=show_coT)

                # Generate and evaluate the fixed code
                fixed_code = execution_error['solution_code']['code']
                if fixed_code:
                    fix_score, error, _, failed_cases = evaluate_generated_code_on_test_cases(
                        fixed_code, test_input=test_input, test_output=test_output
                    )

                    # Exit if failed_cases is encountered
                    if failed_cases:
                        break  # Break out to handle `failed_cases` in the main loop

                    # If fixed code improves, use it in main loop
                    if fix_score > score:
                        best_code = fixed_code
                        best_score = fix_score
                        break  # Exit fix loop if score improved
                fix_attempts += 1
        if failed_cases:
            # Use a combined key of failed cases and current solution formula to track failure history
            combined_key = (str(failed_cases), curr_solution)
            failure_history[combined_key] = failure_history.get(combined_key, 0) + 1
            if show_coT:
                print(f"Failed cases: {failed_cases} with solution '{curr_solution}' (Occurred {failure_history[combined_key]} times)")
            failed_tests = retry(request_improvement_dtfc, max_num_retry, model, tokenizer, code_solution['solution_code'], str(failed_cases), refine_understanding, failure_history, show_coT=show_coT)
            reflection = failed_tests

        attempts += 1

        # If we achieve a perfect score, iterate more with ai-generated tests
        if best_score == 100:
            print("Perfect score achieved on sample test cases!")
            print("Checking this code compatibility on the full input set...")
            final_score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
            best_code, test_input=test_input, test_output=test_output
            )
            if final_score == 100:
                print("Nailed this problem")
                return final_score, best_code
            else:
                while refinement_n < refinement_num:
                    print("Current code passed on test set but failed on the full set.")
                    timeout_msg = None
                    print("Push one more step further, improve the program efficiency...")
                    final_code = retry(request_final_improvement, max_num_retry, model, tokenizer, generated_code, refine_understanding,timeout_msg ,show_coT=show_coT)
                    final_code = final_code["optimization"]["optimized_code"]

                    check_result, error = run_extracted_code_with_timeout(final_code, full_input)
                    
                    # Only evaluate if there’s a valid result from the code execution
                    if check_result:
                        final_score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
                            check_result, test_input=test_input, test_output=test_output
                        )
                        if final_score == 100:
                            print("Nailed this problem!")
                            best_code = final_code  # Corrected assignment here
                            return best_code, best_score
                        else:
                            print("Optimized code didn't successfully passed on the full test case. \nRegenerating...")
                    else:
                        print(f"Error encountered: {error}")

                    timeout_msg = error    
                    generated_code = final_code
                    refinement_n += 1
                print("Code couldn't execute on full inputs")
                return best_code, best_score
            

    return best_code, best_score


def process_problems_sequentially(model, tokenizer, img_model, img_tokenizer, file ,problem_cases, code_iterations, max_num_retry, num_refinement ,show_coT):
    total_problems = len(problem_cases)
    error_msg = []
    result = []
    for index, problem in enumerate(tqdm(problem_cases, desc="Processing problems", unit="problem")):
        try:
            print(f"\nRunning problem {index + 1}/{total_problems} {problem['name']}")
            problem_description = problem["problem_description"]
            input_data = problem["sample_input"]
            expected_output = problem["sample_output"]
            full_input = problem["full_input"]
            if problem['image']:
                img_raw = problem['image']
            else:
                img_raw = None
            generated_code, best_score = run_full_process(model, tokenizer, img_model, img_tokenizer,problem_description, input_data, expected_output, full_input, img_raw, code_iterations, max_num_retry, num_refinement ,show_coT=show_coT)
            
            if best_score > 0:
                log = f"Problem {index + 1}/{total_problems}: {problem['name']}, Score: {best_score}%"
                print(log)
                result.append(log)
            else:
                log = f"Problem {index + 1}/{total_problems}: {problem['name']}, Failed to find solution after {code_iterations} iterations"
                print(log)
                result.append(log)

        except Exception as e:
            error_msg.append(f"ERROR processing problem {index + 1}/{total_problems}: {problem['name']}, Error: {str(e)}")

    with open(file, 'w') as file, Tee(file):
        for ex in result:
            print(ex)
        for err in error_msg:
            print(err)

def main():
    #init arguments
    args = parse_args()
    #init model
    base_model_name = args.model_name
    adapter_path = "../adapter/"
    img_model_name="openbmb/MiniCPM-V-2_6"
    model, tokenizer = load_model_and_tokenizer(base_model_name, adapter_path, lora=args.fine_tuned)
    img_model, img_tokenizer = load_image_model_and_tokenizer(img_model_name)
    with open(args.out, 'w') as f, Tee(f):
        if args.fine_tuned:
            print(f"Using model: {base_model_name} with Lora adapter fine_tuned by Phat")
        else:
            print(f"🚀 Launching central controller: {base_model_name}")
    
        print(f"🔍 Initializing wavelength perception: {img_model_name}")
        # Extract problem cases
        if args.dataset_local_path:  # handle local dataset (folder structured)
            problem_cases = extract_problem_cases_from_folder(args.dataset_local_path)
            if args.local_ds_idx is not None:
                problem_cases = [problem_cases[args.local_ds_idx]]
                print(f"Processing specific problem: {problem_cases[0]['name']}")
            else:
                print(f"Processing all {len(problem_cases)} problems in the folder")
        else:  # handle hf dataset
            ds = load_dataset("hackercupai/hackercup")
            if args.problem_name:
                problem_cases = extract_problem_cases_from_hf(ds, start_index=args.start_index, end_index=args.end_index, problem_name=args.problem_name)
                if not problem_cases:
                    print(f"No problem found with the name '{args.problem_name}'")
                    return
                print(f"Processing specific problem: {problem_cases[0]['name']}")
            else:
                problem_cases = extract_problem_cases_from_hf(ds, start_index=args.start_index, end_index=args.end_index)
                print(f"Processing {len(problem_cases)} problems from the dataset")

        # Process problems sequentially
        process_problems_sequentially(model, tokenizer, img_model, img_tokenizer , args.result_out, problem_cases, args.code_iterations, args.max_num_retry, args.num_refinement ,args.show_coT)

        print("All processing finished.")

if __name__ == "__main__":
    main()


# python main.py --code_iterations 15 --max_num_retry 5 --dataset_local_path "contest_data" --show_coT

#python main.py --code_iterations 10 --max_num_retry 5 --dataset_local_path "contest_data" --show_coT --out "output1.txt"

#python main.py --problem_name "cheeseburger_corollary_ch1" --fine_tuned --show_coT
#python main.py --code_iterations 30 --num_refinement 15 --problem_name "cheeseburger_corollary_ch1" --show_coT --model_name "Qwen/Qwen2.5-72B" --out ../r2_contest_data/outputcheesebg1-7b.txt --result_out ../r2_contest_data/resultcheesebg1-7b.txt

# python main.py --code_iterations 15 --num_refinement 15  --dataset_local_path ../r2_contest_data/ --show_coT --out ../r2_contest_data/output7b.txt --result_out ../r2_contest_data/result7b.txt --model_name "Qwen/Qwen2.5-72B-Instruct"

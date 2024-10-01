import json
import contextlib
import io
from unittest.mock import patch
import time
import re
import traceback
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import subprocess
import time
import multiprocessing as mp  # Import multiprocessing
import sys

from p import (
    get_problem_understanding_template,
    analyze_original_test_cases_template,
    generate_ai_test_cases_prompt,
    get_solution_ideas_template,
    evaluate_solutions_template,
    get_code_generation_template,
    iterate_public_tests
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full process for solving coding problems.")
    parser.add_argument("--code_iterations", type=int, default=5, help="Number of code improvement iterations.")
    parser.add_argument("--max_num_retry", type=int, default=5, help="Maximum number of retries for model responses.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (equal to the number of GPUs).")
    parser.add_argument("--problem_name", type=str, default=None, help="Specify the name of the problem to solve.")
    return parser.parse_args()

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", temperature=0.3, do_sample = True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply chat template for all messages
def apply_chat_template(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Function to interact with the model and return the latest response using chat template
def generate_response(messages, max_new_tokens=2048):
    full_prompt = apply_chat_template(messages)
    
    model_inputs  = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Helper to parse response at each step
def model_response(user_content, system_prompt="You are a helpful assistant whose job is to produce only valid JSON format in every response without any additional text, explanations, or comments. You must always produce correct JSON format including comma, parentheses,etc. If asked to provide information, always structure the output in the JSON format specified by the user. Never include any output outside of the JSON format."):
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content}
    ]
    response = generate_response(messages)
    formatted_response = {"role": "assistant", "content": response}
    print(f"{formatted_response['content']}")
    return formatted_response["content"]

# Load dataset
ds = load_dataset("hackercupai/hackercup")

# Extract problem cases and include sample_input and sample_output in the problem_description
def extract_problem_cases_with_io(dataset):
    problem_cases = []
    for example in dataset['full']:
        sample_input = example["sample_input"]
        sample_output = example["sample_output"]
        
        # Format the problem description
        problem_description = f"""
        {example['statement']}
        
        ### Sample Input
        {sample_input}

        ### Sample Output
        {sample_output}
        """

        # Append the formatted problem description to the list of problems
        problem_cases.append({
            "name": example["name"],
            "year": example["year"],
            "round": example["round"],
            "problem_description": problem_description,
            "sample_input": sample_input,
            "sample_output": sample_output
        })
    return problem_cases


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
        print(f"Error decoding JSON: {e}")
        return None  # Return None if parsing fails
# Retry function to retry any function that uses response_json with added try-except for resilience
def retry(func, max_attempts, *args, **kwargs):
    attempts = 0
    result = None

    while attempts < max_attempts:
        try:
            raw_response = func(*args, **kwargs)
            parsed_response = response_json(raw_response)
            
            if parsed_response is not None and isinstance(parsed_response, dict):
                return parsed_response
        except Exception as e:
            pass  # Silently handle errors and retry

        attempts += 1
    
    return None  # Return None to signal failure

### Helper functions to iterate the code solutions
def extract_python_code(response):
    return response

def check_code_structure(extracted_code):
    """Check if the code contains function definitions and the main block."""
    if '__name__ == "__main__"' not in extracted_code:
        return False, "Missing `if __name__ == \"__main__\":` block."
    
    if 'def ' not in extracted_code:
        return False, "No function definitions found in the code."
    
    return True, None

def run_extracted_code(extracted_code, test_input):
    # Check if test_input is None
    if test_input is None:
        return None, "Error: test_input is None"
    
    # Check if the structure of the code is valid
    is_valid, error_message = check_code_structure(extracted_code)
    if not is_valid:
        return None, f"Code structure error: {error_message}"
    
    output = io.StringIO()
    error = None
    test_input_lines = [line.strip() for line in test_input.strip().split('\n') if line.strip()]

    def mock_input():
        if not test_input_lines:
            raise ValueError("Not enough input data provided")
        return test_input_lines.pop(0)

    local_scope = {'__name__': '__main__'}
    
    with patch('builtins.input', mock_input), contextlib.redirect_stdout(output):
        try:
            # Compile the code object
            code_obj = compile(extracted_code, '<string>', 'exec')
            
            # Execute the compiled code
            exec(code_obj, local_scope)
            
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            
            # Get the line number where the error occurred
            tb = traceback.extract_tb(exc_traceback)
            line_no = tb[-1].lineno
            
            # Get the line of code that caused the error
            code_lines = extracted_code.split('\n')
            error_line = code_lines[line_no - 1] if line_no <= len(code_lines) else "Unknown"
            
            error = f"Error occurred on line {line_no}: {exc_type.__name__}: {str(exc_value)}\n"

    return output.getvalue(), error


# Compare the output generated by the code with the expected output
def compare_with_expected_output(generated_output, expected_output):
    generated_output_lines = generated_output.strip().splitlines()
    expected_output_lines = expected_output.strip().splitlines()

    total_cases = len(expected_output_lines)
    matching_cases = 0
    failed_cases = []
    
    for i, (generated_line, expected_line) in enumerate(zip(generated_output_lines, expected_output_lines), start=1):
        if generated_line.strip() == expected_line.strip():
            matching_cases += 1
        else:
            failed_cases.append(f"Test Case #{i}: Expected '{expected_line}' but got '{generated_line}'")

    score = (matching_cases / total_cases) * 100 if total_cases > 0 else 0
    return score, failed_cases

# Evaluate generated code on test cases
def evaluate_generated_code_on_test_cases(extracted_code, test_input, test_output):
    generated_output, error = run_extracted_code(extracted_code, test_input)
    
    if not generated_output.strip():
        return 0, error, generated_output, []
    
    if error:
        return 0, error, generated_output, []
    
    score, failed_cases = compare_with_expected_output(generated_output, test_output)
    return score, error, generated_output, failed_cases


#BESIDES THE CONTENT, IT IS POSSIBLE PARSE SYSTEM PROMPT HERE

def understanding_problem(problem_description): 
    print("Step 1: Understanding problem:")
    return model_response(get_problem_understanding_template(problem_description))

def analyze_test_cases(problem_description):
    print("Step 2: Analyzing test cases: ")
    return model_response(analyze_original_test_cases_template(problem_description))

def self_generate_test_cases(problem_description,test_case_analysis):
    print("Step 3: Generate more sample test cases")
    return model_response(generate_ai_test_cases_prompt(problem_description,test_case_analysis))

def generate_solution_ideas(problem_description,test_case_analysis , num_solutions):
    print("Step 4: Generate solutions")
    return model_response(get_solution_ideas_template(problem_description, test_case_analysis,num_solutions))

def evaluate_solutions_f(solution_ideas, problem_understanding, problem_difficulty):
    print("Step 5: Evaluating solutions: ")
    return model_response(evaluate_solutions_template(solution_ideas,problem_understanding, problem_difficulty))

def generate_python_code(selected_solution, test_case_analysis):
    print("Step 6: First python code: ")
    return model_response(get_code_generation_template(selected_solution, test_case_analysis))

def request_code_improvement(generated_code, error_message):
    return model_response(iterate_public_tests(generated_code, error_message))

# Main function to run the process
def run_full_process(problem_description, test_input, test_output, code_iterations=5, max_num_retry=5):
    try:
        # Step 1: Understand the problem
        understand = retry(understanding_problem, max_num_retry, problem_description)
        if not understand:
            return None

        # Step 2: Analyze test cases
        analysis = retry(analyze_test_cases, max_num_retry, problem_description)
        if not analysis:
            return None

        # Step 3: Generate AI test cases
        ai_test = retry(self_generate_test_cases, max_num_retry, problem_description, response_json(analysis)['original_test_case_analysis'])
        if not ai_test:
            return None

        # Step 4: Generate solution ideas
        solutions = retry(generate_solution_ideas, max_num_retry, problem_description, response_json(analysis)['original_test_case_analysis'], num_solutions=3)
        if not solutions:
            return None

        # Step 5: Evaluate solutions
        evaluate_solutions = retry(evaluate_solutions_f, max_num_retry, response_json(solutions)['solutions'], response_json(understand)['understanding'], response_json(understand)['understanding']['difficulty_assessment'])
        if not evaluate_solutions:
            return None

        # Step 6: Generate Python code
        code_solution = retry(generate_python_code, max_num_retry, response_json(evaluate_solutions)['selected_solution'], response_json(analysis)['original_test_case_analysis'])
        if not code_solution:
            return None

        generated_code = extract_python_code(code_solution['solution_code']['code'])
        attempts = 0
        best_score = 0
        best_code = generated_code


        # Start the code iteration loop
        while attempts < code_iterations:
            print(f"Code iterations. Attempt #{attempts+1}/{code_iterations}")

            # Run the generated code
            score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
                generated_code, test_input=test_input, test_output=test_output
            )

            # Log the output and error for debugging purposes
            if error:
                print(f"Error during code execution: {error}")
            elif failed_cases:
                print(f"Failed cases: {failed_cases}")

            # If this score is better than the previous best, update the best result
            if score > best_score:
                best_score = score
                best_code = generated_code

            # If we achieve a perfect score, stop and return the best code
            if best_score == 100:
                print(f"Perfect score achieved: {best_score}%")
                return best_code

            # Improvement feedback is empty, continue to the next iteration
            improvement_feedback = error if error else failed_cases

            # print(f"Improvement feedback: {improvement_feedback}")

            # Retry the code improvement
            new_code = retry(request_code_improvement, max_num_retry, generated_code, improvement_feedback)
            if not new_code:
                print("Error in 'request_code_improvement'. Returned None.")
                break
            generated_code = new_code
            attempts += 1

        # After max iterations, return the best result so far if it exists
        if best_score > 0:
            print(f"Returning best code with score {best_score}% after {attempts} attempts.")
            return best_code
        else:
            print("No valid solution generated.")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None



# Process a batch of problems on a specific GPU
def process_problems_on_gpu(gpu_id, problem_batch, code_iterations, max_num_retry):
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)

    # Process each problem
    for problem in problem_batch:
        try:
            problem_description = problem["problem_description"]
            input_data = problem["sample_input"]
            expected_output = problem["sample_output"]

            print(f"Running problem: {problem['name']} from year {problem['year']} round {problem['round']} on GPU {gpu_id}")
            
            generated_code = run_full_process(
                problem_description,
                input_data,
                expected_output,
                code_iterations=code_iterations,
                max_num_retry=max_num_retry
            )

            if generated_code:
                score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
                    generated_code, input_data, expected_output
                )

                if score > 0:
                    print(f"Problem: {problem['name']} passed with score {score}% on GPU {gpu_id}!")
                else:
                    print(f"Problem: {problem['name']} failed with errors on GPU {gpu_id}.")
            else:
                print(f"Could not generate valid code for problem {problem['name']} on GPU {gpu_id}.")
                return None
        except Exception as e:
            print(f"Error processing problem {problem['name']} on GPU {gpu_id}: {e}")


# Main function to run the process
def main():
    args = parse_args()

    # Load the dataset
    ds = load_dataset("hackercupai/hackercup")

    # Extract problem cases from the current split
    problem_cases = extract_problem_cases_with_io(ds)

    # Iterate to find the exact problem using it's name when trying to solve 1 particular problem
    if args.problem_name:
        problem_cases = [problem for problem in problem_cases if problem['name'].lower() == args.problem_name.lower()]
        if not problem_cases:
            print(f"No problem found with the name '{args.problem_name}'")
            return None

    # 1 problem
    if len(problem_cases) == 1:
        # Directly assign the problem to a single worker without splitting
        problem_batches = [problem_cases]
        num_workers = 1
    else: #entrie dataset
        # Split the problem cases into batches for each GPU
        num_workers = min(args.num_workers, len(problem_cases))  # Limit workers to the number of problems
        batch_size = max(1, len(problem_cases) // num_workers)  # Ensure at least one case per batch
        problem_batches = [problem_cases[i:i + batch_size] for i in range(0, len(problem_cases), batch_size)]

    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Create multiprocessing workers (one for each GPU)
    processes = []
    for i in range(num_workers):
        gpu_id = i  # Each worker uses a different GPU
        problem_batch = problem_batches[i]

        p = mp.Process(target=process_problems_on_gpu, args=(gpu_id, problem_batch, args.code_iterations, args.max_num_retry))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()



# python r.py --problem_name "cheeseburger_corollary_ch2" --code_iterations 100 --max_num_retry 10 --num_workers 4

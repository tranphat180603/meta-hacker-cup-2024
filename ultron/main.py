import os
import json
import contextlib
import io
from unittest.mock import patch
import time
import re
import traceback

import argparse
import subprocess
import time
import multiprocessing as mp  # Import multiprocessing
import sys

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from prompts import (
    get_problem_understanding_template,
    analyze_original_test_cases_template,
    generate_ai_test_cases_prompt,
    get_solution_ideas_template,
    evaluate_solutions_template,
    get_code_generation_template,
    iterate_execution_error,
    refine_problem_understanding_template,
    iterate_failed_test_cases
)

from model import (
    load_model_and_tokenizer,
    generate_response
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
    parser.add_argument("--code_iterations", type=int, default=5, help="Number of code improvement iterations.")
    parser.add_argument("--max_num_retry", type=int, default=5, help="Maximum number of retries for model responses.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (equal to the number of GPUs).")
    parser.add_argument("--problem_name", type=str, default=None, help="Specify the name of the problem to solve for hf dataset")
    parser.add_argument("--show_coT", action="store_true", help="Show the Chain of Thought output for debugging.")
    parser.add_argument("--dataset_local_path", type = str, default = "", help = "if specified, open dataset in local machine, problem is formatted the same as online dataset") 
    parser.add_argument("--local_ds_idx", type = int, help = "if specified, solve particular problem in the folder")
    parser.add_argument("--lora", action="store_true", help="flag to use my fine-tuned version")
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
        return None  # Return None if parsing fails
# Retry function to retry any function that uses response_json with added try-except for resilience
def retry(func, max_attempts, *args, **kwargs):
    attempts = 0
    result = None

    while attempts < max_attempts:
        # print(f"Attempt: {attempts + 1}")
        try:
            raw_response = func(*args, **kwargs)
            parsed_response = response_json(raw_response)
            
            if parsed_response is not None and isinstance(parsed_response, dict):
                return parsed_response
        except Exception as e:
            print("Error at function retry")

        attempts += 1
    
    return None  # Return None to signal failure

#call response
def understanding_problem(model, tokenizer, problem_description, show_coT=False): 
    try:
        if show_coT:
            print("Step 1: Understanding problem:")
        return model_response(model, tokenizer, get_problem_understanding_template(problem_description), show_coT=show_coT ,system_prompt = """
        You are an AI assistant specializing in analyzing and structuring programming problem descriptions. 
        Produce only valid JSON based on the provided structure without extra text or explanations. 
        Maintain real-world logical consistency while interpreting the problem, and note any ambiguities or inconsistencies in the description. 
        (For example: a pair of chopsticks can't be 1 chopstick, a dog can't have 3 legs)
        """, temperature = 0.3)
    except Exception as e:
        print(f"Error in understanding_problem: {str(e)}")
        return None

def analyze_test_cases(model, tokenizer, problem_description, show_coT=False):
    try:
        if show_coT:
            print("Step 2: Analyzing test cases: ")
        return model_response(model, tokenizer ,analyze_original_test_cases_template(problem_description),show_coT=show_coT ,system_prompt = """
        You are a specialized assistant tasked with analyzing original test cases from a given problem description. 
        Your job is to extract the input and output format, map each component to its corresponding variable, and explain how the inputs lead to the output. 
        Produce only valid JSON based on the provided structure without extra text or explanations.
        """)
    except Exception as e:
        print(f"Error in analyze_test_cases: {str(e)}")
        return None

def get_refine_understanding(model, tokenizer, problem_understanding, test_case_analysis, show_coT=False):
    try:
        if show_coT:
            print("Step 3: Refine problem understandings: ")
        return model_response(model, tokenizer ,refine_problem_understanding_template(problem_understanding, test_case_analysis),show_coT=show_coT ,system_prompt = """
        Refine the problem understanding by integrating insights from test case analysis. 
        Update constraints, identify edge cases, and resolve discrepancies between initial understanding and test cases. 
        Provide the refined understanding in valid JSON format only.
        """, temperature = 0.2)
    except Exception as e:
        print(f"Error in analyze_test_cases: {str(e)}")
        return None

def self_generate_test_cases(model, tokenizer, problem_description, test_case_analysis, show_coT=False):
    try:
        if show_coT:
            print("Step 4: Generate more sample test cases")
        return model_response(model, tokenizer ,generate_ai_test_cases_prompt(problem_description, test_case_analysis), show_coT=show_coT,system_prompt = """
        You are an AI test case generator. Your task is to produce diverse and challenging test cases, including edge cases, based on the provided problem description and analysis.
        Ensure your output strictly follows the requested JSON structure, without adding any extra text or explanations.
        """)
    except Exception as e:
        print(f"Error in self_generate_test_cases: {str(e)}")
        return None

def generate_solution_ideas(model, tokenizer, problem_description, test_case_analysis, num_solutions, show_coT=False):
    try:
        if show_coT:
            print("Step 5: Generate solutions")
        return model_response(model, tokenizer ,get_solution_ideas_template(problem_description, test_case_analysis, num_solutions), show_coT=show_coT,system_prompt = """
        As an innovative problem solver, generate diverse and creative solution ideas for the given programming problem. 
        Think outside the box while ensuring all solutions can pass the provided test cases.
        Aim for a mix of conventional and novel approaches, considering efficiency, scalability, and unique algorithmic techniques.
        """, temperature = 0.8)
    except Exception as e:
        print(f"Error in generate_solution_ideas: {str(e)}")
        return None

def evaluate_solutions_f(model, tokenizer, solution_ideas, refine_problem_understanding, test_case_analysis, problem_difficulty, show_coT=False):
    try:
        if show_coT:        
            print("Step 6: Evaluating solutions: ")
        return model_response(model, tokenizer ,evaluate_solutions_template(solution_ideas, refine_problem_understanding, test_case_analysis, problem_difficulty), show_coT=show_coT,system_prompt = """
        Critically evaluate the provided solution ideas against the refined problem understanding and test cases. 
        Select the optimal solution considering code simplicity, robustness, efficiency, and scalability relative to the problem's difficulty. 
        Provide a concise, objective assessment in the specified JSON format only.
        """, temperature = 0.4)
    except Exception as e:
        print(f"Error in evaluate_solutions_f: {str(e)}")
        return None

def generate_python_code(model, tokenizer, selected_solution, test_case_analysis, show_coT=False):
    try:
        if show_coT:       
            print("Step 7: First python code: ")
        return model_response(model, tokenizer ,get_code_generation_template(selected_solution, test_case_analysis), show_coT=show_coT,system_prompt = """
        You are tasked with generating Python code for the selected solution that passed all test cases. 
        Your job is to provide code that strictly follows the input-output structure, divides the logic into sub-functions, and handles multiple test cases.   
        Ensure the output is strictly in the specified JSON format without any extra text or explanations.
        """, temperature=0.2)
    except Exception as e:
        print(f"Error in generate_python_code: {str(e)}")
        return None

def request_code_improvement_dte(model, tokenizer, generated_code, error_message, show_coT=False):  # Due to error (execution/runtime issue)
    try:
        if show_coT:
            print("Step 8.1: Iterating on execution error: ")
        return model_response(model, tokenizer ,iterate_execution_error(generated_code, error_message), show_coT=show_coT,system_prompt="""
        You are tasked with modifying and improving Python code to fix a specific execution or runtime error based on the provided error message. 
        Focus on addressing the issue at the indicated line and provide the improved code. 
        Ensure the output is in valid JSON format without any additional text, explanations, or comments.
        """)
    except Exception as e:
        print(f"Error in request_code_improvement_dte: {str(e)}")
        return None


def request_code_improvement_dtfc(model, tokenizer, generated_code, failed_tests, show_coT=False):  # Due to failed cases (logic/approach issue)
    try:
        if show_coT:
            print("Step 8.2: Iterating on failed test cases: ")
        return model_response(model, tokenizer ,iterate_failed_test_cases(generated_code, failed_tests), show_coT=show_coT,system_prompt="""
        Analyze the provided Python code and failed test cases to develop a fundamentally new approach. 
        Prioritize creating an entirely different solution that addresses all test cases, rather than patching the existing code.
        Return only the new, complete solution in valid JSON format, without explanations or comments.
        """, temperature=0.9)
    except Exception as e:
        print(f"Error in request_code_improvement_dtfc: {str(e)}")
        return None

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


def process_problems_sequentially(problem_cases, code_iterations, max_num_retry, show_coT, num_gpus):
    total_problems = len(problem_cases)
    
    with open('results.txt', 'w') as file:
        for index, problem in enumerate(tqdm(problem_cases, desc="Processing problems", unit="problem")):
            gpu_id = index % num_gpus  # Cycle through available GPUs
            torch.cuda.set_device(gpu_id)
            
            try:
                print(f"\nRunning problem {index + 1}/{total_problems} on GPU {gpu_id}: {problem['name']}")
                problem_description = problem["problem_description"]
                input_data = problem["sample_input"]
                expected_output = problem["sample_output"]
                
                generated_code, best_score = run_full_process(problem_description, input_data, expected_output, code_iterations, max_num_retry, show_coT=show_coT)
                
                if best_score > 0:
                    result = f"Problem {index + 1}/{total_problems}: {problem['name']}, Score: {best_score}%"
                else:
                    result = f"Problem {index + 1}/{total_problems}: {problem['name']}, Failed to find solution after {code_iterations} iterations"
                
                print(result)
                file.write(result + '\n')
                file.flush()  # Ensure the result is written immediately
                
            except Exception as e:
                error_msg = f"ERROR processing problem {index + 1}/{total_problems}: {problem['name']}, Error: {str(e)}"
                print(error_msg)
                file.write(error_msg + '\n')
                file.flush()

def main():
    args = parse_args()
    with open(args.out, 'w') as f, Tee(f):

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
        process_problems_sequentially(problem_cases, args.code_iterations, args.max_num_retry, args.show_coT, num_gpus)

        print("All processing finished.")

if __name__ == "__main__":
    main()


# python r.py --code_iterations 15 --max_num_retry 5 --dataset_local_path "contest_data" --show_coT

#python r.py --code_iterations 10 --max_num_retry 5 --dataset_local_path "contest_data" --show_coT --out "output1.txt"
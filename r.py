import json
import contextlib
import io
from unittest.mock import patch
import time
import json
import re
import traceback

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from p import (
    get_problem_understanding_template,
    analyze_original_test_cases_template,
    generate_ai_test_cases_prompt,
    get_solution_ideas_template,
    evaluate_solutions_template,
    get_code_generation_template,
    iterate_public_tests
)

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply chat template for all messages
def apply_chat_template(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Function to interact with the model and return the latest response using chat template
def generate_response(messages, max_new_tokens=2048):
    # Apply chat template to the messages before passing to the model
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
def model_response(user_content, system_prompt="You are Qwen, created by Alibaba Cloud. Your job is to produce only valid JSON format in every response without any additional text, explanations, or comments. You must always produce correct JSON format including comma, parentheses,etc. If asked to provide information, always structure the output in the JSON format specified by the user. Never include any output outside of the JSON format."):
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content}
    ]
    response = generate_response(messages)
    formatted_response = {"role": "assistant", "content": response}
    print(f"{formatted_response['content']}")
    return formatted_response["content"]

#load dataset
ds = load_dataset("hackercupai/hackercup")


# Extract problem cases and include sample_input and sample_output in the problem_description
def extract_problem_cases_with_io(dataset):
    problem_cases = []
    for example in dataset['full']:  # Assuming we're using the 'train' split
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

# Get all problem cases with input/output appended
problem_cases = extract_problem_cases_with_io(ds)


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


# Retry function to retry any function that uses response_json
def retry(func, max_attempts, *args, **kwargs):
    attempts = 0
    result = None

    while attempts < max_attempts:
        # Always get the model's response
        raw_response = func(*args, **kwargs)

        try:
            # Attempt to parse the response using response_json
            parsed_response = response_json(raw_response)
            
            # Check if the parsed response is not None and is a valid dictionary
            if parsed_response is not None and isinstance(parsed_response, dict):
                return parsed_response  # Return parsed response if successful
            else:
                print(f"Attempt {attempts + 1} failed: Unable to parse valid JSON.")
        except Exception as e:
            print(f"Error during attempt {attempts + 1}: {e}")
            

        attempts += 1
        print(f"Retrying {func.__name__}... (Attempt {attempts + 1} of {max_attempts})")
    
    raise ValueError(f"Failed to process valid JSON after {max_attempts} attempts")


###HELPER FUNCTIONS TO ITERATE THE CODE SOLUTIONS

# Extract Python code from the structured output
def extract_python_code(response):
    return response

# Function to execute the extracted python code in memory with mocked input()
def run_extracted_code(extracted_code, test_input):
    output = io.StringIO()
    error = None
    test_input_lines = iter(test_input.splitlines())
    with patch('builtins.input', lambda: next(test_input_lines)), contextlib.redirect_stdout(output):
        try:
            exec(extracted_code, {"__name__": "__main__"})
        except Exception as e:
            # Capture the full traceback to provide more detail about where the error occurred
            error = traceback.format_exc()
    print(f" Model generated_output: {output.getvalue()}")
    return output.getvalue(), error

# Compare the output generated by the code with the expected output, case by case
def compare_with_expected_output(generated_output, expected_output):
    generated_output_lines = generated_output.strip().splitlines()
    expected_output_lines = expected_output.strip().splitlines()

    # Calculate total test cases and matching cases
    total_cases = len(expected_output_lines)
    matching_cases = 0
    failed_cases = []
    
    # Log the output comparison between generated and expected
    print("\n=== Output Comparison ===")
    
    for i, (generated_line, expected_line) in enumerate(zip(generated_output_lines, expected_output_lines), start=1):
        print(f"Test Case #{i}:\nGenerated Output: {generated_line}\nExpected Output: {expected_line}")
        if generated_line.strip() == expected_line.strip():
            matching_cases += 1
        else:
            failed_cases.append(f"Test Case #{i}: Expected '{expected_line}' but got '{generated_line}'")
    
    print("\n=== End of Comparison ===\n")

    # Calculate score as a percentage
    score = (matching_cases / total_cases) * 100 if total_cases > 0 else 0

    return score, failed_cases




def evaluate_generated_code_on_test_cases(extracted_code, test_input, test_output):
    
    # Run the extracted code with the provided test input
    generated_output, error = run_extracted_code(extracted_code, test_input)
    
    # Check if generated_output is empty
    if not generated_output.strip():
        return 0, error, generated_output, []
    
    # Check if there was an error during execution
    if error:
        return 0, error, generated_output, []
    
    # Compare the generated output with the expected output
    score, failed_cases = compare_with_expected_output(generated_output, test_output)
    
    print(f"Evaluation completed. Score: {score}%")
    
    # Return the score, error if any, generated output, and failed cases
    return score, error, generated_output, failed_cases


#BESIDES THE CONTENT, IT IS POSSIBLE PARSE SYSTEM PROMPT HERE

def understanding_problem(problem_description): 
    print("Step 1: Understanding problem:")
    return model_response(get_problem_understanding_template(problem_description))

def analyze_test_cases(problem_description):
    print("Step 2: Analyzing test cases: ")
    return model_response(analyze_original_test_cases_template(problem_description))

def self_generate_test_cases(test_case_analysis):
    print("Step 3: Generate more sample test cases")
    return model_response(generate_ai_test_cases_prompt(problem_description,test_case_analysis))

def generate_solution_ideas(problem_description,test_case_analysis , num_solutions):
    print("Step 4: Generate solutions")
    return model_response(get_solution_ideas_template(problem_description, test_case_analysis,num_solutions))

def evaluate_solutions_f(solution_ideas, test_case_analysis, problem_difficulty):
    print("Step 5: Evaluating solutions: ")
    return model_response(evaluate_solutions_template(solution_ideas,test_case_analysis, problem_difficulty))

def generate_python_code(selected_solution, test_case_analysis):
    print("Step 6: First python code: ")
    return model_response(get_code_generation_template(selected_solution, test_case_analysis))

def request_code_improvement(generated_code, error_message):
    print("Step 7: Code improvement: ")
    return model_response(iterate_public_tests(generated_code, error_message))

# Main function to run the process with chat templates, parse JSON applied at each step
def run_full_process(problem_description, test_input, test_output, code_iterations=5, max_num_retry=5):
    # Step 1: Understand the problem
    understand = retry(understanding_problem, max_num_retry, problem_description)
    
    # Step 2: Analyze test cases
    analysis = retry(analyze_test_cases, max_num_retry, problem_description)

    # Step 3: Generate more test cases based on observation (AI-generated test cases)
    ai_test = retry(self_generate_test_cases, max_num_retry, response_json(analysis)['original_test_case_analysis'])

    # Step 4: Generate solution ideas with increasing complexity
    solutions = retry(generate_solution_ideas, max_num_retry, problem_description, response_json(analysis)['original_test_case_analysis'], num_solutions=3)

    # Step 5: Evaluate solutions based on problem difficulty
    evaluate_solutions = retry(evaluate_solutions_f, max_num_retry, response_json(solutions)['solutions'], response_json(understand)['understanding'], response_json(understand)['understanding']['difficulty_assessment'])

    # Step 6: Generate Python code based on the best solution
    code_solution = retry(generate_python_code, max_num_retry, response_json(evaluate_solutions)['selected_solution'], response_json(analysis)['original_test_case_analysis'])
    generated_code = extract_python_code(code_solution['solution_code']['code'])
    
    attempts = 0
    code_passes = False
    last_ver_code = generated_code

    while attempts < code_iterations and not code_passes:
        # Step 7: Evaluate the generated code with original test cases
        score, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(generated_code, test_input=test_input, test_output=test_output)

        print(f"Attempt {attempts + 1}: Evaluation completed. Score: {score}%")
        
        # Check if code passed
        if score > 0:  
            return generated_code

        # Log errors or failed cases and request code improvement
        if error:
            print(f"Execution Error: {error}")
            improvement_feedback = error  # If execution error, pass it for improvement
        else:
            print("Test cases failed:")
            improvement_feedback = failed_cases  # Pass the failed test cases as feedback for improvement
            print(f"Failed cases:\n{failed_cases}")

        # Step 8: Request code modification based on error or failed cases
        new_code = retry(request_code_improvement, max_num_retry, last_ver_code, improvement_feedback)

        # Extract the new code and check if it's different
        extracted_code = extract_python_code(new_code['solution_code']['code'])

        if extracted_code != last_ver_code:
            generated_code = extracted_code
            last_ver_code = extracted_code
            print("Made some changes to the code.")
        else:
            print("No changes made in this iteration.")
        
        attempts += 1

    # If the code doesn't pass all test cases in the allowed attempts
    print(f"Failed to generate correct code after {code_iterations} attempts.")
    return None

for problem in problem_cases:
    problem_description = problem["problem_description"]
    input_data = problem["sample_input"]
    expected_output = problem["sample_output"]
    
    print(f"Running problem: {problem['name']} from year {problem['year']} round {problem['round']}")
    
    generated_code = run_full_process(problem_description,input_data,expected_output )
    
    if generated_code:
        # If code is generated successfully, execute it
        code_passes, error, generated_output, failed_cases = evaluate_generated_code_on_test_cases(
            generated_code, input_data, expected_output
        )
        
        if code_passes:
            print(f"Code passed for problem {problem['name']}!")
        else:
            print(f"Code failed for problem {problem['name']}. Errors: {error}")
            print(f"Failed cases: {failed_cases}")
    else:
        print(f"Could not generate valid code for problem {problem['name']}.")



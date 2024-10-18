from datasets import load_dataset
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
    model_response
)

# Load the model and tokenizer
def load_model_and_tokenizer(model_name, adapter_path ,temperature=0.3, lora = False):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if lora:
        merged_model = PeftModel.from_pretrained(model, adapter_path)
        return merged_model, tokenizer
    return model, tokenizer


# Apply chat template for all messages
def apply_chat_template(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Function to interact with the model and return the latest response using chat template
def generate_response(model, tokenizer, messages, temperature=0.3, max_new_tokens=2048):
    full_prompt = apply_chat_template(tokenizer, messages)
    
    model_inputs  = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,          
        do_sample=True,                   
        pad_token_id=model.config.eos_token_id 
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# Helper to parse response at each step
def model_response(model, tokenizer, user_content, temperature=0.3, max_new_tokens=2048,show_coT = False ,system_prompt="You are a helpful assstant whose job is to produce only valid JSON format in every response without any additional text, explanations, or comments. You must always produce correct JSON format including comma, parentheses,etc. If asked to provide information, always structure the output in the JSON format specified by the user. Never include any output outside of the JSON format."):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    response = generate_response(model, tokenizer, messages, temperature=temperature, max_new_tokens=max_new_tokens)
    formatted_response = {"role": "assistant", "content": response}
    if show_coT:
        print(f"Generated Response: {formatted_response['content']}", flush=True)
    return formatted_response["content"]


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
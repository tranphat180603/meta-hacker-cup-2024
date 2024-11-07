from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, AutoModel
from peft import PeftModel, PeftConfig
import torch

from PIL import Image
import io
import base64

from prompts import (
    get_problem_understanding_template,
    get_image_understanding_prompt,
    analyze_original_test_cases_template,
    get_solution_ideas_template,
    evaluate_solutions_template,
    get_code_generation_template,
    reflect_execution_error,
    refine_problem_understanding_template,
    reflect_failed_test,
    improve_final_code_efficiency
)


# Function to decode base64 image with padding if necessary
def decode_base64_image(base64_string):
    if not base64_string:
        return None  # Return None if the image string is empty
    
    # Add padding if needed
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None
    
# Load the model and tokenizer
def load_model_and_tokenizer(model_name, adapter_path, lora = False):
    assert model_name is not None, "Must specify model_name"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if lora:
        merged_model = PeftModel.from_pretrained(model, adapter_path)
        return merged_model, tokenizer
    return model, tokenizer

# Load the image model and tokenizer
def load_image_model_and_tokenizer(model_name="openbmb/MiniCPM-V-2_6"):
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16
    )
    model = model.eval().cuda()  # Ensure model is in eval mode and on the correct device
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

# Apply chat template for all messages
def apply_chat_template(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Function to interact with the model and return the latest response using chat template
def generate_response(model, tokenizer, messages, temperature=0.5, max_new_tokens=2048):
    full_prompt = apply_chat_template(tokenizer, messages)
    
    model_inputs  = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
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
def model_response(model, tokenizer, user_content, temperature=0.5, max_new_tokens=2048,show_coT = False ,system_prompt="You are a helpful assstant whose job is to produce only valid JSON format in every response without any additional text, explanations, or comments. You must always produce correct JSON format including comma, parentheses,etc. If asked to provide information, always structure the output in the JSON format specified by the user. Never include any output outside of the JSON format."):

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
        """)
    except Exception as e:
        print(f"Error in understanding_problem: {str(e)}")
        return None

def understanding_image(img_model, tokenizer, problem_description, img_raw, show_coT=False):
    try:
        # Generate the question using the problem description
        question = get_image_understanding_prompt(problem_description)
        
        # Decode the image
        image_bytes = decode_base64_image(img_raw)
        if image_bytes is None:
            image_bytes = ""
            return image_bytes
        
        # Load and convert the image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Prepare messages for the model
        msgs = [{'role': 'user', 'content': [image, question]}]
        
        # Query the model
        res = img_model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        
        # Display the response if needed
        if show_coT:
            print(f"Step 2: Image understanding (if the problem has image):\n{res}")
        
        return res  # Return the model's response with image information
    
    except Exception as e:
        print(f"Error in understanding_image: {str(e)}")
        return None

def analyze_test_cases(model, tokenizer, problem_description, reflection, show_coT=False):
    try:
        if show_coT:
            print("Step 3: Analyzing test cases: ")
        return model_response(model, tokenizer ,analyze_original_test_cases_template(problem_description, reflection),show_coT=show_coT ,system_prompt = """
You are a specialized assistant tasked with analyzing original test cases from a given problem description. 
Your job is to extract the input and output format, map each component to its corresponding variable, and explain how the inputs lead to the output. 
Produce only valid JSON based on the provided structure without extra text or explanations.
        """)
    except Exception as e:
        print(f"Error in analyze_test_cases: {str(e)}")
        return None

def get_refine_understanding(model, tokenizer, problem_understanding, test_case_analysis, reflection, img_understanding,show_coT=False):
    try:
        if show_coT:
            print("Step 4: Refine problem understandings: ")
        return model_response(model, tokenizer, refine_problem_understanding_template(problem_understanding, test_case_analysis, reflection=reflection, img_understanding = img_understanding), show_coT=show_coT, system_prompt="""
Task: Refine your understanding of the problem by integrating key insights from various sources.
Your primary objective is to create a cohesive understanding by combining:
1. The initial problem statement and constraints.
2. Observations from analyzing test cases.
3. Important visual details from the image relevant to the problem.

Consider:
- Look for any visual patterns or elements in the image that could impact or provide constraints to the solution.
- Apply patterns identified from test case analysis to find possible edge cases or hidden requirements.
- If there are insights from previous reflections, apply them to avoid repeating common errors.

Output Requirements:
- Provide the refined problem understanding in JSON format only, ensuring all updates are clearly reflected.
""")
    except Exception as e:
        print(f"Error in analyze_test_cases: {str(e)}")
        return None

def generate_solution_ideas(model, tokenizer, problem_description, test_case_analysis, num_solutions, show_coT=False):
    try:
        if show_coT:
            print("Step 5: Generate solutions")
        return model_response(model, tokenizer ,get_solution_ideas_template(problem_description, test_case_analysis, num_solutions), show_coT=show_coT,system_prompt = """
As an innovative problem solver, generate diverse and creative solution ideas for the given programming problem. 
Think outside the box while ensuring all solutions can pass the provided test cases.
Aim for a mix of conventional and novel approaches, considering efficiency, scalability, and unique algorithmic techniques.
Output only valid JSON in the specified format.
        """)
    except Exception as e:
        print(f"Error in generate_solution_ideas: {str(e)}")
        return None

def evaluate_solutions_f(model, tokenizer, solution_ideas, refine_problem_understanding, test_case_analysis, show_coT=False):
    try:
        if show_coT:        
            print("Step 6: Evaluating solutions: ")
        return model_response(model, tokenizer ,evaluate_solutions_template(solution_ideas, refine_problem_understanding, test_case_analysis), show_coT=show_coT,system_prompt = """
Critically evaluate the provided solution ideas against the refined problem understanding and test cases. 
Select the optimal solution considering code simplicity, robustness, efficiency, and scalability relative to the problem's difficulty. 
Provide a concise, objective assessment in the specified JSON format only.
        """)
    except Exception as e:
        print(f"Error in evaluate_solutions_f: {str(e)}")
        return None

def generate_python_code(model, tokenizer, selected_solution, test_case_analysis, refine_problem_understanding, show_coT=False):
    try:
        if show_coT:
            print("Step 7: Generating first solution code: ")
        return model_response(
            model,
            tokenizer,
            get_code_generation_template(selected_solution, test_case_analysis, refine_problem_understanding),
            show_coT=show_coT,
            system_prompt="""
Act as an independent, autonomous coding agent tasked with solving the problem effectively. Focus solely on implementing a functional solution that meets the problem requirements and passes all test cases.

Guidelines:
1. Develop Python code that handles multiple test cases in the specified input-output structure.
2. Avoid error handling or comments, and don’t include explanations—produce only essential code.
3. Structure code logically, using sub-functions where appropriate to streamline logic and readability.
4. Output only valid JSON in the specified format.

Output only valid JSON in the specified format.
            """,
        )
    except Exception as e:
        print(f"Error in generate_python_code: {str(e)}")
        return None

def request_improvement_dte(model, tokenizer, generated_code, error_message, analysis, error_history ,show_coT=False):  # Due to error (execution/runtime issue)
    try:
        if show_coT:
            print("Step 8.1: Iterating on execution error:")
        return model_response(
            model, 
            tokenizer, 
            reflect_execution_error(generated_code, error_message, analysis, error_history), 
            show_coT=show_coT, 
            system_prompt="""
Act as an independent, autonomous coding agent tasked with solving the problem effectively. Focus solely on reflecting and propose a change on the Python code by focusing on the specific execution error identified in the error message. 

- Address the line causing the error and prevent similar issues, especially those with multiple occurrences in the error history.
- Use the test case analysis and error history to improve the code’s robustness.

Respond in JSON format only, with the corrected code and explanations according to the provided structure.
""", 
        )
    except Exception as e:
        print(f"Error in request_improvement_dte: {str(e)}")
        return None

def request_improvement_dtfc(model, tokenizer, generated_code, failed_tests, refine_problem_understanding, failure_history, used_solution, show_coT=False):  # Due to failed cases (logic/approach issue)
    try:
        if show_coT:
            print("Step 8.2: Iterating on failed test cases:")
        return model_response(
            model, 
            tokenizer, 
            reflect_failed_test(generated_code, failed_tests, refine_problem_understanding, failure_history, used_solution), 
            show_coT=show_coT, 
            system_prompt="""
Act as an independent, autonomous coding agent with the task of thoroughly solving the problem. Focus on reflecting on all failure points and propose a new, comprehensive solution to address the issues from the failed test cases, prioritizing fresh strategies over incremental fixes.

Guidelines:
1. **Deep Analysis of Failures**: Analyze all failed cases in detail to identify root causes. Use `failure_history` to focus on issues that have recurred and those that have persisted through multiple solutions.
2. **Explore and Experiment**: Develop a fundamentally new approach that differs from any previously attempted solutions listed in `used_solution`. Try different techniques, structures, or logic to avoid previous mistakes and comprehensively address the problem’s requirements.
3. **Creativity and Innovation**: Do not hesitate to apply a fresh perspective, aiming for a holistic solution. Prioritize simplicity and robustness, keeping the problem’s constraints in mind.
4. **Avoid Patches**: This solution should not be a patch or incremental improvement but a clean, redesigned approach that addresses the root of each identified issue.

Return your new approach in the following JSON format, with no additional comments or explanations.
""",
            temperature=0.9 
        )
    except Exception as e:
        print(f"Error in request_improvement_dtfc: {str(e)}")
        return None

def request_final_improvement(model, tokenizer, generated_code, refine_problem_understanding, timeout_msg ,show_coT=False):
    try:
        if show_coT:
            print("Step 9: Final attempt to improve the code!")
            return model_response(
                model, 
                tokenizer, 
                improve_final_code_efficiency(generated_code, refine_problem_understanding, timeout_msg), 
                show_coT=show_coT, 
                system_prompt="""
Act as an independent, autonomous coding agent tasked with solving the problem effectively. The current solution needs transformative changes to handle large inputs effectively. Small, incremental improvements are not enough.

Your goal is to:
- Rethink the problem approach entirely, aiming for groundbreaking efficiency.
- Focus on discovering entirely new algorithms, logical simplifications, and optimal data structures to minimize computation.
- Remove bottlenecks by exploring alternatives to nested loops, brute-force methods, or redundant calculations.

This process is about reimagining the solution, not just minor tweaks. Aim for a revolutionary change in approach.

Please provide your response in the following JSON format, with no extra text outside the JSON.
""",
            )
    except Exception as e:
        print(f"Error in request_final_improvement: {str(e)}")
        return None



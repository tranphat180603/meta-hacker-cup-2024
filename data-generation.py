from datasets import load_dataset, concatenate_datasets
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the dataset
ds = load_dataset("BEE-spoke-data/code_contests_instruct", "default")

# Combine all splits into one dataset
combined_ds = concatenate_datasets([ds['train'], ds['test'], ds['valid']])

# Filter to only include entries where 'language' == 'PYTHON3'
python_ds = combined_ds.filter(lambda example: example['language'] == 'PYTHON3')


model_name = "arcee-ai/Llama-3.1-SuperNova-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", pad_token_id=tokenizer.eos_token_id)


#do as warning says
tokenizer.eos_token_id = 128001
# Set the pad_token_id to eos_token_id for both tokenizer and model
model.config.pad_token_id = tokenizer.eos_token_id  # Update model's config

# Apply chat template for all messages
def apply_chat_template(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_response(messages, max_new_tokens=2048):
    full_prompt = apply_chat_template(messages)
    
    model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id= tokenizer.eos_token_id  # Use pad_token_id from tokenizer
    )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = generated_text[len(full_prompt):].strip()
    return response

import json

def model_response(user_content, system_prompt="You are a helpful assistant."):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    response = generate_response(messages)
    return {"instruction": user_content, "output": response, "system": system_prompt, "history": []}


def generate_prompt(description, solution):
    prompt = f"""
    You are a competitive programming expert. Your task is to break down the problem-solving approach into detailed, structured steps. Finally write clear comments for the lines of code.

    Problem Description:
    {description}

    Solution:
    ```python
    {solution}
    ```

    Your task:
    1. **Problem Understanding**: Briefly summarize what the problem is asking and clarify any important constraints or requirements.
    2. **Plan/Approach**: Outline a clear plan or algorithm to solve the problem, mentioning key steps such as input parsing, mathematical formulas, loops, or condition checks.
    3. **Edge Cases**: Mention any edge cases that need to be considered (e.g., boundary conditions, special inputs like zero, large numbers, etc.).
    4. **Input and Output Format**: Pay special attention to the expected input and output format. Clearly explain how the input should be parsed, and ensure that the output matches the expected format (e.g., integer vs float, proper rounding, multiple test cases, etc.).
    5. **Rewrite the code with clear comments labeled by you.
    """
    return prompt


# Create an empty list to store results
results = []

instruction_prefix = "Your task is to solve the following competitive programming problem. Go step by step, explain clearly your approach in solving the problem. Finally, write Python code to solve that.\n\n"

# Iterate over the filtered dataset and generate responses
for example in python_ds:
    description = example['description'] #problem statement
    solution = example['solution']# problem solution (only code)
    
    # Manually create a prompt using the problem description and solution
    user_prompt = generate_prompt(description, solution)
    
    # Generate a response from the model
    messages = [{"role": "user", "content": user_prompt}]
    response = generate_response(messages) #output for fine-tuning
    
    # Store the response in the Alpaca format
    results.append({
        "instruction": description,         # Human instruction (problem description)
        "output": response,                 # Model response (steps and comments)
        "system": "You are a competitive programming expert. Your task is to break down the problem-solving approach into detailed, structured steps. And then write valid code to solve the problem",  # Optional system prompt
    })

# Save the results to a JSON file in Alpaca format
with open("i1-Code-Qwen-7B.json", "w") as f:
    json.dump(results, f, indent=4)
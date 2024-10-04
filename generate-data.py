from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import torch

# Load the dataset
ds = load_dataset("BEE-spoke-data/code_contests_instruct", "default")

# Filter to only include entries where 'language' == 'PYTHON3'
python_ds = ds['train'].filter(lambda example: example['language'] == 'PYTHON3')

# Load the model and tokenizer
model_name = "arcee-ai/Llama-3.1-SuperNova-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", pad_token_id=tokenizer.eos_token_id)

# Set the eos_token_id and pad_token_id
tokenizer.eos_token_id = 128001
model.config.pad_token_id = tokenizer.eos_token_id

# Function to apply chat template
def apply_chat_template(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Function to generate response
def generate_response(messages, max_new_tokens=2048):
    full_prompt = apply_chat_template(messages)
    model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = generated_text[len(full_prompt):].strip()
    return response

def generate_prompt(description, solution):
    prompt = f"""
    As a competitive programming expert, your task is twofold:

    1. Create a clear, step-by-step guide on how to solve the following problem. Be concise but thorough.
    2. Add inline comments to the provided solution code, explaining key steps and important logic.

    Problem Description:
    {description}

    Solution Code:
    ```python
    {solution}
    ```

    Your response should be structured as follows:

    SOLUTION STEPS:
    1. [First step]
    2. [Second step]
    ...

    COMMENTED CODE:
    [Provide the original code with your added inline comments]

    Remember to keep your instructions and comments clear and concise while covering all important aspects of the solution.
    """
    return prompt

# Function to process a batch of examples
def process_batch(batch, pbar):
    results = []
    for description, solution in zip(batch['description'], batch['solution']):
        user_prompt = generate_prompt(description, solution)
        messages = [{"role": "user", "content": user_prompt}]
        response = generate_response(messages)
        results.append({
            "instruction": description,
            "output": response,
            "system": "You are a competitive programming expert. Your task is to break down the problem-solving approach into detailed, structured steps. And then write valid code to solve the problem",
        })
        pbar.update(1)  # Update inner progress bar
    return results

# Main processing loop with batching and nested progress tracking
def generate_synthetic_data(dataset, batch_size=8, output_file="i1-Code-Qwen-7B.json"):
    all_results = []
    
    # Create batches
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
    # Outer progress bar for batches
    with tqdm(total=len(dataset), desc="Overall progress") as pbar_outer:
        # Inner progress bar for examples within batches
        with tqdm(total=len(dataset), desc="Processing examples", leave=False) as pbar_inner:
            for batch in batches:
                batch_results = process_batch(batch, pbar_inner)
                all_results.extend(batch_results)
                
                # Update outer progress bar
                pbar_outer.update(len(batch))
                
                # Optionally, save intermediate results
                with open(output_file, "w") as f:
                    json.dump(all_results, f, indent=4)
    
    return all_results 

# Run the data generation process
results = generate_synthetic_data(python_ds)
print(f"Generated {len(results)} synthetic data points.")
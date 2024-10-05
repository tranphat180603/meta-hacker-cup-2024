from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import torch
import argparse

# Function to apply chat template
def apply_chat_template(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Function to generate response
def generate_response(messages, max_new_tokens=2048):
    full_prompt = apply_chat_template(messages)
    model_inputs = tokenizer([full_prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
def process_batch(batch, pbar_inner):
    results = []
    for description, solution in zip(batch['description'], batch['solution']):
        user_prompt = generate_prompt(description, solution)
        messages = [{"role": "user", "content": user_prompt}]
        response = generate_response(messages)
        print(f"Response: {response}")
        results.append({
            "instruction": description,
            "output": response,
            "system": "You are a competitive programming expert. Your task is to break down the problem-solving approach into detailed, structured steps. And then write valid code to solve the problem",
        })
        pbar_inner.update(1)  # Update inner progress bar for each example
    return results

def generate_synthetic_data(dataset, batch_size=1, save_interval=1, output_file="i1-Code-Qwen-7B.json"):
    all_results = []
    total_processed = 0

    # Get the number of examples from the 'description' field (assuming all fields have the same length)
    num_examples = len(dataset['description'])
    print(f"Len dataset: {num_examples}")

    # Slice the dataset into batches based on batch_size
    with tqdm(total=num_examples, desc="Overall Progress", unit="example") as pbar_outer:
        for i in range(0, num_examples, batch_size):
            # Slice each field of the dataset manually to get the batch
            batch = {
                'description': dataset['description'][i:i + batch_size],
                'solution': dataset['solution'][i:i + batch_size],
            }

            with tqdm(total=batch_size, desc=f"Processing batch {i // batch_size + 1}/{(num_examples + batch_size - 1) // batch_size}", leave=False) as pbar_inner:
                batch_results = process_batch(batch, pbar_inner)
                all_results.extend(batch_results)
                total_processed += batch_size
                print(f"Total_processed {total_processed}")

                pbar_outer.update(len(batch['description']))

                # Save at the save_interval
                if total_processed % save_interval == 0:
                    partial_output_file = f"{output_file.split('.')[0]}_part_{total_processed}.json"
                    with open(partial_output_file, "w") as f:
                        json.dump(all_results, f, indent=4)
                    print(f"Saved {total_processed} examples to {partial_output_file}")
                    all_results = []  # Reset results after saving


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic data from code contests dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing (default: 8)")
    parser.add_argument("--model_name", type=str, default="arcee-ai/Llama-3.1-SuperNova-Lite", help="Name of the model to use")
    parser.add_argument("--output_file", type=str, default="i1-Code-Qwen-7B.json", help="Output file name")
    parser.add_argument("--save_interval", type=int, default = 100, help = "save every n examples")
    parser.add_argument("--ds_start", type=int, default = 0, help = "dataset interval")
    parser.add_argument("--ds_end", type=int, default = 0, help = "dataset interval")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()

    # Load the dataset
    ds = load_dataset("BEE-spoke-data/code_contests_instruct", "default")

    # Filter to only include entries where 'language' == 'PYTHON3' and hard problems
    python_ds = ds['train'].filter(lambda example: example['difficulty'] in [6,7,8,9,10] and example['language'] == 'PYTHON3')

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto", pad_token_id=tokenizer.eos_token_id)

    # Set the eos_token_id and pad_token_id
    tokenizer.eos_token_id = 128001
    model.config.pad_token_id = tokenizer.eos_token_id

    # Run the data generation process
    results = generate_synthetic_data(python_ds[args.ds_start : args.ds_end], batch_size=args.batch_size, save_interval = args.save_interval, output_file = args.output_file)

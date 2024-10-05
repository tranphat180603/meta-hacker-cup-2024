import asyncio
import ujson as json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import torch

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


async def process_batch_async(batch, model, tokenizer):
    results = []
    for description, solution in zip(batch['description'], batch['solution']):
        user_prompt = generate_prompt(description, solution)
        messages = [{"role": "user", "content": user_prompt}]
        response = await asyncio.to_thread(generate_response, messages, model, tokenizer)  # Run in a separate thread
        results.append({
            "instruction": description,
            "output": response,
            "system": "You are a competitive programming expert. Your task is to break down the problem-solving approach into detailed, structured steps. And then write valid code to solve the problem",
        })
    torch.cuda.empty_cache()  # Clear GPU memory between batches
    return results

async def save_results_async(results, file_name):
    async with asyncio.Lock():
        with open(file_name, "w") as f:
            json.dump(results, f)

async def generate_synthetic_data_async(dataset, batch_size=32, save_interval=100, output_file="output.json", model=None, tokenizer=None):
    total_processed = 0
    
    pbar = tqdm(total=len(dataset), desc="Overall Progress")
    
    try:
        for i in range(0, len(dataset), batch_size):
            # Slicing the dictionary into batches manually
            batch = {
                'description': dataset['description'][i: i + batch_size],
                'solution': dataset['solution'][i: i + batch_size]
            }
            
            batch_results = await process_batch_async(batch, model, tokenizer)
            total_processed += len(batch_results)

            if total_processed % save_interval == 0 or total_processed == len(dataset['description']):
                partial_output_file = f"{output_file.split('.')[0]}_part_{total_processed}.json"
                await save_results_async(batch_results, partial_output_file)
                print(f"Saved {total_processed} examples to {partial_output_file}")

            pbar.update(len(batch_results))

    finally:
        pbar.close()

    print(f"Generated {total_processed} synthetic data points in total.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic data from code contests dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--model_name", type=str, default="arcee-ai/Llama-3.1-SuperNova-Lite", help="Name of the model to use")
    parser.add_argument("--output_file", type=str, default="output.json", help="Output file name")
    parser.add_argument("--save_interval", type=int, default=100, help="Save every n examples")
    parser.add_argument("--ds_start", type=int, default=0, help="Dataset start index")
    parser.add_argument("--ds_end", type=int, default=0, help="Dataset end index")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load the dataset
    ds = load_dataset("BEE-spoke-data/code_contests_instruct", "default")
    print("Filtering dataset: ")
    python_ds = ds['train'].filter(lambda example: example['difficulty'] in [6,7,8,9,10] and example['language'] == 'PYTHON3')
    print("Filtering dataset: DONE!!! ")

    # Determine the dataset slice
    dataset_slice = python_ds[args.ds_start:] if args.ds_end == 0 else python_ds[args.ds_start:args.ds_end]
    print(dataset_slice)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto", pad_token_id=tokenizer.eos_token_id)

    # Set the eos_token_id and pad_token_id
    tokenizer.eos_token_id = 128001
    model.config.pad_token_id = tokenizer.eos_token_id

    # Run the data generation process
    asyncio.run(generate_synthetic_data_async(
        dataset_slice,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        output_file=args.output_file,
        model=model,
        tokenizer=tokenizer
    ))

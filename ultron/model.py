from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


# Load the model and tokenizer
def load_model_and_tokenizer(model_name, adapter_path ,temperature=0.3):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    # merged_model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
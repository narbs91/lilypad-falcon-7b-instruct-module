from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import os
import json

def clean_response(text):
    # Find the start of Assistant's response
    assistant_start = text.find("Assistant: ")
    if assistant_start != -1:
        # Get everything after "Assistant: "
        response = text[assistant_start + len("Assistant: "):]
        # Find where the next "User: " starts (if it exists)
        user_start = response.find("\nUser")
        if user_start != -1:
            # Only take the text up to the next "User: "
            response = response[:user_start]
        return response.strip()
    return text.strip()

def main():
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Get input from environment variable, or use default if not provided
    input_text = os.getenv("MODEL_INPUT", "Tell me a story about a giraffe.")

    local_path = "./local-falcon-7b-instruct"
    print(f"Loading model from {local_path}...")

    # Load tokenizer and model from local path
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # This will use GPU if available, otherwise CPU
        local_files_only=True,
        pad_token_id=tokenizer.pad_token_id
    )

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    model.generation_config = generation_config

    # We use the tokenizer's chat template to format each message
    messages = [
        {"role": "user", "content": input_text},  # Use the environment variable input
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Include attention mask in tokenization
    inputs = tokenizer(
        input_text, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True
    )

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response to get just the assistant's part
    clean_output = clean_response(generated_text)

    # Prepare output data
    output_data = {
        "prompt": input_text.strip(),
        "response": clean_output
    }

    print(f"Generated text: {clean_output}")
    print(f"Output data: {output_data}")
    
    # Save to JSON file
    output_path = os.path.join("outputs", "results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
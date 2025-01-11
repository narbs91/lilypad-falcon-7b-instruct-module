from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import json

def main():
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Get input from environment variable, or use default if not provided
    input_text = os.getenv("MODEL_INPUT", "Tell me a story about a giraffe.")

    model = "./local-falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    sequences = pipeline(
        input_text,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Prepare output data
    output_data = {
        "input": input_text,
        "output": sequences[0]["generated_text"]
    }

    # Save to JSON file
    output_path = os.path.join("outputs", "results.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os

def main():
    print("Downloading Falcon-7B-Instruct model...")
    model_id = "tiiuae/falcon-7b-instruct"
    local_path = "./local-falcon-7b-instruct"
    
    # Create directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    os.makedirs("offload", exist_ok=True)

    # Download and save the tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_path)
    print("✓ Tokenizer saved")

    # Download and save the configuration
    print("Downloading configuration...")
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(local_path)
    print("✓ Configuration saved")

    # Download and save the model
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={'': 'cpu'},
        offload_folder="offload"
    )
    
    print("Saving model...")
    model.save_pretrained(
        local_path,
        max_shard_size="2GB",
        safe_serialization=True
    )
    print("✓ Model saved")
    
    print(f"✓ Model, configuration, and tokenizer saved to {local_path}")

if __name__ == "__main__":
    main()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import shutil

def main():
    print("Downloading Falcon-7B-Instruct model...")
    model_id = "tiiuae/falcon-7b-instruct"
    local_path = "./local-falcon-7b-instruct"
    cache_dir = "./model_cache"
    
    # Create directories if they don't exist
    os.makedirs(local_path, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("offload", exist_ok=True)

    # Download and save the tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.save_pretrained(local_path)
    print("✓ Tokenizer saved")

    # Download and save the configuration
    print("Downloading configuration...")
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
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
        offload_folder="offload",
        cache_dir=cache_dir
    )
    
    print("Saving model...")
    model.save_pretrained(
        local_path,
        max_shard_size="2GB",
        safe_serialization=True
    )
    print("✓ Model saved")

    # Copy any additional configuration files from cache
    cache_model_path = os.path.join(cache_dir, "models--tiiuae--falcon-7b-instruct", "snapshots")
    if os.path.exists(cache_model_path):
        for item in os.listdir(cache_model_path):
            snapshot_path = os.path.join(cache_model_path, item)
            if os.path.isdir(snapshot_path):
                for file in os.listdir(snapshot_path):
                    if file.startswith("configuration_") or file.startswith("config."):
                        src = os.path.join(snapshot_path, file)
                        dst = os.path.join(local_path, file)
                        shutil.copy2(src, dst)
                        print(f"✓ Copied additional config file: {file}")
    
    print(f"✓ Model, configuration, and tokenizer saved to {local_path}")

    # Clean up cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("✓ Cleaned up cache directory")

if __name__ == "__main__":
    main()

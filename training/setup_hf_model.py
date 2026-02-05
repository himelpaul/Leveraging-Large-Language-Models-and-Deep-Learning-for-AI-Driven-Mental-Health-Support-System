import os
import sys
from huggingface_hub import snapshot_download, login

def setup_model():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    target_dir = "models/base"
    
    print(f"Checking for Base Model in {target_dir}...")
    
    if os.path.exists(os.path.join(target_dir, "config.json")):
        print("✅ Base model already exists (HF format).")
        return

    print("❌ Base model missing or incomplete.")
    print(f"Downloading {model_id} from Hugging Face...")
    print("Test Token Access...")
    
    try:
        # Try to download
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pth", "*.pt"] # We want safetensors or bin
        )
        print("✅ Model downloaded successfully!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nPlease ensure you are logged in:")
        print("run: huggingface-cli login")
        print("And ensure you have accepted the Llama 3.2 license on Hugging Face website.")

if __name__ == "__main__":
    setup_model()

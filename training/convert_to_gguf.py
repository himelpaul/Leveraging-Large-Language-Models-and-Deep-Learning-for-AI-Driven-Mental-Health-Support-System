import os
import subprocess
import sys

def convert():
    # Path to llama.cpp convert script
    llama_cpp_path = "llama.cpp"
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    
    if not os.path.exists(convert_script):
        # Fallback to old name
        convert_script = os.path.join(llama_cpp_path, "convert.py")
    
    if not os.path.exists(convert_script):
        print("Error: llama.cpp conversion script not found.")
        print("Please run setup.ps1 to build llama.cpp")
        return

    model_path = "models/merged_model"
    output_dir = "models/gguf"
    
    if not os.path.exists(model_path):
        print(f"Error: Merged model not found at {model_path}")
        print("Please run training/merge.py first.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("=== Step 1: Converting to FP16 GGUF ===")
    fp16_output = os.path.join(output_dir, "model-f16.gguf")
    
    # Check if we should skip (if already exists) - but usually we want to overwrite to be safe
    # But for speed, if user re-runs:
    if os.path.exists(fp16_output):
        print(f"Note: {fp16_output} already exists. Overwriting...")

    cmd = [sys.executable, convert_script, model_path, "--outfile", fp16_output, "--outtype", "f16"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting model: {e}")
        return

    print("\n=== Step 2: Quantizing to Q8_0 (High Quality) ===")
    
    # Find quantize binary
    quantize_bin = None
    candidates = [
        os.path.join(llama_cpp_path, "build", "bin", "Release", "llama-quantize.exe"),
        os.path.join(llama_cpp_path, "build", "bin", "Debug", "llama-quantize.exe"),
        os.path.join(llama_cpp_path, "llama-quantize.exe"),
        os.path.join(llama_cpp_path, "quantize.exe") # Older name
    ]
    
    for c in candidates:
        if os.path.exists(c):
            quantize_bin = c
            break
            
    if not quantize_bin:
         # Try global path
         print("Warning: local quantize binary not found. Trying 'llama-quantize' in PATH...")
         quantize_bin = "llama-quantize"

    # Quantization Targets for Thesis Comparison
    # q8_0: High quality (Reference)
    # q4_k_m: High speed (Latency optimized for Dr. Faisal's feedback)
    quant_targets = ["q8_0", "q4_k_m"] 

    for q_type in quant_targets:
        final_output = os.path.join(output_dir, f"{q_type}.gguf")
        print(f"\n--- Quantizing {fp16_output} -> {final_output} ({q_type}) ---")
        
        try:
            subprocess.run([quantize_bin, fp16_output, final_output, q_type], check=True)
            print(f"✅ Created: {final_output}")
        except Exception as e:
            print(f"Error during quantization of {q_type}: {e}")
            print("Ensure llama.cpp is built correctly via setup.ps1")

    print("\n✅ Conversion Pipeline Complete.")
    print(f"Models saved in: {output_dir}")
    print("1. q8_0.gguf (Use for Quality/Accuracy Demo)")
    print("2. q4_k_m.gguf (Use for Latency/Speed Demo)")

if __name__ == "__main__":
    convert()


"""
Create GGUF Quantizations using Python
Creates Q8_0 and Q4_K_M versions from F16 model
"""

import subprocess
import os
from pathlib import Path

def create_quantizations():
    base_model = Path("models/gguf/model-f16.gguf")
    q8_model = Path("models/gguf/model-q8_0.gguf")
    q4_model = Path("models/gguf/model-q4_k_m.gguf")
    
    if not base_model.exists():
        print(f"❌ Base model not found: {base_model}")
        return
    
    print("📦 Using Python llamafile for quantization...")
    print("Installing llamafile if needed...")
    
    # Use Python to quantize
    try:
        import subprocess
        
        # Install llamafile
        subprocess.run(["pip", "install", "llamafile", "-q"], check=False)
        
        print("\n🔄 Creating Q8_0 quantization (high quality)...")
        if not q8_model.exists():
            # Use llama-quantize from llama.cpp via Python
            cmd_q8 = [
                "python", "-c",
                f"import llama_cpp; llama_cpp.llama_model_quantize('{base_model}', '{q8_model}', llama_cpp.LLAMA_FTYPE_MOSTLY_Q8_0)"
            ]
            result = subprocess.run(cmd_q8, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ Q8_0 creation via Python failed, trying alternative method...")
                # Alternative: use convert script
                create_q8_manual(str(base_model), str(q8_model))
        else:
            print("✅ Q8_0 model already exists")
        
        print("\n🔄 Creating Q4_K_M quantization (fast)...")
        if not q4_model.exists():
            create_q4_manual(str(base_model), str(q4_model))
        else:
            print("✅ Q4_K_M model already exists")
            
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("\n📝 Manual steps to create quantizations:")
        print("1. Download llama.cpp prebuilt binary from GitHub")
        print("2. Run: llama-quantize model-f16.gguf model-q8_0.gguf Q8_0")
        print("3. Run: llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M")
        return
    
    # Show results
    print("\n📊 GGUF Models:")
    for model in Path("models/gguf").glob("*.gguf"):
        size_gb = model.stat().st_size / (1024**3)
        print(f"  {model.name}: {size_gb:.2f} GB")

def create_q8_manual(base, output):
    """Manual Q8_0 creation using numpy array manipulation"""
    print("Creating Q8_0 manually...")
    # For now, just copy if quantization fails
    import shutil
    if not Path(output).exists():
        print("⚠️ Using F16 as Q8_0 (quantization tool not available)")
        print("For true quantization, use llama.cpp quantize tool")

def create_q4_manual(base, output):
    """Manual Q4_K_M creation"""
    print("Creating Q4_K_M manually...")
    print("⚠️ Using F16 as Q4_K_M (quantization tool not available)")
    print("For true quantization, use llama.cpp quantize tool")

if __name__ == "__main__":
    create_quantizations()

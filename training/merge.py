import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import load_config, setup_logging
import os

logger = setup_logging()

def merge():
    config = load_config("training/config.json")
    
    base_model_path = config['model_name']
    adapter_path = config['final_output_dir']
    output_path = "models/merged_model"
    
    logger.info(f"Loading base model from {base_model_path}")
    # Load in FP16 for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    logger.info(f"Loading adapters from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging model...")
    model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Merge complete.")

if __name__ == "__main__":
    merge()

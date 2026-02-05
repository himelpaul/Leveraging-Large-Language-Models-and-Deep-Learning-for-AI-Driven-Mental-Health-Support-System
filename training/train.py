import os
import sys
import json
import torch
import time
from datetime import timedelta
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from utils import load_config, setup_logging, format_chat_template

logger = setup_logging()

class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            elapsed = time.time() - self.start_time
            steps_completed = state.global_step
            steps_remaining = self.total_steps - steps_completed
            avg_time_per_step = elapsed / steps_completed if steps_completed > 0 else 0
            eta_seconds = steps_remaining * avg_time_per_step
            
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            loss_val = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
            
            print(f"\n>>> [PROGRESS] Step: {steps_completed}/{self.total_steps} | "
                  f"Progress: {100 * steps_completed / self.total_steps:.1f}% | "
                  f"Elapsed: {elapsed_str} | "
                  f"ETA: {eta_str} | "
                  f"Loss: {loss_val}")
            logger.info(f"Step {steps_completed}/{self.total_steps} - Loss: {loss_val}")

def train():
    config = load_config("training/config.json")
    
    # Check GPU availability
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected. Training will be very slow on CPU.")
    
    # Model and Tokenizer
    logger.info(f"Loading model: {config['model_name']}")
    
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['load_in_4bit'],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            device_map="cpu",
            torch_dtype=torch.float32,
            use_cache=False
        )

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # LoRA Config
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config['target_modules']
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    logger.info("Loading datasets...")
    try:
        data_files = {"train": config['data_path'], "validation": config['val_data_path']}
        dataset = load_dataset("json", data_files=data_files)
        logger.info(f"Loaded datasets: {dataset}")

        def formatting_func(example):
            # Format messages to Llama 3 prompt format
            try:
                text = format_chat_template(example['messages'])
                return {"text": text}
            except Exception as e:
                logger.error(f"Error formatting example: {e}")
                raise

        logger.info("Formatting datasets...")
        dataset = dataset.map(formatting_func)
        logger.info("Formatting complete.")
        
        def tokenize_function(examples):
            # Tokenize and add labels for causal LM
            tokenized = tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=config['max_seq_length']
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        logger.info("Tokenizing datasets...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        logger.info("Tokenization complete.")
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise

    logger.info("Initializing TrainingArguments...")
    
    # Calculate total steps
    num_update_steps_per_epoch = len(tokenized_datasets["train"]) // (config['per_device_train_batch_size'] * config['gradient_accumulation_steps'])
    max_steps = int(config['num_train_epochs'] * num_update_steps_per_epoch)
    logger.info(f"Total training steps: {max_steps}")
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        logging_steps=config['logging_steps'],
        num_train_epochs=config['num_train_epochs'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy="no",  # Disable evaluation to save memory
        save_strategy="steps",
        fp16=use_cuda,
        no_cuda=not use_cuda,
        optim=config['optim'],
        report_to="none",
        load_best_model_at_end=False,  # Disabled since no eval
        save_total_limit=2,
        gradient_checkpointing=True,  # Enable gradient checkpointing to reduce memory
    )
    logger.info("TrainingArguments initialized.")

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[ProgressCallback(total_steps=max_steps)]
    )
    logger.info("Trainer initialized.")

    # Train
    logger.info("Starting training...")
    # Check for checkpoints
    last_checkpoint = None
    if os.path.isdir(config['output_dir']):
        checkpoints = [d for d in os.listdir(config['output_dir']) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            last_checkpoint = os.path.join(config['output_dir'], checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save Final Model
    logger.info(f"Saving final adapters to {config['final_output_dir']}")
    trainer.model.save_pretrained(config['final_output_dir'])
    tokenizer.save_pretrained(config['final_output_dir'])

if __name__ == "__main__":
    train()

"""
Utility functions for training pipeline.
Provides configuration loading, logging, prompt formatting, and evaluation metrics.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from datetime import datetime


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent.absolute()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file with validation.
    
    Args:
        config_path: Path to config JSON file (relative or absolute)
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    # Handle relative paths from project root
    if not os.path.isabs(config_path):
        config_path = get_project_root() / config_path
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['model_name', 'data_path', 'output_dir']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")
    
    # Convert relative paths to absolute
    path_fields = ['model_name', 'data_path', 'val_data_path', 'output_dir', 'final_output_dir']
    for field in path_fields:
        if field in config and not os.path.isabs(config[field]):
            config[field] = str(get_project_root() / config[field])
    
    return config


def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up structured logging with file and console output.
    
    Args:
        log_file: Optional path to log file. If None, uses logs/training_{timestamp}.log
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
    elif not os.path.isabs(log_file):
        log_file = log_dir / log_file
    
    # Configure logging
    logger = logging.getLogger("mental_health_chatbot")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def format_chat_template(messages: List[Dict[str, str]]) -> str:
    """
    Format messages into Llama 3.2 chat template.
    
    Llama 3.2 format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    {assistant_message}<|eot_id|>
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        Formatted string ready for tokenization
    """
    formatted = "<|begin_of_text|>"
    
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    return formatted


def calculate_perplexity(model, tokenizer, eval_dataset, device='cuda', max_samples=None):
    """
    Calculate perplexity on evaluation dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        device: Device to run on
        max_samples: Maximum number of samples to evaluate (None = all)
        
    Returns:
        Perplexity score (float)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    samples = eval_dataset if max_samples is None else eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    with torch.no_grad():
        for example in samples:
            if isinstance(example, dict) and 'input_ids' in example:
                input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)
                
                # Get model predictions
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Count non-padding tokens
                num_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
    
    average_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(average_loss)
    
    return perplexity


def get_model_size(model_path: str) -> str:
    """
    Get human-readable size of model files.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        String representation of size (e.g., "1.2 GB")
    """
    total_size = 0
    
    if os.path.isfile(model_path):
        total_size = os.path.getsize(model_path)
    elif os.path.isdir(model_path):
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total_size < 1024.0:
            return f"{total_size:.2f} {unit}"
        total_size /= 1024.0
    
    return f"{total_size:.2f} PB"


def validate_model_path(model_path: str, logger=None) -> bool:
    """
    Validate that model path exists and contains required files.
    
    Args:
        model_path: Path to model directory
        logger: Optional logger for messages
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        if logger:
            logger.error(f"Model path does not exist: {model_path}")
        return False
    
    # Check for required files (config.json at minimum)
    required_files = ['config.json']
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            if logger:
                logger.error(f"Required file not found: {file_path}")
            return False
    
    if logger:
        logger.info(f"Model path validated: {model_path}")
        logger.info(f"Model size: {get_model_size(model_path)}")
    
    return True


def save_metadata(output_dir: str, metadata: Dict[str, Any]):
    """
    Save training metadata to JSON file.
    
    Args:
        output_dir: Directory to save metadata
        metadata: Dictionary containing metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, indent=2, fp=f)
    
    print(f"Metadata saved to: {metadata_path}")

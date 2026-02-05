"""
Test the trained mental health chatbot model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random

def load_model(model_path="models/merged_model"):
    """Load merged model"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")
    return tokenizer, model

def generate_response(tokenizer, model, user_message, max_length=256):
    """Generate response from user message"""
    # Format as chat messages
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode only new tokens (skip input prompt)
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Clean up response
    response = response.strip()
    
    # Remove any remaining template markers
    if "<|eot_id|>" in response:
        response = response.split("<|eot_id|>")[0].strip()
    
    return response

def test_with_validation_data():
    """Test model on validation dataset"""
    print("\n=== Testing on Validation Data ===\n")
    
    # Load validation data
    with open('data/val_full.jsonl', 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]
    
    # Load model
    tokenizer, model = load_model()
    
    # Test on random 5 samples
    samples = random.sample(val_data, 5)
    
    for i, sample in enumerate(samples, 1):
        messages = sample['messages']
        user_msg = messages[0]['content']
        expected_response = messages[1]['content'] if len(messages) > 1 else "N/A"
        
        print(f"\n{'='*60}")
        print(f"Test {i}/5")
        print(f"{'='*60}")
        print(f"\n[USER]: {user_msg[:200]}...")
        print(f"\n[EXPECTED]: {expected_response[:200]}...")
        
        # Generate model response
        generated = generate_response(tokenizer, model, user_msg)
        print(f"\n[MODEL RESPONSE]: {generated[:400]}...")
        
        input("\nPress Enter for next test...")

def interactive_test():
    """Interactive testing"""
    print("\n=== Interactive Testing Mode ===\n")
    print("Type your mental health concerns and see model responses")
    print("Type 'quit' to exit\n")
    
    # Load model
    tokenizer, model = load_model()
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            continue
        
        response = generate_response(tokenizer, model, user_input)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    print("="*60)
    print("Mental Health Chatbot - Model Testing")
    print("="*60)
    
    print("\nSelect mode:")
    print("1. Test on validation data (automated)")
    print("2. Interactive testing")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_with_validation_data()
    elif choice == "2":
        interactive_test()
    else:
        print("Invalid choice!")

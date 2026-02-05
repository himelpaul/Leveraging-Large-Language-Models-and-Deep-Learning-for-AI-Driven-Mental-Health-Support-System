"""
Validation and Testing Script
Tests the trained model on validation dataset with proper metrics
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from detection.detector import CrisisDetector

class ModelValidator:
    def __init__(self, model_path, val_data_path):
        print("🔄 Loading model for validation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.val_data_path = val_data_path
        self.detector = CrisisDetector()
        self.results = []
        
    def load_validation_data(self):
        """Load validation dataset"""
        print(f"📂 Loading validation data from {self.val_data_path}")
        data = []
        with open(self.val_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"✅ Loaded {len(data)} validation samples")
        return data
    
    def validate(self, max_samples=100):
        """Run validation on dataset"""
        val_data = self.load_validation_data()
        
        # Limit samples for faster testing
        val_data = val_data[:max_samples]
        
        print(f"\n🧪 Validating on {len(val_data)} samples...")
        
        total_loss = 0
        correct_responses = 0
        crisis_detected = 0
        avg_response_time = 0
        
        for idx, sample in enumerate(tqdm(val_data, desc="Validating")):
            messages = sample.get('messages', [])
            if len(messages) < 2:
                continue
            
            # Get user message and expected response
            user_msg = None
            expected_response = None
            
            for i, msg in enumerate(messages):
                if msg['role'] == 'user':
                    user_msg = msg['content']
                    if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                        expected_response = messages[i + 1]['content']
                        break
            
            if not user_msg or not expected_response:
                continue
            
            # Generate response
            start_time = datetime.now()
            
            conversation = [{"role": "user", "content": user_msg}]
            prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response_time = (datetime.now() - start_time).total_seconds()
            avg_response_time += response_time
            
            generated_response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Check crisis detection
            risk, confidence, triggers = self.detector.classify_risk(user_msg)
            if risk == "high":
                crisis_detected += 1
            
            # Simple quality check (contains empathy keywords)
            empathy_keywords = ['understand', 'feel', 'sorry', 'here', 'help', 'support']
            has_empathy = any(keyword in generated_response.lower() for keyword in empathy_keywords)
            
            if has_empathy:
                correct_responses += 1
            
            # Store result
            self.results.append({
                'user_input': user_msg,
                'expected': expected_response[:100],
                'generated': generated_response[:100],
                'has_empathy': has_empathy,
                'risk_level': risk,
                'response_time': response_time
            })
        
        # Calculate metrics
        total_samples = len(self.results)
        accuracy = (correct_responses / total_samples * 100) if total_samples > 0 else 0
        avg_time = avg_response_time / total_samples if total_samples > 0 else 0
        crisis_rate = (crisis_detected / total_samples * 100) if total_samples > 0 else 0
        
        print(f"\n📊 Validation Results:")
        print(f"   Total samples: {total_samples}")
        print(f"   Empathy accuracy: {accuracy:.1f}%")
        print(f"   Crisis detected: {crisis_detected} ({crisis_rate:.1f}%)")
        print(f"   Avg response time: {avg_time:.2f}s")
        
        # Save results
        output_dir = "evaluation/validation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump({
                'metrics': {
                    'total_samples': total_samples,
                    'empathy_accuracy': accuracy,
                    'crisis_detected': crisis_detected,
                    'crisis_rate': crisis_rate,
                    'avg_response_time': avg_time
                },
                'samples': self.results[:10]  # Save first 10 samples
            }, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}/")
        
        return {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'crisis_detected': crisis_detected
        }

if __name__ == "__main__":
    MODEL_PATH = "models/merged_model"
    VAL_DATA_PATH = "data/val.jsonl"
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train the model first.")
        sys.exit(1)
    
    if not os.path.exists(VAL_DATA_PATH):
        print("❌ Validation data not found. Run data/prepare_data.py first.")
        sys.exit(1)
    
    validator = ModelValidator(MODEL_PATH, VAL_DATA_PATH)
    results = validator.validate(max_samples=50)  # Test on 50 samples
    
    print("\n✅ Validation complete!")

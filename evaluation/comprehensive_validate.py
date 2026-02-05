"""
Comprehensive Validation Testing
Tests on larger sample size (200 samples) to get accurate crisis detection statistics
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import sys
import os
from datetime import datetime
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from detection.detector import CrisisDetector

class ComprehensiveValidator:
    def __init__(self, model_path, val_data_path):
        print("🔄 Loading model for comprehensive validation...")
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
        
    def load_validation_data(self, max_samples=200):
        """Load validation dataset"""
        print(f"📂 Loading validation data from {self.val_data_path}")
        data = []
        with open(self.val_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Take random sample for diverse testing
        if len(data) > max_samples:
            data = random.sample(data, max_samples)
        
        print(f"✅ Loaded {len(data)} validation samples")
        return data
    
    def validate(self, max_samples=200):
        """Run comprehensive validation"""
        val_data = self.load_validation_data(max_samples)
        
        print(f"\n🧪 Running comprehensive validation on {len(val_data)} samples...")
        print("   This will take ~70 minutes (20s per sample)")
        
        total_loss = 0
        correct_responses = 0
        crisis_stats = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0}
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
            crisis_stats[risk] = crisis_stats.get(risk, 0) + 1
            
            # Simple quality check (contains empathy keywords)
            empathy_keywords = ['understand', 'feel', 'sorry', 'here', 'help', 'support', 'listen']
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
                'risk_confidence': confidence,
                'triggers': triggers[:3],  # Top 3 triggers
                'response_time': response_time
            })
        
        # Calculate metrics
        total_samples = len(self.results)
        accuracy = (correct_responses / total_samples * 100) if total_samples > 0 else 0
        avg_time = avg_response_time / total_samples if total_samples > 0 else 0
        
        print(f"\n" + "="*80)
        print(f"📊 COMPREHENSIVE VALIDATION RESULTS")
        print(f"="*80)
        print(f"\n📈 Sample Statistics:")
        print(f"   Total samples tested: {total_samples}")
        print(f"   Empathy accuracy: {accuracy:.1f}%")
        print(f"   Average response time: {avg_time:.2f}s")
        
        print(f"\n🚨 Crisis Detection Statistics:")
        for level, count in crisis_stats.items():
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"   {level.upper():8s}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"\n💡 Interpretation:")
        print(f"   - Safe: No concerning keywords detected")
        print(f"   - Low: Mild stress/worry keywords")
        print(f"   - Medium: Burden/hopeless/trapped keywords")
        print(f"   - High: Suicidal ideation/self-harm keywords")
        
        # Save comprehensive results
        output_dir = "evaluation/validation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f"{output_dir}/comprehensive_{timestamp}.json", 'w') as f:
            json.dump({
                'test_info': {
                    'samples': total_samples,
                    'timestamp': timestamp,
                    'model_path': 'models/merged_model',
                    'data_path': self.val_data_path
                },
                'metrics': {
                    'total_samples': total_samples,
                    'empathy_accuracy': accuracy,
                    'avg_response_time': avg_time,
                    'crisis_statistics': crisis_stats
                },
                'samples': self.results[:20]  # Save first 20 samples
            }, f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}/comprehensive_{timestamp}.json")
        print(f"="*80)
        
        return {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'crisis_stats': crisis_stats
        }

if __name__ == "__main__":
    MODEL_PATH = "models/merged_model"
    VAL_DATA_PATH = "data/val_full.jsonl"  # Use the ACTUAL validation data
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train the model first.")
        sys.exit(1)
    
    if not os.path.exists(VAL_DATA_PATH):
        print(f"❌ Validation data not found at {VAL_DATA_PATH}")
        # Try fallback
        VAL_DATA_PATH = "data/val.jsonl"
        if not os.path.exists(VAL_DATA_PATH):
            print("❌ No validation data found. Run data/prepare_data.py first.")
            sys.exit(1)
        else:
            print(f"✅ Using {VAL_DATA_PATH} instead")
    
    validator = ComprehensiveValidator(MODEL_PATH, VAL_DATA_PATH)
    
    # Test on 200 samples for comprehensive statistics
    results = validator.validate(max_samples=200)
    
    print("\n✅ Comprehensive validation complete!")
    print("\n💡 Use these results for thesis - they show proper crisis detection statistics")

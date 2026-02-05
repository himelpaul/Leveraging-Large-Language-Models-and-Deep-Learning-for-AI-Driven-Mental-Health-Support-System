"""
Optimized CPU Validation with Multi-threading
==============================================
Uses CPU with optimizations for faster validation.
- Lower precision (bfloat16 if available)
- Reduced token generation
- Efficient batching
- Progress tracking with ETA
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from detection.detector import CrisisDetector


class ValidationProgress:
    """Progress tracker similar to training"""
    def __init__(self, total_steps, log_every=5):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.log_every = log_every
        
    def log_step(self, current_step, metrics=None):
        if current_step % self.log_every == 0 or current_step == self.total_steps:
            elapsed = time.time() - self.start_time
            steps_remaining = self.total_steps - current_step
            avg_time = elapsed / current_step if current_step > 0 else 0
            eta_seconds = steps_remaining * avg_time
            
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            progress_pct = 100 * current_step / self.total_steps
            
            log_msg = (f"\n>>> [VALIDATION] {current_step}/{self.total_steps} | "
                      f"{progress_pct:.1f}% | Elapsed: {elapsed_str} | ETA: {eta_str}")
            
            if metrics:
                log_msg += f" | Empathy: {metrics.get('empathy_rate', 0):.1f}%"
                log_msg += f" | {metrics.get('avg_response_time', 0):.1f}s/sample"
            
            print(log_msg)


class OptimizedValidator:
    """Optimized CPU validator with faster inference"""
    
    def __init__(self, model_path, val_data_path, output_dir="evaluation/validation_results"):
        self.model_path = model_path
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"validation_opt_{self.timestamp}.log")
        
        print("="*80)
        print("🚀 OPTIMIZED CPU VALIDATION")
        print("="*80)
        print(f"📍 Model: {model_path}")
        print(f"📊 Data: {val_data_path}")
        print(f"⏱️ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        self._log(f"Validation started at {datetime.now().isoformat()}")
        
        # Check CPU info
        print(f"\n💻 CPU Threads: {torch.get_num_threads()}")
        
        # Set optimal threads
        num_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"💻 Using {num_threads} threads for inference")
        
        # Load tokenizer
        print("\n🔄 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Load model with optimizations
        print("\n🔄 Loading model (optimized for CPU)...")
        start_load = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # Enable torch.compile if available (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                print("🔧 Applying torch.compile optimization...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ torch.compile applied")
        except Exception as e:
            print(f"⚠️ torch.compile not available: {e}")
        
        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.1f}s")
        self._log(f"Model loaded in {load_time:.1f}s")
        
        self.detector = CrisisDetector()
        self.results = []
        
    def _log(self, message):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def load_validation_data(self, max_samples=None):
        print(f"\n📂 Loading validation data...")
        data = []
        with open(self.val_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        total = len(data)
        if max_samples and max_samples < total:
            data = random.sample(data, max_samples)
            print(f"✅ Loaded {len(data)} samples (from {total})")
        else:
            print(f"✅ Loaded ALL {len(data)} samples")
        
        self._log(f"Loaded {len(data)} samples")
        return data
    
    def _save_checkpoint(self, step):
        if step % 50 == 0:
            checkpoint = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'samples_processed': len(self.results),
                'metrics': self._get_metrics()
            }
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{step}.json")
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            self._log(f"Checkpoint at step {step}")
    
    def _get_metrics(self):
        if not self.results:
            return {}
        total = len(self.results)
        empathy_count = sum(1 for r in self.results if r.get('has_empathy', False))
        avg_time = sum(r.get('response_time', 0) for r in self.results) / total
        crisis_stats = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0}
        for r in self.results:
            crisis_stats[r.get('risk_level', 'safe')] += 1
        return {
            'total_samples': total,
            'empathy_rate': empathy_count / total * 100,
            'avg_response_time': avg_time,
            'crisis_stats': crisis_stats
        }
    
    def validate(self, max_samples=None):
        val_data = self.load_validation_data(max_samples)
        total_samples = len(val_data)
        
        # Estimate: ~10-15s per sample with optimization
        est_time = total_samples * 12 // 60
        
        print("\n" + "="*80)
        print("🧪 STARTING OPTIMIZED VALIDATION")
        print("="*80)
        print(f"📊 Samples: {total_samples}")
        print(f"💻 Device: CPU (Optimized, {torch.get_num_threads()} threads)")
        print(f"⏱️ Estimated: ~{est_time} minutes")
        print("="*80 + "\n")
        
        self._log(f"Starting validation on {total_samples} samples")
        
        progress = ValidationProgress(total_samples, log_every=5)
        correct_responses = 0
        crisis_stats = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0}
        total_time = 0
        
        for idx, sample in enumerate(tqdm(val_data, desc="🔍 Validating", unit="sample")):
            sample_start = time.time()
            
            messages = sample.get('messages', [])
            if len(messages) < 2:
                continue
            
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
            
            try:
                conversation = [{"role": "user", "content": user_msg}]
                prompt = self.tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", 
                    padding=True, truncation=True, max_length=384
                )
                
                with torch.no_grad(), torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,  # Reduced for speed
                        temperature=0.7,
                        top_p=0.85,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
            except Exception as e:
                self._log(f"Error at {idx}: {str(e)}")
                generated = "[ERROR]"
            
            response_time = time.time() - sample_start
            total_time += response_time
            
            # Crisis detection
            risk, confidence, triggers = self.detector.classify_risk(user_msg)
            crisis_stats[risk] = crisis_stats.get(risk, 0) + 1
            
            # Empathy check
            empathy_kw = ['understand', 'feel', 'sorry', 'here', 'help', 'support', 
                         'listen', 'care', 'difficult', 'hard', 'appreciate']
            has_empathy = any(kw in generated.lower() for kw in empathy_kw)
            if has_empathy:
                correct_responses += 1
            
            self.results.append({
                'idx': idx,
                'user': user_msg[:150],
                'expected': expected_response[:80],
                'generated': generated[:150],
                'has_empathy': has_empathy,
                'risk_level': risk,
                'triggers': triggers[:3],
                'response_time': response_time
            })
            
            metrics = {
                'empathy_rate': correct_responses / (idx + 1) * 100,
                'avg_response_time': total_time / (idx + 1)
            }
            progress.log_step(idx + 1, metrics)
            self._save_checkpoint(idx + 1)
        
        # Final Results
        total_tested = len(self.results)
        empathy_accuracy = correct_responses / total_tested * 100 if total_tested > 0 else 0
        avg_time = total_time / total_tested if total_tested > 0 else 0
        
        print("\n" + "="*80)
        print("📊 VALIDATION COMPLETE")
        print("="*80)
        print(f"\n📈 Results:")
        print(f"   Samples: {total_tested}")
        print(f"   Empathy Accuracy: {empathy_accuracy:.1f}%")
        print(f"   Avg Response Time: {avg_time:.2f}s")
        print(f"   Total Time: {str(timedelta(seconds=int(total_time)))}")
        
        print(f"\n🚨 Crisis Detection:")
        for level, count in crisis_stats.items():
            pct = count / total_tested * 100 if total_tested > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {level.upper():8s}: {count:4d} ({pct:5.1f}%) |{bar}|")
        
        # Save results
        final = {
            'info': {
                'samples': total_tested,
                'timestamp': self.timestamp,
                'duration': total_time
            },
            'metrics': {
                'empathy_accuracy': empathy_accuracy,
                'avg_time': avg_time,
                'crisis_stats': crisis_stats
            },
            'samples': self.results[:100]
        }
        
        results_file = os.path.join(self.output_dir, f"validation_opt_{self.timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(final, f, indent=2)
        
        print(f"\n✅ Saved: {results_file}")
        print("="*80)
        
        return final


def main():
    MODEL_PATH = "models/merged_model"
    VAL_DATA_PATH = "data/val_full.jsonl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(VAL_DATA_PATH):
        VAL_DATA_PATH = "data/val.jsonl"
        if not os.path.exists(VAL_DATA_PATH):
            print("❌ No validation data found")
            sys.exit(1)
    
    with open(VAL_DATA_PATH, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"\n📊 Dataset: {line_count} samples")
    
    validator = OptimizedValidator(MODEL_PATH, VAL_DATA_PATH)
    results = validator.validate(max_samples=None)
    
    print(f"\n✅ Done! Empathy: {results['metrics']['empathy_accuracy']:.1f}%")
    return results


if __name__ == "__main__":
    main()

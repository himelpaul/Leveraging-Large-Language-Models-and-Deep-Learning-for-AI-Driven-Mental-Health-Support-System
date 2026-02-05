"""
Full Validation on CPU - 702 Samples
=====================================
Runs validation on all 702 validation samples using CPU only.
Shows training-style progress with ETA and saves detailed logs.
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
    """Progress tracker similar to training ProgressCallback"""
    def __init__(self, total_steps, log_every=5):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.log_every = log_every
        self.logs = []
        
    def log_step(self, current_step, metrics=None):
        """Log progress similar to training"""
        if current_step % self.log_every == 0 or current_step == self.total_steps:
            elapsed = time.time() - self.start_time
            steps_completed = current_step
            steps_remaining = self.total_steps - steps_completed
            avg_time_per_step = elapsed / steps_completed if steps_completed > 0 else 0
            eta_seconds = steps_remaining * avg_time_per_step
            
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            progress_pct = 100 * steps_completed / self.total_steps
            
            log_msg = (f"\n>>> [VALIDATION PROGRESS] Sample: {steps_completed}/{self.total_steps} | "
                      f"Progress: {progress_pct:.1f}% | "
                      f"Elapsed: {elapsed_str} | "
                      f"ETA: {eta_str}")
            
            if metrics:
                log_msg += f" | Empathy: {metrics.get('empathy_rate', 0):.1f}%"
                log_msg += f" | Avg Time: {metrics.get('avg_response_time', 0):.1f}s"
            
            print(log_msg)
            
            self.logs.append({
                'step': steps_completed,
                'elapsed': elapsed,
                'eta': eta_seconds,
                'metrics': metrics
            })
            
            return log_msg
        return None


class CPUValidator:
    """CPU-only validator for 702 validation samples"""
    
    def __init__(self, model_path, val_data_path, output_dir="evaluation/validation_results"):
        self.model_path = model_path
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging file
        self.log_file = os.path.join(output_dir, f"validation_{self.timestamp}.log")
        
        print("="*80)
        print("🔧 CPU-ONLY VALIDATION SETUP")
        print("="*80)
        print(f"📍 Model Path: {model_path}")
        print(f"📊 Validation Data: {val_data_path}")
        print(f"📁 Output Directory: {output_dir}")
        print(f"⏱️ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        self._log(f"Validation started at {datetime.now().isoformat()}")
        
        # Load model on CPU
        print("\n🔄 Loading tokenizer...")
        self._log("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        print("\n🔄 Loading model on CPU (this may take a few minutes)...")
        self._log("Loading model on CPU...")
        start_load = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU requires float32
            device_map="cpu",  # Force CPU
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_load
        print(f"✅ Model loaded on CPU in {load_time:.1f} seconds")
        self._log(f"Model loaded in {load_time:.1f}s")
        
        # Initialize crisis detector
        self.detector = CrisisDetector()
        
        # Initialize results storage
        self.results = []
        self.checkpoint_data = {
            'config': {
                'model_path': model_path,
                'val_data_path': val_data_path,
                'device': 'cpu',
                'timestamp': self.timestamp
            },
            'progress': [],
            'samples': [],
            'metrics': {}
        }
        
    def _log(self, message):
        """Write to log file"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - {message}\n")
    
    def load_validation_data(self, max_samples=None):
        """Load all validation data"""
        print(f"\n📂 Loading validation data from {self.val_data_path}...")
        data = []
        with open(self.val_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        total_available = len(data)
        
        if max_samples and max_samples < total_available:
            # Random sample for specified number
            data = random.sample(data, max_samples)
            print(f"✅ Loaded {len(data)} samples (randomly selected from {total_available})")
        else:
            print(f"✅ Loaded ALL {len(data)} validation samples")
        
        self._log(f"Loaded {len(data)} validation samples")
        return data
    
    def _save_checkpoint(self, step, force=False):
        """Save checkpoint similar to training"""
        if step % 50 == 0 or force:
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.json")
            
            checkpoint = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'samples_processed': len(self.results),
                'partial_results': self.results[-50:] if len(self.results) > 50 else self.results,
                'current_metrics': self._calculate_current_metrics()
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2)
            
            self._log(f"Checkpoint saved at step {step}")
    
    def _calculate_current_metrics(self):
        """Calculate current metrics from results"""
        if not self.results:
            return {}
        
        total = len(self.results)
        empathy_count = sum(1 for r in self.results if r.get('has_empathy', False))
        avg_time = sum(r.get('response_time', 0) for r in self.results) / total
        
        crisis_stats = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0}
        for r in self.results:
            level = r.get('risk_level', 'safe')
            crisis_stats[level] = crisis_stats.get(level, 0) + 1
        
        return {
            'total_samples': total,
            'empathy_rate': (empathy_count / total * 100) if total > 0 else 0,
            'avg_response_time': avg_time,
            'crisis_stats': crisis_stats
        }
    
    def validate(self, max_samples=None):
        """Run full validation with training-style progress"""
        val_data = self.load_validation_data(max_samples)
        total_samples = len(val_data)
        
        print("\n" + "="*80)
        print("🧪 STARTING CPU VALIDATION")
        print("="*80)
        print(f"📊 Total Samples: {total_samples}")
        print(f"💻 Device: CPU (torch.float32)")
        print(f"⏱️ Estimated Time: ~{total_samples * 15 // 60} minutes to {total_samples * 30 // 60} minutes")
        print("   (CPU inference is slower than GPU)")
        print("="*80 + "\n")
        
        self._log(f"Starting validation on {total_samples} samples")
        
        # Initialize progress tracker
        progress = ValidationProgress(total_samples, log_every=5)
        
        # Metrics tracking
        correct_responses = 0
        crisis_stats = {'safe': 0, 'low': 0, 'medium': 0, 'high': 0}
        total_response_time = 0
        
        # Process each sample with tqdm progress bar
        for idx, sample in enumerate(tqdm(val_data, desc="🔍 Validating", unit="sample")):
            sample_start = time.time()
            
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
            try:
                conversation = [{"role": "user", "content": user_msg}]
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,  # Shorter for faster inference
                        temperature=0.7,
                        top_p=0.85,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
            except Exception as e:
                self._log(f"Error at sample {idx}: {str(e)}")
                generated_response = "[ERROR]"
            
            response_time = time.time() - sample_start
            total_response_time += response_time
            
            # Check crisis detection
            risk, confidence, triggers = self.detector.classify_risk(user_msg)
            crisis_stats[risk] = crisis_stats.get(risk, 0) + 1
            
            # Check for empathy keywords
            empathy_keywords = ['understand', 'feel', 'sorry', 'here', 'help', 'support', 'listen', 
                               'care', 'difficult', 'hard', 'tough', 'appreciate', 'thank']
            has_empathy = any(keyword in generated_response.lower() for keyword in empathy_keywords)
            
            if has_empathy:
                correct_responses += 1
            
            # Store result
            result = {
                'sample_idx': idx,
                'user_input': user_msg[:200],  # Truncate for storage
                'expected': expected_response[:100],
                'generated': generated_response[:200],
                'has_empathy': has_empathy,
                'risk_level': risk,
                'risk_confidence': confidence,
                'triggers': triggers[:3],
                'response_time': response_time
            }
            self.results.append(result)
            
            # Log progress periodically
            current_metrics = {
                'empathy_rate': (correct_responses / (idx + 1)) * 100,
                'avg_response_time': total_response_time / (idx + 1)
            }
            progress.log_step(idx + 1, current_metrics)
            
            # Save checkpoint
            self._save_checkpoint(idx + 1)
        
        # Final metrics
        total_tested = len(self.results)
        empathy_accuracy = (correct_responses / total_tested * 100) if total_tested > 0 else 0
        avg_time = total_response_time / total_tested if total_tested > 0 else 0
        
        # Print final results
        print("\n" + "="*80)
        print("📊 VALIDATION RESULTS - COMPLETE")
        print("="*80)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total samples validated: {total_tested}")
        print(f"   Empathy accuracy: {empathy_accuracy:.1f}%")
        print(f"   Average response time: {avg_time:.2f}s per sample")
        print(f"   Total validation time: {str(timedelta(seconds=int(total_response_time)))}")
        
        print(f"\n🚨 Crisis Detection Statistics:")
        for level, count in crisis_stats.items():
            percentage = (count / total_tested * 100) if total_tested > 0 else 0
            bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            print(f"   {level.upper():8s}: {count:4d} ({percentage:5.1f}%) |{bar}|")
        
        print(f"\n💡 Interpretation:")
        print(f"   - Safe:   No concerning keywords detected")
        print(f"   - Low:    Mild stress/worry keywords")
        print(f"   - Medium: Burden/hopeless/trapped keywords")
        print(f"   - High:   Suicidal ideation/self-harm keywords")
        
        # Save final results
        final_results = {
            'test_info': {
                'total_samples': total_tested,
                'timestamp': self.timestamp,
                'model_path': self.model_path,
                'data_path': self.val_data_path,
                'device': 'cpu',
                'validation_duration_seconds': total_response_time
            },
            'metrics': {
                'total_samples': total_tested,
                'empathy_accuracy': empathy_accuracy,
                'avg_response_time': avg_time,
                'crisis_statistics': crisis_stats
            },
            'sample_results': self.results[:100]  # First 100 samples for review
        }
        
        results_file = os.path.join(self.output_dir, f"full_validation_{self.timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        print(f"📝 Log file: {self.log_file}")
        print("="*80)
        
        self._log(f"Validation complete. Results saved to {results_file}")
        
        return final_results


def main():
    # Configuration
    MODEL_PATH = "models/merged_model"
    VAL_DATA_PATH = "data/val_full.jsonl"
    
    # Check paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please ensure the merged model exists.")
        sys.exit(1)
    
    if not os.path.exists(VAL_DATA_PATH):
        print(f"❌ Validation data not found at {VAL_DATA_PATH}")
        # Try fallback
        VAL_DATA_PATH = "data/val.jsonl"
        if not os.path.exists(VAL_DATA_PATH):
            print("❌ No validation data found.")
            sys.exit(1)
        print(f"✅ Using fallback: {VAL_DATA_PATH}")
    
    # Check validation data size
    with open(VAL_DATA_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f)
    print(f"\n📊 Validation dataset contains {line_count} samples")
    
    # Create validator
    validator = CPUValidator(MODEL_PATH, VAL_DATA_PATH)
    
    # Run validation on ALL samples (702)
    print(f"\n🚀 Running validation on ALL {line_count} samples...")
    print("   This will take a while on CPU. Progress will be shown below.\n")
    
    results = validator.validate(max_samples=None)  # None = all samples
    
    print("\n✅ CPU Validation Complete!")
    print(f"📊 Final Empathy Accuracy: {results['metrics']['empathy_accuracy']:.1f}%")
    print(f"⏱️ Average Response Time: {results['metrics']['avg_response_time']:.2f}s")
    
    return results


if __name__ == "__main__":
    main()

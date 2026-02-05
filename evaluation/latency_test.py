"""
Latency Measurement Script
Measures REAL response times, tokens/second, and generates latency analysis.
This is REAL testing, not estimated data.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Add project root
sys.path.append(os.path.dirname(__file__))

# Model path
MODEL_PATH = "models/merged_model"

# Test prompts of varying lengths (short, medium, long)
TEST_PROMPTS = [
    # Short prompts (< 10 words)
    {"text": "I feel sad today.", "category": "short"},
    {"text": "I'm stressed about work.", "category": "short"},
    {"text": "Help me with anxiety.", "category": "short"},
    {"text": "I can't sleep well.", "category": "short"},
    {"text": "I feel lonely.", "category": "short"},
    {"text": "I'm worried.", "category": "short"},
    {"text": "Feeling depressed lately.", "category": "short"},
    {"text": "Need someone to talk to.", "category": "short"},
    {"text": "Life is hard.", "category": "short"},
    {"text": "I feel lost.", "category": "short"},
    
    # Medium prompts (10-30 words)
    {"text": "I've been feeling really anxious about my job interview next week and I don't know how to calm down.", "category": "medium"},
    {"text": "My relationship with my parents has been strained lately and it's affecting my mental health a lot.", "category": "medium"},
    {"text": "I keep having negative thoughts about myself and I don't know how to stop them from spiraling.", "category": "medium"},
    {"text": "Work has been overwhelming and I feel like I'm burning out but I can't take a break.", "category": "medium"},
    {"text": "I've been isolating myself from friends and family and I don't know why I do this.", "category": "medium"},
    {"text": "Sometimes I feel like I'm not good enough compared to everyone around me.", "category": "medium"},
    {"text": "I'm struggling with motivation to do anything productive and it's frustrating me a lot.", "category": "medium"},
    {"text": "My sleep schedule is completely messed up and it's affecting everything else in my life.", "category": "medium"},
    {"text": "I feel like nobody understands what I'm going through and it makes me feel even more alone.", "category": "medium"},
    {"text": "I've been having panic attacks recently and they're really scary and overwhelming.", "category": "medium"},
    
    # Long prompts (30-50 words)
    {"text": "I've been dealing with depression for several months now and while I've tried some things like exercise and meditation, nothing seems to help. My family doesn't really understand mental health issues and I feel like I'm carrying this burden alone. What should I do?", "category": "long"},
    {"text": "My marriage has been going through a rough patch and the constant arguments are really taking a toll on my mental health. I have two kids and I'm trying to stay strong for them but inside I feel like I'm falling apart. How do I cope with this?", "category": "long"},
    {"text": "I recently lost my job and I'm feeling completely worthless. I've been applying to many places but keep getting rejected. The financial stress is overwhelming and I find myself unable to sleep at night. Sometimes I wonder if things will ever get better.", "category": "long"},
    {"text": "As a student, the pressure to perform well academically is crushing me. My parents have high expectations and I feel like I'll disappoint them if I don't succeed. I've been experiencing severe anxiety before every exam and my performance is actually getting worse because of it.", "category": "long"},
    {"text": "I've been having intrusive thoughts lately that really scare me. I don't want to act on them but they keep coming back. I haven't told anyone about this because I'm afraid they'll think I'm crazy. Is this normal or should I be worried?", "category": "long"},
    
    # Very long prompts (50+ words)
    {"text": "My father passed away six months ago and I still can't seem to move on from the grief. Every day feels heavy and meaningless without him. I used to be close to my siblings but after his death, we've drifted apart due to inheritance disputes. I feel like I've lost my entire family, not just my dad. I've been neglecting my own health and relationships because I just don't have the energy to care. Sometimes I wonder if this pain will ever go away.", "category": "very_long"},
    {"text": "I've been working in a toxic workplace for three years now. My boss constantly belittles me in front of colleagues, I'm overworked with unrealistic deadlines, and there's no room for growth. I stay because the economy is bad and I have bills to pay. But recently, I've started having chest pains and my doctor says it's stress-related. My wife is worried about me but I feel trapped. I don't know how to balance providing for my family with taking care of my mental health.", "category": "very_long"},
]


def load_model():
    """Load the fine-tuned model."""
    print("🔄 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Model loaded!")
    return model, tokenizer


def measure_latency(model, tokenizer, prompt_text, max_tokens=256):
    """
    Measure latency for a single prompt.
    Returns: dict with timing metrics
    """
    conversation = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs.input_ids.shape[1]
    input_tokens = input_len
    
    # Warmup GPU (first generation might be slower)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Measure time
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.85,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time = end_time - start_time
    output_tokens = outputs.shape[1] - input_len
    tokens_per_second = output_tokens / total_time if total_time > 0 else 0
    time_per_token = total_time / output_tokens if output_tokens > 0 else 0
    
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_time_seconds": total_time,
        "tokens_per_second": tokens_per_second,
        "time_per_token_ms": time_per_token * 1000,  # Convert to milliseconds
        "response_preview": response[:100]
    }


def run_latency_test():
    """Run full latency test and return results."""
    model, tokenizer = load_model()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "device": str(next(model.parameters()).device),
        "total_tests": len(TEST_PROMPTS),
        "measurements": [],
        "by_category": {},
        "summary": {}
    }
    
    print("\n" + "="*80)
    print("⏱️  LATENCY MEASUREMENT TEST - REAL DATA")
    print("="*80)
    print(f"Testing {len(TEST_PROMPTS)} prompts...")
    print(f"Device: {results['device']}")
    print()
    
    all_times = []
    all_tokens_per_sec = []
    all_time_per_token = []
    
    for i, prompt_data in enumerate(TEST_PROMPTS, 1):
        prompt_text = prompt_data["text"]
        category = prompt_data["category"]
        
        print(f"[{i}/{len(TEST_PROMPTS)}] {category.upper()}: {prompt_text[:40]}...", end=" ")
        
        # Measure latency
        metrics = measure_latency(model, tokenizer, prompt_text)
        
        print(f"⏱️ {metrics['total_time_seconds']:.2f}s ({metrics['tokens_per_second']:.1f} tok/s)")
        
        # Store result
        measurement = {
            "prompt": prompt_text,
            "category": category,
            "input_length": len(prompt_text.split()),
            **metrics
        }
        results["measurements"].append(measurement)
        
        # Aggregate metrics
        all_times.append(metrics["total_time_seconds"])
        all_tokens_per_sec.append(metrics["tokens_per_second"])
        all_time_per_token.append(metrics["time_per_token_ms"])
        
        # By category
        if category not in results["by_category"]:
            results["by_category"][category] = {
                "times": [],
                "tokens_per_sec": [],
                "time_per_token_ms": []
            }
        results["by_category"][category]["times"].append(metrics["total_time_seconds"])
        results["by_category"][category]["tokens_per_sec"].append(metrics["tokens_per_second"])
        results["by_category"][category]["time_per_token_ms"].append(metrics["time_per_token_ms"])
    
    # Calculate summary statistics
    results["summary"] = {
        "total_prompts_tested": len(TEST_PROMPTS),
        "average_response_time_seconds": np.mean(all_times),
        "median_response_time_seconds": np.median(all_times),
        "min_response_time_seconds": np.min(all_times),
        "max_response_time_seconds": np.max(all_times),
        "std_response_time_seconds": np.std(all_times),
        "p50_response_time": np.percentile(all_times, 50),
        "p90_response_time": np.percentile(all_times, 90),
        "p95_response_time": np.percentile(all_times, 95),
        "p99_response_time": np.percentile(all_times, 99),
        "average_tokens_per_second": np.mean(all_tokens_per_sec),
        "average_time_per_token_ms": np.mean(all_time_per_token),
    }
    
    # Category summaries
    for cat, data in results["by_category"].items():
        results["by_category"][cat] = {
            "count": len(data["times"]),
            "avg_time": np.mean(data["times"]),
            "avg_tokens_per_sec": np.mean(data["tokens_per_sec"]),
            "avg_time_per_token_ms": np.mean(data["time_per_token_ms"])
        }
    
    # Print summary
    print("\n" + "="*80)
    print("📊 LATENCY TEST RESULTS")
    print("="*80)
    print(f"Total Prompts Tested: {results['summary']['total_prompts_tested']}")
    print()
    print("Response Time Statistics:")
    print(f"  Average: {results['summary']['average_response_time_seconds']:.2f}s")
    print(f"  Median:  {results['summary']['median_response_time_seconds']:.2f}s")
    print(f"  Min:     {results['summary']['min_response_time_seconds']:.2f}s")
    print(f"  Max:     {results['summary']['max_response_time_seconds']:.2f}s")
    print()
    print("Percentiles:")
    print(f"  P50: {results['summary']['p50_response_time']:.2f}s")
    print(f"  P90: {results['summary']['p90_response_time']:.2f}s")
    print(f"  P95: {results['summary']['p95_response_time']:.2f}s")
    print(f"  P99: {results['summary']['p99_response_time']:.2f}s")
    print()
    print("Throughput:")
    print(f"  Average Tokens/Second: {results['summary']['average_tokens_per_second']:.1f}")
    print(f"  Average Time/Token:    {results['summary']['average_time_per_token_ms']:.1f}ms")
    print()
    print("By Prompt Length:")
    for cat, data in results["by_category"].items():
        print(f"  {cat}: {data['avg_time']:.2f}s avg, {data['avg_tokens_per_sec']:.1f} tok/s")
    
    # Usability assessment
    avg_time = results['summary']['average_response_time_seconds']
    print()
    print("="*80)
    if avg_time < 2:
        print("🎯 VERDICT: EXCELLENT - Real-time chat suitable! (<2s response)")
    elif avg_time < 5:
        print("✅ VERDICT: GOOD - Acceptable for chat applications (2-5s response)")
    elif avg_time < 10:
        print("⚠️ VERDICT: FAIR - Noticeable delay but usable (5-10s response)")
    else:
        print("❌ VERDICT: SLOW - May frustrate users (>10s response)")
    print("="*80)
    
    # Save results
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"latency_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = run_latency_test()

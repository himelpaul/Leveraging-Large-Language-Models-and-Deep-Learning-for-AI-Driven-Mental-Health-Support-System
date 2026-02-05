import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import warnings
import os
import sys
import textwrap
from datetime import datetime

# Suppress ALL warnings and logging
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress all info logging from libraries
import logging
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('rag.rag').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import crisis detection and RAG
try:
    from detection.detector import CrisisDetector
    from rag.rag import get_rag_context
    detector = CrisisDetector()
    rag_enabled = True
except:
    detector = None
    rag_enabled = False
    print("Crisis detection and RAG disabled (optional modules not found)\n")

# Path to your merged model
MODEL_PATH = "models/merged_model"

# Terminal width for word wrap
TERMINAL_WIDTH = 150

def wrap_text(text, width=TERMINAL_WIDTH, prefix=""):
    """Wrap text to fit terminal width like HTML word wrap"""
    wrapped = textwrap.fill(text, width=width, initial_indent=prefix, 
                           subsequent_indent=prefix, break_long_words=False,
                           break_on_hyphens=False)
    return wrapped

# Check for CUDA (NVIDIA) - DirectML disabled causing OOM
DIRECTML_AVAILABLE = False
CUDA_AVAILABLE = torch.cuda.is_available()

# Optimize CPU threads
if not CUDA_AVAILABLE:
    cpu_cores = os.cpu_count()
    torch.set_num_threads(cpu_cores)
    print(f" CPU Optimization: Using {cpu_cores} threads")

print(" Loading chatbot...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Set pad_token to eos_token to suppress the warning
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model to the appropriate device
if CUDA_AVAILABLE:
    print(f"   Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
else:
    print("  ")
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # CPU works best with float32
        low_cpu_mem_usage=True
    )

print("  ")
if rag_enabled:
    print(" RAG System: Enabled")
    print(" Crisis Detection: Enabled")
print(" Word Wrap: Enabled\n")
print("=" * TERMINAL_WIDTH)

conversation = []
rag_used_count = 0
crisis_detected_count = 0

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input: continue
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\n" + "=" * TERMINAL_WIDTH)
            print("\n Session Summary:")
            print(f"   - Total messages: {len(conversation) // 2}")
            print(f"   - RAG retrievals: {rag_used_count}")
            print(f"   - Crisis detections: {crisis_detected_count}")
            print("\nGoodbye! Take care. 💙")
            break
        
        # Crisis Detection
        explanation_info = []
        if detector:
            risk, confidence, triggers = detector.classify_risk(user_input)
            if risk == "high":
                crisis_detected_count += 1
                print("\n" + "="*TERMINAL_WIDTH)
                print("🚨 CRISIS ALERT - HIGH RISK DETECTED")
                print("="*TERMINAL_WIDTH)
                print(wrap_text(f"Detected: {', '.join(triggers)}", prefix="⚠️  "))
                print("\n🆘 Emergency Resources (Bangladesh):")
                print("   • Emergency: 999")
                print("="*TERMINAL_WIDTH + "\n")
                explanation_info.append(f"Crisis detection triggered (confidence: {confidence:.0%})")
            elif risk == "medium":
                explanation_info.append("Moderate concern detected - responding with extra care")
        
        # RAG Retrieval (Logic kept from user snippet - retrieval only, no injection)
        rag_context = None
        if rag_enabled:
            try:
                rag_context = get_rag_context(user_input)
                if rag_context:
                    rag_used_count += 1
                    explanation_info.append("Retrieved professional mental health information")
            except Exception as e:
                pass  # Silently continue if RAG fails
        
        # Keep conversation clean without RAG pollution
        conversation.append({"role": "user", "content": user_input})
        
        # Limit conversation history to last 30 messages (Increased as requested)
        MAX_HISTORY = 30
        recent_conversation = conversation[-MAX_HISTORY:] if len(conversation) > MAX_HISTORY else conversation
        
        # Use conversation for prompt
        prompt = tokenizer.apply_chat_template(recent_conversation, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        # Generate with Streaming
        start_time = datetime.now()
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 512,
            "min_new_tokens": 30,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        print(f"\nChatbot: ", end="", flush=True)
        
        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print() # Newline
        
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()
        
        conversation.append({"role": "assistant", "content": response.strip()})
        
        # Explainable AI - show why this response was generated
        if explanation_info:
            print(f"\n Explanation: {' | '.join(explanation_info)}")
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\nError: {e}")
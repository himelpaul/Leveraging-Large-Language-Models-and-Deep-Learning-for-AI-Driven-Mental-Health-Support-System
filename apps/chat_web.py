import sys
import os
import yaml
import gradio as gr
from llama_cpp import Llama

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.rag import get_rag_context
from detection.detector import CrisisDetector

# Initialize detector globally
detector = CrisisDetector()

def load_config():
    with open("apps/config.yaml", 'r') as f:
        return yaml.safe_load(f)

# Load model globally
config = load_config()
model_path = config['model_path']
llm = None

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=config['context_window'],
            verbose=False,
            # chat_format="llama-3" # optional, we construct prompt manually
        )
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model not found at {model_path}. Please run training and conversion first.")

def chat_response(message, history):
    if not llm:
        yield "Error: Model not loaded. Please check if the model exists (run training first).", ""
        return

    # Risk Detection (XAI Enhanced)
    risk_level = "low"
    risk_score = 0.0
    triggers = []
    
    try:
        risk_level, risk_score, triggers = detector.classify_risk(message)
    except Exception as e:
        print(f"XAI Error: {e}")
    
    # Generate XAI Explanation with more details
    xai_explanation = f"### 🧠 System Diagnostics (XAI)\n"
    xai_explanation += f"- **Risk Assessment**: {risk_level.upper()}\n"
    xai_explanation += f"- **Confidence Score**: {risk_score:.2f}\n"
    
    alert_msg = ""
    if risk_level == "high":
        alert_msg = "**⚠️ SYSTEM ALERT: High risk detected. Please call 988 or emergency services immediately.**\n\n"
        xai_explanation += f"- **Triggered By**: {', '.join(triggers)}\n"
        xai_explanation += "- **Action**: Emergency Protocol Activated (Keyword Triggered)\n"
    elif risk_level == "medium":
        alert_msg = "*Note: I sense you are going through a tough time. I'm here to listen, but please consider talking to a professional.*\n\n"
        xai_explanation += f"- **Triggered By**: {', '.join(triggers)}\n"
    
    # RAG Retrieval (XAI Enhanced)
    rag_context = get_rag_context(message)
    
    if rag_context:
        xai_explanation += "- **Knowledge Source**: RAG Retrieval (Verified Documents)\n"
        # Extract source hints if possible, for now we just show context length
        xai_explanation += f"- **Context Retrieved**: {len(rag_context)} characters\n"
    else:
        xai_explanation += "- **Knowledge Source**: Internal Model Weights (General Reasoning)\n"

    augmented_message = message
    if rag_context:
        augmented_message = f"Context information is below.\n---------------------\n{rag_context}\n---------------------\nQuery: {message}"

    # Prompt Construction
    system_prompt = config['system_prompt']
    
    # Llama 3 Format
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    
    # History (Limited to last 10 turns to match CLI and save context window)
    for user_msg, bot_msg in history[-10:]:
        if user_msg:
            formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        if bot_msg:
            formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>"
    
    formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{augmented_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    # STOPPING FIX: Add multiple stop tokens so it doesn't ramble
    stream = llm(
        formatted_prompt,
        max_tokens=config['max_tokens'],
        stop=["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "User:", "\nUser"],
        echo=False,
        temperature=config['temperature'],
        top_p=config['top_p'],
        stream=True

    )
    
    partial_response = alert_msg
    
    for chunk in stream:
        delta = chunk['choices'][0]['text']
        partial_response += delta
        yield partial_response, xai_explanation

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Mental Health Support Chatbot (Thesis Project)")
    gr.Markdown("A specialized AI assistant with Crisis Detection and Explainability (XAI).")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat Window", height=500)
            msg = gr.Textbox(label="Type your message here...", placeholder="How are you feeling today?")
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear History")
                
        with gr.Column(scale=1):
            xai_panel = gr.Markdown("### 🔍 Explainable AI (XAI) Panel\nDiagnostics will appear here after your message.")
    
    def user_turn(user_message, history):
        return "", history + [[user_message, None]]

    def bot_turn(history):
        user_message = history[-1][0]
        history_list = history[:-1] # Exclude current empty bot message
        
        bot_response_generator = chat_response(user_message, history_list)
        
        for response, explanation in bot_response_generator:
            history[-1][1] = response
            yield history, explanation
            
    msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot], [chatbot, xai_panel]
    )
    submit_btn.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_turn, [chatbot], [chatbot, xai_panel]
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True)

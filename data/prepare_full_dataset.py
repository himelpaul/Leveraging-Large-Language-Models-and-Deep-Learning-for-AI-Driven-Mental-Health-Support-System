"""
Prepare LARGE combined dataset:
- 500 complete conversations from train.csv (best quality, multi-turn)
- ALL 3512 from Mental Health Counseling Conversations
- ALL 3000 from cultural_synthetic.json
Total: ~7000+ conversations
"""
import pandas as pd
import json
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_train_csv_conversation(conv_str):
    """Parse train.csv conversation string using regex"""
    pattern = r"\{'from':\s*'(human|gpt)',\s*'value':\s*\"((?:[^\"\\]|\\.)*)\"\}"
    matches = re.findall(pattern, conv_str, re.DOTALL)
    
    messages = []
    for role, content in matches:
        if role == 'human':
            messages.append({"role": "user", "content": content})
        elif role == 'gpt':
            messages.append({"role": "assistant", "content": content})
    
    return messages

def calculate_quality_score(messages):
    """Calculate quality score for ranking"""
    if not messages:
        return 0
    
    num_turns = len(messages)
    avg_length = sum(len(msg['content']) for msg in messages) / num_turns
    
    # Prefer 8-20 turns
    turn_score = min(num_turns / 20.0, 1.0) * 40
    
    # Prefer substantive messages
    if 100 <= avg_length <= 500:
        length_score = 40
    elif 50 <= avg_length < 100 or 500 < avg_length <= 800:
        length_score = 20
    else:
        length_score = 10
    
    # Balance
    user_count = sum(1 for m in messages if m['role'] == 'user')
    assistant_count = sum(1 for m in messages if m['role'] == 'assistant')
    balance_score = 20 if abs(user_count - assistant_count) <= 1 else 10
    
    return turn_score + length_score + balance_score

def load_train_csv_best500():
    """Load best 500 complete conversations from train.csv"""
    logger.info("Loading train.csv...")
    df = pd.read_csv('data/train.csv')
    logger.info(f"Total rows in train.csv: {len(df)}")
    
    conversations_with_scores = []
    
    logger.info("Parsing and scoring conversations...")
    for idx, row in df.iterrows():
        try:
            messages = parse_train_csv_conversation(row['conversations'])
            
            if len(messages) >= 6:
                score = calculate_quality_score(messages)
                conversations_with_scores.append({
                    'messages': messages,
                    'score': score,
                    'num_turns': len(messages)
                })
        except Exception as e:
            continue
    
    # Sort by quality and take top 500
    conversations_with_scores.sort(key=lambda x: x['score'], reverse=True)
    top_500 = conversations_with_scores[:500]
    
    logger.info(f"Selected top 500 conversations from train.csv")
    logger.info(f"  Avg turns: {sum(c['num_turns'] for c in top_500) / len(top_500):.1f}")
    
    return [{"messages": c['messages']} for c in top_500]

def load_mental_health_all():
    """Load ALL Mental Health Counseling Conversations"""
    logger.info("Loading Mental_Health_Counseling_Conversations.csv...")
    df = pd.read_csv('data/Mental_Health_Counseling_Conversations.csv')
    logger.info(f"Total rows: {len(df)}")
    
    conversations = []
    for _, row in df.iterrows():
        # Ensure content is string
        context = str(row['Context']) if pd.notna(row['Context']) else ""
        response = str(row['Response']) if pd.notna(row['Response']) else ""
        
        messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": response}
        ]
        conversations.append({"messages": messages})
    
    logger.info(f"Loaded {len(conversations)} Mental Health conversations")
    return conversations

def load_cultural_synthetic_all():
    """Load ALL cultural_synthetic.json"""
    logger.info("Loading cultural_synthetic.json...")
    with open('data/cultural_synthetic.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} Cultural Synthetic conversations")
    return data

def main():
    # Load from all three sources
    logger.info("\n=== Loading Data from 3 Sources ===\n")
    
    train_csv_500 = load_train_csv_best500()
    mental_health_all = load_mental_health_all()
    cultural_synthetic_all = load_cultural_synthetic_all()
    
    # Combine all
    all_conversations = train_csv_500 + mental_health_all + cultural_synthetic_all
    
    logger.info(f"\n=== COMBINED DATASET ===")
    logger.info(f"train.csv (best 500): {len(train_csv_500)}")
    logger.info(f"Mental Health (all): {len(mental_health_all)}")
    logger.info(f"Cultural Synthetic (all): {len(cultural_synthetic_all)}")
    logger.info(f"TOTAL: {len(all_conversations)} conversations")
    
    # Split 90% train, 10% validation
    split_idx = int(len(all_conversations) * 0.9)
    train_data = all_conversations[:split_idx]
    val_data = all_conversations[split_idx:]
    
    logger.info(f"\nTrain: {len(train_data)}")
    logger.info(f"Validation: {len(val_data)}")
    
    # Save to JSONL
    train_file = 'data/train_full.jsonl'
    val_file = 'data/val_full.jsonl'
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"\n✓ Created {train_file}")
    logger.info(f"✓ Created {val_file}")
    
    # Show samples from each source
    print("\n=== SAMPLE FROM TRAIN.CSV (Multi-turn) ===")
    sample = train_csv_500[0]['messages'][:4]
    for msg in sample:
        print(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")
    
    print("\n=== SAMPLE FROM MENTAL HEALTH ===")
    sample = mental_health_all[0]['messages']
    for msg in sample:
        print(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")
    
    print("\n=== SAMPLE FROM CULTURAL SYNTHETIC ===")
    sample = cultural_synthetic_all[0]['messages']
    for msg in sample:
        print(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")

if __name__ == "__main__":
    main()

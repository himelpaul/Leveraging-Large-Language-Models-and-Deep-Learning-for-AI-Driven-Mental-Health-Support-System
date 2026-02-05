import json
import sys
import os

def check_dataset(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "messages" not in data:
                        print(f"Error line {i+1}: Missing 'messages' key")
                        return False
                    if not isinstance(data["messages"], list):
                        print(f"Error line {i+1}: 'messages' must be a list")
                        return False
                    for msg in data["messages"]:
                        if "role" not in msg or "content" not in msg:
                            print(f"Error line {i+1}: Message missing role or content")
                            return False
                except json.JSONDecodeError:
                    print(f"Error line {i+1}: Invalid JSON")
                    return False
        print("Dataset is VALID.")
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = ["data/train.jsonl", "data/val.jsonl", "data/test.jsonl"]
    
    all_valid = True
    for f in files:
        if not check_dataset(f):
            all_valid = False
    
    if not all_valid:
        sys.exit(1)

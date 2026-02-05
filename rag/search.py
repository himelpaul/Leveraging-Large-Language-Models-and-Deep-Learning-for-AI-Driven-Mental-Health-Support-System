import sys
from rag import get_rag_context

def main():
    print("RAG Search Tool")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter query: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        context = get_rag_context(query)
        if context:
            print("\n--- Retrieved Context ---")
            print(context)
            print("-------------------------")
        else:
            print("No context found or index not built.")

if __name__ == "__main__":
    main()

from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Consistent with create_database.py
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

def query_chroma(query_text: str):
    # Use the same embeddings as in create_database.py
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the Chroma DB with the new import
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Perform the query
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    # Process and return results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    return formatted_results

def main():
    import sys
    if len(sys.argv) < 2:
        print("Please provide a query as an argument.")
        return
    
    query_text = " ".join(sys.argv[1:])
    results = query_chroma(query_text)
    
    print(f"Results for query: '{query_text}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['content']}")
        print(f"Source: {result['metadata'].get('source', 'N/A')}")
        print("-" * 80)

if __name__ == "__main__":
    main()
    
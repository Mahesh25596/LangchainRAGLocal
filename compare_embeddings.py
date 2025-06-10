from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    # Initialize HuggingFace embeddings (same model as used in your other scripts)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Get embedding for a word
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector[:5]}...")  # Showing first 5 dimensions for brevity
    print(f"Vector length: {len(vector)}")

    # Compare vectors of two words
    evaluator = load_evaluator("pairwise_embedding_distance", embeddings=embedding_function)
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

    # Additional comparison examples
    comparisons = [
        ("apple", "banana"),
        ("apple", "fruit"),
        ("apple", "computer")
    ]
    for word1, word2 in comparisons:
        result = evaluator.evaluate_string_pairs(prediction=word1, prediction_b=word2)
        print(f"Similarity between '{word1}' and '{word2}': {result['score']:.4f}")

if __name__ == "__main__":
    main()

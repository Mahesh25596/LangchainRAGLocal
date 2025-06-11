from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.llms import Ollama  # For local LLMs via Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

def query_chroma(query_text: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
    return formatted_results

def generate_response(query_text: str, context: str):
    # Initialize a local LLM (using Ollama as an example)
    llm = Ollama(model="llama2")  # or "mistral", "gemma", etc.

    # Define a prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on the provided context. 
        If you don't know the answer, say you don't know.

        Question: {question}

        Context: {context}

        Answer:"""
    )

    # Create a chain
    chain = prompt | llm | StrOutputParser()

    # Generate the response
    response = chain.invoke({
        "question": query_text,
        "context": context
    })

    return response

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

    # Combine the top results as context for the LLM
    context = "\n\n".join([result['content'] for result in results])
    
    # Generate a response using the local LLM
    llm_response = generate_response(query_text, context)
    
    print("\nGenerated LLM Response:")
    print(llm_response)

if __name__ == "__main__":
    main()
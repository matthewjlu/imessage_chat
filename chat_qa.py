from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from datetime import datetime
import os

def load_messages(csv_file):
    """Load messages from CSV and format them with timestamps"""
    df = pd.read_csv(csv_file)
    formatted_messages = []
    for _, row in df.iterrows():
        timestamp = pd.to_datetime(row['Timestamp'])
        formatted_messages.append(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {row['Message']}")
    return "\n".join(formatted_messages)

def setup_qa_system():
    """Set up the question answering system"""
    # Load messages
    messages = load_messages("keerthi_combined.csv")
    
    # Split text into chunks with better overlap and smaller size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for more precise context
        chunk_overlap=100,  # More overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Better separators for text splitting
    )
    chunks = text_splitter.split_text(messages)
    
    # Use a better embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Better embedding model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Normalize embeddings for better similarity search
    )
    
    # Check if vector store exists
    if os.path.exists("vector_store"):
        print("Loading existing vector store...")
        vectorstore = FAISS.load_local(
            "vector_store", 
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new vector store...")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local("vector_store")
    
    # Use a better LLM model
    llm = Ollama(
        model="mistral:latest",  # Use latest version
        temperature=0.1,  # Lower temperature for more focused answers
        num_ctx=4096  # Larger context window
    )
    
    # Create a custom prompt template
    prompt_template = """You are a helpful assistant that answers questions about text message conversations.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If the question is about timing or dates, make sure to include the specific timestamp from the context.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain with better parameters
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 5,  # Retrieve more documents for better context
                "score_threshold": 0.5  # Only use relevant documents
            }
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    print("Setting up the question answering system...")
    qa_chain = setup_qa_system()
    print("\nSystem ready! You can now ask questions about your conversations.")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        try:
            result = qa_chain.invoke({"query": question})
            print("\nAnswer:", result['result'])
            print("\nSources:")
            for doc in result['source_documents']:
                print(f"- {doc.page_content[:200]}...")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
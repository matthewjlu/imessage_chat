from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM 
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pandas as pd
from datetime import datetime
import os
import re
from last_csv import combined_csv

def load_messages(csv_file):
    """Load messages from CSV and format them with timestamps and message type"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(
        df['Timestamp'],
        format="%b %d, %Y %I:%M:%S %p",
        errors="raise"
    )
    df = df.sort_values('datetime')
    
    # Format messages with timestamp and whether they were sent or received
    formatted_messages = []
    for _, row in df.iterrows():
        timestamp = row['datetime']
        message_type = row['Type']  # 'sent' or 'received'
        formatted_messages.append(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{message_type}] {row['Message']}")
    
    return "\n".join(formatted_messages), df

def setup_qa_system():
    """Set up the question answering system"""
    # Load messages
    messages, messages_df = load_messages(combined_csv)
    print(f"Loaded {len(messages.split(chr(10)))} messages")
    
    # Split text into chunks with better overlap and smaller size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks to keep more context
        chunk_overlap=200,  # More overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Better separators for text splitting
    )
    chunks = text_splitter.split_text(messages)
    print(f"Created {len(chunks)} chunks")
    
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
    
    # Use the new LLM model with the updated class
    llm = OllamaLLM(
        model="mistral:latest",  # Use latest version
        temperature=0.1,  # Lower temperature for more focused answers
        num_ctx=4096  # Larger context window
    )
    
    # Create a more restrictive prompt template to prevent hallucination
    prompt_template = """You are a helpful assistant that answers questions about text message conversations. Your task is to answer the question using ONLY the specific context provided below.

CRITICAL INSTRUCTIONS:
1. You MUST use ONLY the information from the context provided below.
2. If the answer is not explicitly in the context, say "I cannot find this information in the provided messages."
3. Do NOT use any prior knowledge or make assumptions outside of what is provided.
4. Do NOT invent or hallucinate any timestamps, dates, or message content.
5. Timestamps in the context will appear in this format: [YYYY-MM-DD HH:MM:SS]. For example: [2025-03-19 22:55:34]
6. When you reference a timestamp in your answer, REFORMAT the date part into "Month Day, Year" format. For example:
   - If the timestamp is [2025-03-19 22:55:34], say: "March 19, 2025 at 10:55 PM"
7. DO NOT repeat the timestamp in YYYY-MM-DD format. Always convert it as shown.
8. Always mention whether the message was [sent] or [received].

When answering questions about when something happened, ALWAYS reformat the timestamp as described above.

Context:
{context}

Question: {question}

Your answer MUST be based ONLY on the above context. If the information isn't present in the context, admit you don't know.

Answer:"""

    
    # Create a prompt using PromptTemplate
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Build a runnable pipeline: prompt | llm
    chain = prompt | llm
    
    # Keep the vectorstore retriever for content-based searches
    # Replace your existing retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"fetch_k": 20, "k": 5, "lambda_mult": 0.5}
    )
    
    return chain, retriever, messages_df

def is_temporal_query(query):
    """Check if the query is related to time or recency"""
    temporal_patterns = [
        r'recent', r'last', r'latest', r'newest', r'most recent',
        r'today', r'yesterday', r'this week', r'this month',
        r'first', r'earlier', r'oldest', r'when'
    ]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in temporal_patterns)

def is_sender_specific_query(query):
    """Check if the query is about who sent/received messages"""
    sender_patterns = [
        r'(did|have|has) (I|you|we) (send|sent|receive|received)',
        r'(my|your|our) (message|text)',
        r'who (sent|send|wrote|texted)',
        r'was (sent|received)',
        r'(sent|send) (by|from)'
    ]
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in sender_patterns)

def get_docs_by_time(df, query, k=5):
    """Get documents based on time criteria"""
    # Check for recency keywords
    if re.search(r'recent|last|latest|newest', query, re.IGNORECASE):
        if re.search(r'(I|we) (send|sent)', query, re.IGNORECASE):
            filtered_df = df[df['Type'] == 'sent'].tail(k)
        elif re.search(r'(received|got)', query, re.IGNORECASE):
            filtered_df = df[df['Type'] == 'received'].tail(k)
        else:
            filtered_df = df.tail(k)
    elif re.search(r'oldest|first|earlier', query, re.IGNORECASE):
        if re.search(r'(I|we) (send|sent)', query, re.IGNORECASE):
            filtered_df = df[df['Type'] == 'sent'].head(k)
        elif re.search(r'(received|got)', query, re.IGNORECASE):
            filtered_df = df[df['Type'] == 'received'].head(k)
        else:
            filtered_df = df.head(k)
    else:
        filtered_df = df.tail(k)
    
    # Convert to Document format
    docs = []
    for _, row in filtered_df.iterrows():
        timestamp = row['datetime']
        message_type = row['Type']
        content = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{message_type}] {row['Message']}"
        docs.append(Document(page_content=content))
    
    return docs

def get_docs_by_sender(df, query, k=5):
    """Get documents based on sender/receiver criteria"""
    if re.search(r'(I|we) (send|sent)', query, re.IGNORECASE):
        filtered_df = df[df['Type'] == 'sent'].tail(k)
    elif re.search(r'(received|got)', query, re.IGNORECASE):
        filtered_df = df[df['Type'] == 'received'].tail(k)
    else:
        filtered_df = df.tail(k)
    
    docs = []
    for _, row in filtered_df.iterrows():
        timestamp = row['datetime']
        message_type = row['Type']
        content = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{message_type}] {row['Message']}"
        docs.append(Document(page_content=content))
    
    return docs

def verify_answer(answer, context, question):
    """Basic verification to catch obvious hallucinations"""
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates_in_answer = re.findall(date_pattern, answer)
    dates_in_context = re.findall(date_pattern, context)
    
    for date in dates_in_answer:
        if date not in dates_in_context:
            return False, f"Warning: Date '{date}' mentioned in the answer is not found in the context."
    
    if is_temporal_query(question) and re.search(r'recent|last|latest', question, re.IGNORECASE):
        if dates_in_answer and dates_in_context:
            latest_date_in_context = sorted(dates_in_context)[-1]
            if any(date != latest_date_in_context for date in dates_in_answer):
                return False, f"Warning: Answer mentions a date that is not the most recent date in the context."
    
    return True, None

def main():
    print("Setting up the question answering system...")
    chain, retriever, messages_df = setup_qa_system()
    print("\nSystem ready! You can now ask questions about your conversations.")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        try:
            # Determine the query type and use appropriate retrieval method
            if is_temporal_query(question):
                print("Detected temporal query - using time-based retrieval")
                docs = get_docs_by_time(messages_df, question, k=5)
            elif is_sender_specific_query(question):
                print("Detected sender-specific query - using sender-based retrieval")
                docs = get_docs_by_sender(messages_df, question, k=5)
            else:
                print("Using semantic similarity search")
                docs = retriever.get_relevant_documents(question)
            
            # Print retrieved documents
            print("\nRelevant messages:")
            for i, doc in enumerate(docs, 1):
                print(f"\n{i}. {doc.page_content}")
            
            # Use the LLM chain with the retrieved documents
            context = "\n".join(doc.page_content for doc in docs)
            result = chain.invoke({
                "question": question,
                "context": context
            })
            
            # Extract answer text (if response is a dict)
            answer = result.get('text') if isinstance(result, dict) else result
            
            # Verify the answer for potential hallucinations
            is_valid, warning = verify_answer(answer, context, question)
            if not is_valid:
                print("\nWARNING:", warning)
                print("Original answer may contain hallucinations. Attempting to correct...")
                corrected_context = f"IMPORTANT: The correct information is LIMITED TO these messages ONLY:\n\n{context}"
                result = chain.invoke({
                    "question": question,
                    "context": corrected_context
                })
                answer = result.get('text') if isinstance(result, dict) else result
            
            print("\nAnswer:", answer)
                
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()

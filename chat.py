import os
import re
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict
from typing import Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from last_csv import combined_csv
from config import (
    PERPLEXITY_API_KEY,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    OPENAI_API_KEY
)

# Additional imports for improved display using Rich
from rich.console import Console
from rich.table import Table

# Set up logging for debug information and error tracking.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def setup_environment():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    os.environ['PINECONE_ENVIRONMENT'] = PINECONE_ENVIRONMENT
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

PROMPT = PromptTemplate(
    template="""You are a helpful assistant that answers questions about text message conversations. Your task is to answer the question using ONLY the specific context provided below.

CRITICAL INSTRUCTIONS:
1. You MUST use ONLY the information from the context provided below.
2. If the answer is not explicitly in the context, say "I cannot find this information in the provided messages.".
3. Do NOT use any prior knowledge or make assumptions outside of what is provided.
4. Do NOT invent or hallucinate any timestamps, dates, or message content.
5. Timestamps in the context will appear in this format: [YYYY-MM-DD HH:MM:SS].
6. When you reference a timestamp, reformat it as "Month Day, Year at HH:MM AM/PM".
7. Always indicate [sent] or [received].

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

INTENT_PROMPT = PromptTemplate(
    template="""Classify the following question as asking for the earliest, latest, or neither message. Respond ONLY with one of: earliest, latest, neither.

Examples:
Question: What was our first text message? -> earliest
Question: When did we last text? -> latest
Question: What did I ask you yesterday? -> neither

Question: {question}
Answer:""",
    input_variables=["question"]
)

def load_messages(csv_path: str) -> Tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file at {csv_path}: {e}")
        raise

    required_columns = ['Timestamp', 'Type', 'Message']
    if not all(col in df.columns for col in required_columns):
        error_msg = f"CSV file is missing one of the required columns: {required_columns}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    try:
        df['datetime'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y %I:%M:%S %p")
    except Exception as e:
        logging.error(f"Error parsing 'Timestamp' column: {e}")
        raise

    df = df.sort_values('datetime').reset_index(drop=True)
    text = "\n".join(
        f"[{row.datetime:%Y-%m-%d %H:%M:%S}] [{row.Type}] {row.Message}"
        for _, row in df.iterrows()
    )
    return df, text

def setup_qa_system():
    try:
        df, text = load_messages(combined_csv)
    except Exception as e:
        logging.error("Error loading messages.")
        raise e

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        raise

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
        vectorstore = PineconeVectorStore.from_texts(chunks, embeddings, index_name=PINECONE_INDEX_NAME, text_key="text")
    except Exception as e:
        logging.error(f"Error setting up embeddings or vector store: {e}")
        raise

    try:
        # Use 'similarity' search instead of 'mmr' for a more direct relevance ranking.
        retriever = MultiQueryRetriever.from_llm(
            vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            llm=ChatPerplexity(model="llama-3.1-sonar-small-128k-online", temperature=0.1, pplx_api_key=PERPLEXITY_API_KEY)
        )
    except Exception as e:
        logging.error(f"Error setting up MultiQueryRetriever: {e}")
        raise

    try:
        intent_chain = INTENT_PROMPT | ChatPerplexity(model="llama-3.1-sonar-small-128k-online", temperature=0.0, pplx_api_key=PERPLEXITY_API_KEY)
        qa_chain = PROMPT | ChatPerplexity(model="llama-3.1-sonar-small-128k-online", temperature=0.1, pplx_api_key=PERPLEXITY_API_KEY)
    except Exception as e:
        logging.error(f"Error setting up chains: {e}")
        raise

    return qa_chain, retriever, df, intent_chain

def display_top_messages(messages, top_n=5):
    """Display top N messages (assumed sorted by relevance) using a Rich table."""
    console = Console()
    table = Table(title=f"Top {top_n} Most Related Responses", show_header=True, header_style="bold magenta")
    table.add_column("Index", style="dim", width=6)
    table.add_column("Message", min_width=60)
    for idx, doc in enumerate(messages[:top_n], start=1):
        table.add_row(str(idx), doc.page_content)
    console.print(table)

def main():
    setup_environment()
    try:
        qa_chain, retriever, df, intent_chain = setup_qa_system()
    except Exception as e:
        logging.error("Failed to set up the QA system. Exiting.")
        return

    print("System ready — ask your questions (type 'exit' to quit)")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except EOFError:
            break

        if question.lower() == 'exit':
            break

        try:
            intent_response = intent_chain.invoke({"question": question})
            intent = intent_response.content.strip().lower()
        except Exception as e:
            logging.error(f"Error invoking intent chain: {e}")
            print("An error occurred while processing your question. Please try again.")
            continue

        record = None
        if intent == 'latest':
            record = df.iloc[-1]
        elif intent == 'earliest':
            record = df.iloc[0]

        if record is not None:
            ts = record.datetime.strftime("%B %d, %Y at %I:%M %p")
            direction = '[sent]' if record.Type.lower() == 'sent' else '[received]'
            print(f"\nAnswer: The {intent} message was {direction} on {ts}: {record.Message}")
            continue

        try:
            docs = retriever.invoke(question)
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            print("An error occurred while retrieving messages. Please try again.")
            continue

        if not docs:
            print("No relevant messages found.")
            continue

        # Take the top 5 most related documents.
        top_docs = docs[:5]
        print("\nTop 5 most related responses:")
        display_top_messages(top_docs, top_n=5)
        # Use the top responses as context.
        context = "\n".join(doc.page_content for doc in top_docs)

        try:
            resp = qa_chain.invoke({"question": question, "context": context})
            answer_content = resp.content.strip()
        except Exception as e:
            logging.error(f"Error invoking QA chain: {e}")
            print("An error occurred while generating an answer. Please try again.")
            continue

        print("\nAnswer:", answer_content)

if __name__ == '__main__':
    main()

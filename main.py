import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

##########################################
# Part 1: Data Ingestion & Preprocessing #
##########################################
def load_and_preprocess_data(csv_file):
    """
    Load the CSV file and preprocess the conversation data.
    Assumes the CSV has "Timestamp" and "Message" columns.
    """
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates()
    df['Message'] = df['Message'].astype(str).str.strip()
    # Parse timestamps; unparseable values become NaT
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%b %d, %Y %I:%M:%S %p", errors='coerce')
    return df

##########################################
# Part 2: Create a Synthetic Temporal Dataset #
##########################################
def create_temporal_dataset(df):
    """
    Generate synthetic training examples from the CSV.
    For example, for the first few messages and the latest message:
       "What is our first text together?" -> first message
       "What is our second text together?" -> second message
       ...
       "What is our latest text together?" -> last message
    Returns a list of dictionaries.
    """
    df_valid = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    messages = df_valid["Message"].tolist()
    num_messages = len(messages)
    examples = []
    ordinals = ["first", "second", "third", "fourth", "fifth"]
    # Create examples for the first few messages.
    for i in range(min(5, num_messages)):
        query = f"What is our {ordinals[i]} text together?"
        answer = messages[i]
        examples.append({"input_text": query, "target_text": answer})
    # Create example for the latest message.
    if num_messages > 0:
        query = "What is our latest text together?"
        answer = messages[-1]
        examples.append({"input_text": query, "target_text": answer})
    return examples

##########################################
# Part 3: Fine-tune a T5 Model for Temporal Queries #
##########################################
def fine_tune_temporal_model(df, model_name="t5-small", output_dir="./fine_tuned_temporal_model"):
    """
    Fine-tune T5 on synthetic temporal queries derived from df.
    Returns the fine-tuned model and tokenizer.
    """
    # Create synthetic training examples
    examples = create_temporal_dataset(df)
    print("Training examples:")
    for ex in examples:
        print(f"Q: {ex['input_text']}\nA: {ex['target_text']}\n")
    
    # Create a Hugging Face dataset
    dataset = Dataset.from_dict({
        "input_text": [ex["input_text"] for ex in examples],
        "target_text": [ex["target_text"] for ex in examples]
    })
    
    # Load T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Define tokenization function (we don't add padding here; the data collator will handle it)
    def tokenize_function(example):
        model_inputs = tokenizer(example["input_text"], max_length=64, truncation=True)
        targets = tokenizer(example["target_text"], max_length=128, truncation=True)
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create a data collator that pads inputs dynamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,              # Adjust epochs as needed
        per_device_train_batch_size=2,    # Adjust based on GPU memory
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting fine-tuning of temporal model...")
    trainer.train()
    print("Fine-tuning complete.")
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer

##########################################
# Part 4: Retrieval and QA Pipeline (as before) #
##########################################
def compute_embeddings(texts, model):
    return model.encode(texts, convert_to_numpy=True)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_index(query, embedding_model, index, texts, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    candidates = [texts[i] for i in indices[0]]
    return candidates

def answer_question_with_qa(question, embedding_model, index, texts, qa_pipeline, top_k=5):
    candidates = search_index(question, embedding_model, index, texts, top_k=top_k)
    context = " ".join(candidates)
    print("\nRetrieved candidate snippets:")
    for i, snippet in enumerate(candidates, start=1):
        print(f"{i}. {snippet}")
    try:
        qa_result = qa_pipeline(question=question, context=context)
        answer = qa_result.get("answer", "No answer found.")
    except Exception as e:
        print(f"QA pipeline error: {e}")
        answer = "Error generating answer."
    return answer

##########################################
# Part 5: Temporal Query Handling in Main Pipeline #
##########################################
def answer_question_temporal(user_query, df, embedding_model, index, texts, qa_pipeline,
                             temporal_model=None, temporal_tokenizer=None, top_k=5):
    """
    Handle temporal queries using fine-tuned model if available.
    - "first" queries: return the earliest message.
    - "latest" queries: return the most recent message.
    - "conversation ..." queries: return all messages for a given month.
    - Otherwise, if the query seems temporal and a fine-tuned temporal model is available, use it.
    - Else, use standard QA retrieval.
    """
    query_lower = user_query.lower()

    # "first" query
    if re.search(r'\b(first text together|first message together|our first text|first message between)\b', query_lower):
        df_valid = df.dropna(subset=["Timestamp"])
        if df_valid.empty:
            return "No valid conversation data available."
        earliest_row = df_valid.sort_values('Timestamp').iloc[0]
        answer = f"The earliest text was sent at {earliest_row['Timestamp']:%Y-%m-%d %H:%M:%S} and it said: {earliest_row['Message']}"
        return answer

    # "latest" query
    elif re.search(r'\b(latest text together|latest message together|our latest text|latest message between|most recent text)\b', query_lower):
        df_valid = df.dropna(subset=["Timestamp"])
        if df_valid.empty:
            return "No valid conversation data available."
        latest_row = df_valid.sort_values('Timestamp', ascending=False).iloc[0]
        answer = f"The latest text was sent at {latest_row['Timestamp']:%Y-%m-%d %H:%M:%S} and it said: {latest_row['Message']}"
        return answer

    # "conversation" query (by month)
    elif re.search(r'\b(conversation|chat)\b', query_lower) and extract_month_from_query(user_query):
        month = extract_month_from_query(user_query)
        df_month = df[df['Timestamp'].dt.month == month].dropna(subset=["Timestamp"])
        if df_month.empty:
            return f"No messages found for month {month}."
        conversation_text = "\n".join(df_month.sort_values('Timestamp')['Message'].tolist())
        answer = f"Conversation in month {month}:\n{conversation_text}"
        return answer

    # If query is temporal (e.g., mentions "text together") and we have a fine-tuned temporal model:
    elif re.search(r'\b(text together|message together)\b', query_lower):
        if temporal_model is not None and temporal_tokenizer is not None:
            print("\nUsing fine-tuned temporal model...")
            input_ids = temporal_tokenizer.encode(user_query, return_tensors="pt", truncation=True)
            outputs = temporal_model.generate(input_ids, max_length=128)
            return temporal_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Otherwise, use standard QA retrieval.
    return answer_question_with_qa(user_query, embedding_model, index, texts, qa_pipeline, top_k)

##########################################
# Part 6: Main Pipeline Integration #
##########################################
if __name__ == '__main__':
    csv_file = "keerthi_combined.csv"  # Update with your CSV file path
    df = load_and_preprocess_data(csv_file)
    
    # Optional: initial filtering (e.g., "messages from june")
    initial_query = input("Enter an initial filter query (e.g., 'messages from june') or press Enter to use all data: ").strip()
    if initial_query:
        month_filter = extract_month_from_query(initial_query)
        if month_filter:
            df_filtered = df[df['Timestamp'].dt.month == month_filter]
            if df_filtered.empty:
                print(f"No messages found in month {month_filter}. Using all data.")
                texts = df['Message'].tolist()
            else:
                texts = df_filtered['Message'].tolist()
                print(f"Found {len(texts)} messages from month {month_filter}.")
        else:
            texts = df['Message'].tolist()
    else:
        texts = df['Message'].tolist()
    
    # Initialize the embedding model and build the FAISS index
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = compute_embeddings(texts, embedding_model)
    index = build_faiss_index(embeddings)
    
    # Initialize the QA pipeline (using a Hugging Face model)
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # Fine-tune the temporal model on synthetic data.
    print("\nGenerating synthetic temporal training data and fine-tuning T5...")
    temporal_examples = create_temporal_dataset(df)
    if temporal_examples:
        dataset = Dataset.from_dict({
            "input_text": [ex["input_text"] for ex in temporal_examples],
            "target_text": [ex["target_text"] for ex in temporal_examples]
        })
        model_name = "t5-small"
        temporal_tokenizer = T5Tokenizer.from_pretrained(model_name)
        temporal_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        def tokenize_function(example):
            model_inputs = temporal_tokenizer(example["input_text"], max_length=64, truncation=True)
            targets = temporal_tokenizer(example["target_text"], max_length=128, truncation=True)
            model_inputs["labels"] = targets["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(temporal_tokenizer, model=temporal_model)
        
        training_args = TrainingArguments(
            output_dir="./fine_tuned_temporal_model",
            num_train_epochs=10,
            per_device_train_batch_size=2,
            save_steps=10,
            save_total_limit=2,
            logging_steps=5,
            learning_rate=2e-5,
            weight_decay=0.01,
            report_to="none",
        )
        
        from transformers import Trainer
        trainer = Trainer(
            model=temporal_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        temporal_model.save_pretrained("./fine_tuned_temporal_model")
        temporal_tokenizer.save_pretrained("./fine_tuned_temporal_model")
        print("Fine-tuning complete. Temporal model saved.")
    else:
        temporal_model, temporal_tokenizer = None, None
    
    print("\nThe system is now ready to answer your questions based on the conversation data.")
    print("Type your question and press Enter (or type 'exit' to quit).")
    
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() == "exit":
            break
        final_answer = answer_question_temporal(
            user_question, df, embedding_model, index, texts, qa_pipeline,
            temporal_model=temporal_model, temporal_tokenizer=temporal_tokenizer, top_k=5
        )
        print(f"\nAnswer: {final_answer}")

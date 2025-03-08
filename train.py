import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load your CSV file into a DataFrame (assume a column 'message')

with open('received_texts.csv', 'r') as f:
    lines = f.read().splitlines()

df = pd.DataFrame(lines, columns=['message'])

# Create a dataset from the 'message' column
dataset = Dataset.from_pandas(df[['message']])
dataset = dataset.rename_column("message", "text")  # Ensure column is named 'text'

# Use a pre-trained tokenizer corresponding to your model, e.g., GPT-2
model_checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Set the pad token since GPT-2 doesn't have one by default
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=32  # adjust max_length as needed
    )
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

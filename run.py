# language: python
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import pipeline

# Load the fine-tuned model and tokenizer and use the mps device if available
generator = pipeline('text-generation',
                     model="./fine_tuned_model",
                     tokenizer="./fine_tuned_model",
                     device='mps')  # or device=0 if using cuda

instruction = "Talk like an old friend of mine "

# Generate text that mimics your texting style
while True:
    prompt = input("Enter a message: ")
    if prompt == "exit":
        break
    engineered_prompt = instruction + prompt
    generated = generator(engineered_prompt, truncation=True, num_return_sequences=1)
    generated_text = generated[0]['generated_text']
    if generated_text.startswith(instruction):
        generated_text = generated_text[len(instruction):].strip()
    print(generated_text)
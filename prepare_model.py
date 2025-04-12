from transformers import T5Tokenizer

# Load tokenizer from HuggingFace
tokenizer = T5Tokenizer.from_pretrained("t5-large")

# Save tokenizer files to your folder
tokenizer.save_pretrained("./t5_finetuned_qa_large")

print("Tokenizer files added successfully.")

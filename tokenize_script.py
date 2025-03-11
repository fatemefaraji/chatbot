import os
import json
import sentencepiece as spm
import numpy as np

# Paths
data_path = "C:/Users/admin/Videos/New folder/Week 4/data"
tokenizer_model_prefix = os.path.join(data_path, "tokenizer")
training_data_file = os.path.join(data_path, "chatbot_training_data.json")

# Load chatbot data
with open(training_data_file, "r", encoding="utf-8") as file:
    training_data = json.load(file)

# Extract texts for tokenizer training
text_corpus = [item["input"] for item in training_data] + [item["output"] for item in training_data]

# Save texts into a temporary file
corpus_file = os.path.join(data_path, "corpus.txt")
with open(corpus_file, "w", encoding="utf-8") as file:
    file.write("\n".join(text_corpus))

# Train SentencePiece tokenizer with a smaller vocab size
spm.SentencePieceTrainer.train(
    input=corpus_file, 
    model_prefix=tokenizer_model_prefix, 
    vocab_size=500  # Lower vocabulary size to avoid the error
)

# Load the trained tokenizer
sp = spm.SentencePieceProcessor(model_file=f"{tokenizer_model_prefix}.model")

# Extract vocabulary from SentencePiece model
vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

# Save vocabulary
vocab_file = os.path.join(data_path, "vocab.json")
with open(vocab_file, "w", encoding="utf-8") as file:
    json.dump(vocab, file, indent=4)

print(f"✅ Vocabulary saved: {vocab_file}")

# Convert text data into tokenized sequences
tokenized_data = [
    {
        "input_ids": sp.encode(item["input"], out_type=int),
        "output_ids": sp.encode(item["output"], out_type=int)
    }
    for item in training_data
]

# Save tokenized data
tokenized_data_file = os.path.join(data_path, "tokenized_training_data.json")
with open(tokenized_data_file, "w", encoding="utf-8") as file:
    json.dump(tokenized_data, file, indent=4)

print(f"✅ Tokenized data saved: {tokenized_data_file}")
print(tokenized_data[:5])  # نمایش 5 نمونه اول برای بررسی ساختار

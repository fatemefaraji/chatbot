import os
import json
import random  # Although random is imported, it's not actually used in this code

# Path to the processed data files
data_path = "c:/Users/admin/Videos/New folder/Week 4/data"

# Load the database, slots, and ontology
with open(os.path.join(data_path, "processed_database.json"), "r", encoding="utf-8") as file:
    database = json.load(file)

with open(os.path.join(data_path, "processed_slots.json"), "r", encoding="utf-8") as file:
    slots = json.load(file)

with open(os.path.join(data_path, "processed_ontology.json"), "r", encoding="utf-8") as file:
    ontology = json.load(file)

# List to store the chatbot training data (input-output pairs)
training_data = []

# Create synthetic dialogues for chatbot training
for domain, entries in database.items():
    if domain in slots:  # Check if the domain has associated slots
        for entry in entries:  # Iterate through each entry in the database for that domain
            for slot in slots[domain]:  # Iterate through each slot for that domain
                slot_key = slot.replace(domain + "-", "")  # Remove the domain prefix from the slot name

                if slot_key in entry:  # Check if the current entry has a value for the current slot
                    question = f"What is the {slot_key} of the {domain}?"  # Construct the question
                    answer = str(entry[slot_key])  # Get the corresponding answer from the database entry
                    
                    training_data.append({"input": question, "output": answer})  # Add the question-answer pair to the training data

# Save the training data in JSON format
output_file = os.path.join(data_path, "chatbot_training_data.json")
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(training_data, file, indent=4, ensure_ascii=False)

print(f"✅ Chatbot training data prepared and saved to '{output_file}'!")
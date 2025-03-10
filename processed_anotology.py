import os
import json

# Path to your data files (same as before)
data_path = "c:/Users/admin/Videos/New folder/Week 4/data"

# Process ontology values
ontology_file = os.path.join(data_path, "ontology.json")

with open(ontology_file, "r", encoding="utf-8") as file:
    ontology_data = json.load(file)

# Organize possible values for each slot
ontology_dict = {}

for slot, values in ontology_data.items():
    domain = slot.split("-")[0]  # Extract domain
    if domain not in ontology_dict:
        ontology_dict[domain] = {}
    
    ontology_dict[domain][slot] = values

# Save the processed ontology values
output_ontology_file = os.path.join(data_path, "processed_ontology.json")
with open(output_ontology_file, "w", encoding="utf-8") as file:
    json.dump(ontology_dict, file, indent=4, ensure_ascii=False)

print(f"✅ Ontology values processed and saved to '{output_ontology_file}'.")
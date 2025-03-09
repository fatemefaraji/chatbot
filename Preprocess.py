import os
import json

# Path to your data files
data_path = "c:/Users/admin/Videos/New folder/Week 4/data"

# List of your database files
dataset_files = [
    "police_db.json",
    "slot_descriptions.json",
    "ontology.json",
    "attraction_db.json",
    "hospital_db.json",
    "hotel_db.json",
    "restaurant_db.json",
    "taxi_db.json",
    "train_db.json"
]

# Dictionary to hold the processed data
structured_data = {}

def load_data(file_name):
    """Loads data from a JSON file"""
    with open(os.path.join(data_path, file_name), 'r', encoding="utf-8") as file:
        return json.load(file)

def extract_keys(d):
    """Extracts unique keys from a JSON structure (dicts and lists)"""
    keys = set()
    if isinstance(d, dict):
        for key, value in d.items():
            keys.add(key)
            keys.update(extract_keys(value))  # Recursively get keys from nested structures
    elif isinstance(d, list):
        for item in d:
            keys.update(extract_keys(item))  # Get keys from items within lists
    return keys

for file_name in dataset_files:
    try:
        data = load_data(file_name)
        keys = extract_keys(data)
        
        print(f"Keys in {file_name}:")
        print(keys)
        print("-" * 50)

        # Extract the domain name from the filename (e.g., "hotel" from "hotel_db.json")
        domain = file_name.replace("_db.json", "").replace(".json", "")

        # Store the processed data for database files only
        if "_db.json" in file_name:
            structured_data[domain] = data

    except json.JSONDecodeError as e:
        print(f"⚠️ Error decoding JSON from file {file_name}: {e}")
    except Exception as e:
        print(f"⚠️ An error occurred while processing file {file_name}: {e}")

# Save the processed database data to a new JSON file
output_file = os.path.join(data_path, "processed_database.json")
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(structured_data, file, indent=4, ensure_ascii=False)

print(f"✅ Processed database data saved to '{output_file}'.")
# Preprocess Script

This script preprocesses the raw database files and extracts unique keys from the JSON structures. It organizes the data by domain and saves the processed data into a single JSON file.

## Requirements

- Python 3.x
- JSON files: `police_db.json`, `slot_descriptions.json`, `ontology.json`, `attraction_db.json`, `hospital_db.json`, `hotel_db.json`, `restaurant_db.json`, `taxi_db.json`, `train_db.json`

## Usage

1. Ensure that the required JSON files are located in the specified `data_path` directory.
2. Run the script `Preprocess.py`.
3. The script will generate a `processed_database.json` file in the same directory, containing the structured data.

## Output

The script will print the unique keys found in each file and save the processed data:

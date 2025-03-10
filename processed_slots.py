import os
import json

# Path to your data files (same as before)
data_path = "c:/Users/admin/Videos/New folder/Week 4/data"

# Process conversational slots
slot_file = os.path.join(data_path, "slot_descriptions.json")

with open(slot_file, "r", encoding="utf-8") as file:
    slot_data = json.load(file)

# Organize slots by domain
slot_dict = {}

for slot in slot_data.keys():
    domain = slot.split("-")[0]  # Extract domain from slot name
    if domain not in slot_dict:
        slot_dict[domain] = []
    slot_dict[domain].append(slot)

# Save the processed slots
output_slot_file = os.path.join(data_path, "processed_slots.json")
with open(output_slot_file, "w", encoding="utf-8") as file:
    json.dump(slot_dict, file, indent=4, ensure_ascii=False)

print(f"✅ Conversational slots processed and saved to '{output_slot_file}'.")
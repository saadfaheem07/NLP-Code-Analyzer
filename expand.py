import json

def load_json(file_path):
    """Try to load a JSON file, handling JSON errors gracefully."""
    valid_entries = []
    skipped_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                valid_entries.append(data)
            except json.JSONDecodeError:
                skipped_count += 1
    print(f"Skipped {skipped_count} invalid entries.")
    return valid_entries

def save_json(data, file_path):
    """Save a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file)

def extract_valid_ast_entries(full_dataset, max_entries=1000):
    """Extract valid AST entries from the full dataset."""
    valid_entries = []
    for entry in full_dataset:
        if isinstance(entry, list) and len(valid_entries) < max_entries:
            valid_entries.append(entry)
    print(f"Extracted {len(valid_entries)} valid entries.")
    return valid_entries

def expand_dataset(original_file, output_file, max_entries=5000):
    """Expand the dataset by extracting more valid entries."""
    print(f"Loading original dataset from {original_file}...")
    original_data = load_json(original_file)
    
    print(f"Extracting up to {max_entries} valid entries...")
    new_valid_entries = extract_valid_ast_entries(original_data, max_entries=max_entries)
    
    print(f"Loading existing smaller dataset from {output_file}...")
    smaller_data = load_json(output_file)
    
    print(f"Appending {len(new_valid_entries)} new entries to smaller dataset...")
    expanded_dataset = smaller_data + new_valid_entries
    
    print(f"Saving expanded dataset to {output_file}...")
    save_json(expanded_dataset, output_file)

    print(f"Expanded dataset now contains {len(expanded_dataset)} entries.")

if __name__ == "__main__":
    # Path to the full original dataset and the smaller dataset
    full_dataset_path = 'data/python100k_train.json'  # Replace with your original dataset file
    smaller_dataset_path = 'data/smaller_dataset_train.json'  # Replace with your smaller dataset file

    # Expand the smaller dataset by adding 5000 more valid entries
    expand_dataset(full_dataset_path, smaller_dataset_path, max_entries=5000)

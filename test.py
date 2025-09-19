import json

def inspect_dataset(dataset, num_entries=5):
    """
    Print the keys of the first few entries in the dataset to understand its structure.
    """
    if not dataset:
        print("The dataset is empty or could not be loaded.")
        return

    print(f"Inspecting the first {min(num_entries, len(dataset))} entries:")
    
    for i, entry in enumerate(dataset[:num_entries]):
        if isinstance(entry, dict):
            print(f"Entry {i+1} keys: {list(entry.keys())}")
        else:
            print(f"Entry {i+1} is not a dictionary.")


def load_json(file_path):
    """Load the dataset from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"Successfully loaded {len(data)} entries from {file_path}")
            return data
    except Exception as e:
        print(f"Failed to load {file_path}: {str(e)}")
        return []

def save_json(data, file_path):
    """Save a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file)

def validate_entry(entry):
    """
    Validate entries based on new criteria.
    For now, let's check if 'type' and 'children' exist.
    """
    return isinstance(entry, dict) and 'type' in entry and 'children' in entry

# Function to expand and validate the dataset
def expand_dataset(original_file, smaller_file, max_entries=5000):
    # Load the original dataset and the smaller one
    original_data = load_json(original_file)
    smaller_data = load_json(smaller_file)

    print(f"Loaded {len(original_data)} entries from {original_file}")
    
    # Inspect the structure of a few entries to understand the keys
    inspect_dataset(original_data, num_entries=5)
    
    # Extract valid entries based on updated validation logic
    valid_entries = [entry for entry in original_data if validate_entry(entry)]
    
    print(f"Extracted {len(valid_entries)} valid entries from {original_file}")

    # Append to the smaller dataset
    smaller_data.extend(valid_entries[:max_entries])

    # Save the expanded dataset
    save_json(smaller_file, smaller_data)
    print(f"Expanded dataset now contains {len(smaller_data)} entries.")

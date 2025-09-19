import json

def load_original_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line)
                # If entry is a list, extend the data with the list's elements
                if isinstance(entry, list):
                    data.extend(entry)
                else:
                    data.append(entry)
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line[:100]}...")  # Print first 100 characters of invalid line
    return data

def preprocess_dataset(data):
    preprocessed_data = []
    non_dict_count = 0  # To count non-dictionary entries
    dict_count = 0  # To count valid dictionary entries
    
    for entry in data:
        if isinstance(entry, dict):
            # Only process dictionary entries
            dict_count += 1
            # Check if the dictionary has both 'children' and 'type'
            if 'children' in entry and 'type' in entry:
                features = entry['children']
                label = entry['type']
                preprocessed_data.append({
                    'features': features,
                    'label': label
                })
            else:
                print(f"Skipping dictionary entry without required keys: {entry}")
        else:
            non_dict_count += 1

    print(f"Total dictionary entries: {dict_count}")
    print(f"Total non-dictionary entries: {non_dict_count}")
    return preprocessed_data

def save_preprocessed_dataset(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    # Load the original dataset
    dataset_path = 'data/python100k_train.json'
    print("Loading original dataset...")
    data = load_original_dataset(dataset_path)

    # Preprocess the dataset
    print("Preprocessing dataset...")
    preprocessed_data = preprocess_dataset(data)

    # Save the preprocessed dataset
    output_path = 'data/preprocessed_python100k_train.json'
    print(f"Saving preprocessed data to {output_path}...")
    save_preprocessed_dataset(preprocessed_data, output_path)

    print(f"Preprocessing complete. Total valid entries: {len(preprocessed_data)}")

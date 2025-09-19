from some_module import load_preprocessed_dataset, train_model, evaluate_model, get_max_feature_length
from sklearn.model_selection import train_test_split
import json

def main():
    train_data_path = 'data/preprocessed_python100k_train.json'

    # Set a limit on the number of entries to load to prevent memory overuse
    dataset_limit = 20000  # Adjust this number based on memory constraints

    # Load the preprocessed dataset with a limit
    print("Loading preprocessed dataset with a limit of", dataset_limit, "entries...")
    full_data = load_preprocessed_dataset(train_data_path, limit=dataset_limit)

    # Split the dataset into training and evaluation sets
    print("Splitting the dataset into training and evaluation sets...")
    train_data, eval_data = train_test_split(full_data, test_size=0.2, random_state=42)

    print(f"Training dataset size: {len(train_data)}")
    print(f"Evaluation dataset size: {len(eval_data)}")

    # Determine the maximum feature length across both datasets
    target_length = get_max_feature_length(train_data, eval_data)

    # Train the model
    print("Training the model...")
    model, feature_importances = train_model(train_data, target_length)  # Pass target_length here
    print("Training complete.")

    # Evaluate the model
    print("Evaluating the model...")
    evaluation_results = evaluate_model(model, eval_data, target_length)  # Pass target_length here as well

    print(f"Evaluation Results: {evaluation_results}")

    # Save extracted values for plotting later
    data_for_plotting = {
        'y_true': evaluation_results['y_true'],
        'y_scores': evaluation_results['y_scores'],
        'feature_importances': feature_importances
    }

    with open('/content/plot_data.json', 'w') as f:
        json.dump(data_for_plotting, f)

    print("Data saved for plotting.")

if __name__ == '__main__':
    main()

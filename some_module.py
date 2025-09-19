import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to load the preprocessed dataset
def load_preprocessed_dataset(file_path, limit=None):
    print(f"Loading preprocessed dataset from {file_path}...")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Limit the dataset if a limit is provided
        if limit:
            data = data[:limit]
            print(f"Loaded {len(data)} entries from the dataset (limited to {limit} entries).")
        else:
            print(f"Loaded {len(data)} entries from the dataset.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

# Function to get the maximum feature length across datasets
def get_max_feature_length(train_data, eval_data):
    max_length = max(len(entry['features']) for entry in train_data + eval_data)
    print(f"Maximum feature length found: {max_length}")
    return max_length

# Function to train the model using GridSearchCV
def train_model(train_data, target_length):
    X_train = [entry['features'] + [0] * (target_length - len(entry['features'])) for entry in train_data]
    y_train = [entry['label'] for entry in train_data]

    # Define parameter grid for RandomForest
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Get feature importances from the trained model
    feature_importances = best_model.feature_importances_

    return best_model, feature_importances

# Function to evaluate the model
def evaluate_model(model, eval_data, target_length):
    X_eval = [entry['features'] + [0] * (target_length - len(entry['features'])) for entry in eval_data]
    y_eval = [entry['label'] for entry in eval_data]

    # Get the predicted probabilities instead of predicted labels
    y_scores = model.predict_proba(X_eval)[:, 1]  # Get probability for the positive class

    # Get the actual predictions (class labels)
    y_pred = model.predict(X_eval)

    # Calculate metrics
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='weighted')
    recall = recall_score(y_eval, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'y_true': y_eval,     # Actual true labels
        'y_scores': y_scores  # Predicted probabilities
    }

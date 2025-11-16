"""
Random Forest Classifier for Magnetite Genetic Type Classification
This script implements a Random Forest classifier to predict the genetic classes of magnetite samples based on their
elemental composition. This model is designed to distinguish igneous (I), hydrothermal (H), and sedimentary (S)
magnetite types.

Author: Qi-Tao Hu
Contact: qitaohu@mail.ustc.edu.cn
GitHub Repository: https://github.com/Qitaohu/Random-Forest-Classifier-for-Magnetite

Requirements:
- Python 3.7+
- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

Usage:
1. Modify the file path in the configuration section
2. Run the script to train the model and make predictions
3. Results will be saved as plots and spreedsheets

For faster training: If you want to reduce training time, you can modify the Gridsearch parameters to use the
best-performing hyperpatameters validated in our study: 'max_depth=20', 'max_features=3', 'min_samples_leaf=1',
'min_samples_split=2', 'n_estimators=200'. This will skip the exhaustive hyperparameter search and directly train with
these optimal values.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import self
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os
from typing import Counter

# =============================================================================
# CONFIGURATION SECTION - USER SHOULD MODIFY THESE PATHS
# =============================================================================
# Path to training data file (Centre-log-ratio transformed database)
# Expected format: Excel file with columns for 'type' and elemental concentrations
Traning_DATA_PATH = r"Training DATA CLR.xlsx"
# Path to prediction data file (new samples to classify)
PREDICTION_DATA_PATH = r"Example DATA CLR.xlsx"
# Output file names for results
FEATURE_IMPORTANCE_PLOT = 'feature_importances.svg'
CONFUSION_MATRIX_SVG = 'confusion_matrix_RF.svg'
CONFUSION_MATRIX_PDF = 'confusion_matrix_RF.pdf'
# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_training_data(file_path):
    """
    Load and preprocess training data for magnetite classification

    Parameters:
    file_path (str): Path to Excel file containing training data

    Returns:
    tuple: (X_features, y_labels, class_names)
    """
    print("Loading training data...")

    # Load data from Excel file
    data = pd.read_excel(file_path)

    # Select relevant features and target variable
    # Features: Mg, Al, Ti, V, Cr, Mn, Co, Mn, Co, Ni, Zn, Ga, Sn concentrations
    # Target: genetic type (H,I or S)
    df = data.loc[:, ["type", "Mg", "Al", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Zn", "Ga", "Sn"]]

    # Seperate features (X) and target (y)
    X = df.copy(deep=True)
    y = X.pop('type')

    # Display class distribution
    print("Class distribution in training data:")
    print(y.value_counts())

    # Convert string labels to integer codes
    y_int, class_names = pd.factorize(y, sort=True)
    y = y_int

    print(f"Classes mapped as: {list(class_names)} -> {list(range(len(class_names)))}")

    return X, y, class_names

def handle_class_imbalance(X_train, y_train):
    """
    Apply SMOTE to handle class imbalance in training data

    Parameters:
    X_train (array): Training features
    y_train (array): Training labels

    Returns:
    tuple: Resampled (X_train, y_train)
    """
    print("Applying SMOTE for class imbalance...")

    # Apply Synthetic Minority Over-sampleing Technique
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

    # print(f"Training data shape after SMOTE: {X_resampled.shape}")

    return X_resampled, y_resampled

# =============================================================================
# MODEL TRAINING AND HYPERPARAMETER TUNING
# =============================================================================
def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier with hyperparameter tuning

    Parameters:
    X_train (array): Training features
    y_train (array):Training lables

    Returns:
    tuple: (best_classifier, grid_search_results)
    """
    print("Training Random Forest classifier...")

    # Create pipeline with Random Forest
    pipeline = make_pipeline(
        RandomForestClassifier(
            class_weight="balanced",    # Handel class imbalance
            random_state=10,            # For reproducibility
            bootstrap=True,             # Use bootstrap sampling
            oob_score=True,             # Calculate out-of-bag score
        )
    )

    # Define hyperparameter grid for GridSearch
    param_grid = {
        "randomforestclassifier__n_estimators": [150, 200, 250],       # Number of trees
        "randomforestclassifier__max_depth": [16, 18, 20],                  # Maximum tree depth
        "randomforestclassifier__min_samples_split": [2, 5],                # Min samples to split node
        "randomforestclassifier__min_samples_leaf": [1, 2, 5],              # Min samples at leaf node
        "randomforestclassifier__max_features": [3, 4, 5]                   # Number of features to consider for split
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,                       # 5-fold cross-validation
        scoring="f1_macro",         # Use macro F1-score for evaluation
        n_jobs=-1,                  # Use all available CPUs
        refit=True,                 # Refit best model on entire training set
    )

    # Training the model
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search
def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):
    """
    Evaluate model performance on training and test sets.

    Parameters:
    model: Trained classifier
    X_train,y _train: Training data
    X_test, y_test: Test data
    class_names: Original class labels

    Returns:
    tuple:(y_train_pred, y_test_pred)
    """
    print("\n"+"="*50)
    print("MODEL EVALUATION")
    print("="*50)

    # Training set predictions
    y_train_pred = model.predict(X_train)
    print("Training Set Performance:")
    print(classification_report(y_train, y_train_pred, target_names=class_names))

    # Test set predictions
    y_test_pred = model.predict(X_test)
    print("Test Set Performance:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    return y_test_pred, y_test_pred

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_feature_importance(model, feature_names, output_file):
    """
    Create and save feature importance plot

    Parameters:
    model: Trained Random Forest model
    feature_names: List of feature names
    output_file: Output file path
    """
    # Extract feature importances from the model
    feature_importances = model.named_steps['randomforestclassifier'].feature_importances_

    # Sort features by importace
    indices = np.argsort(feature_importances)[::-1]

    # Set up plot style
    plt.rc('font', family='Arial', size=14)
    plt.figure(figsize=(10/2.54, 10/2.54))

    # Create bar plot
    plt.bar(range(len(feature_importances)),
            feature_importances[indices],
            color="grey", align="center")

    # Customize plot
    plt.xticks(range(len(feature_importances)),
               [feature_names[i] for i in indices],
               rotation=45, size=14)
    plt.tick_params(axis='both', direction='in', labelsize=14)
    plt.xlim([-1, len(feature_importances)])
    plt.tight_layout()

    # Save plot
    plt.savefig(output_file)
    plt.show()

    # Print feature importance values
    print("\nFeature Importances:")
    for i in indices:
        print(f"{feature_names[i]:>3}-{feature_importances[i]:.3f}")

def plot_confusion_matrix(y_true, y_pred, class_names, output_files):
    """
    Create and save confusion matrix visualization

    Parameters:
    y_true: True Tables
    y_pred: Predicted labels
    class_names: Original class names
    output_files: Tuple of (svg_file, pdf_file) paths
    """
    # Create confusion matrix
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        columns=class_names,
        index=class_names
    )

    # Set up plot style
    plt.rc('font', family='Arial', size=14)
    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54))

    sns.heatmap(cm, annot=True, fmt="d", linewidths=0.5, ax=ax, cmap="YlGn")
    ax.set_xlabel("Predictions", fontsize=18)
    ax.set_ylabel("True labels", fontsize=18)
    plt.tight_layout()

    # Save in multiple formats
    for output_file in output_files:
        plt.savefig(output_file, bbox_inches='tight')

    plt.show()

# =============================================================================
# PREDICTION ON NEW DATA
# =============================================================================
def predict_new_samples(model, data_file_path, class_names):
    """
    Predict genetic types for new magnetite samples.

    Parameters:
    model: Trained classifier
    data_file_path: Path to new data file
    class_names: Original class names

    Returns:
    DataFrame: Prediction results with probabilities
    """
    print("\n"+"="*50)
    print("PREDICTING NEW SAMPLES")
    print("="*50)

    # Load prediction data
    df = pd.read_excel(data_file_path)

    # Define required elements for prediction
    elements = ["Mg", "Al", "Ti", "V", "Cr", "Mn", "Co", "Ni", "Zn", "Ga", "Sn"]

    # Convert element columns to numeric, coerce error to NaN
    for element in elements:
        df[element] = pd.to_numeric(df[element], errors="coerce")

    # Select samples with complete data
    to_predict = df.loc[:, elements].dropna()
    to_predict.reset_index(drop=True, inplace=True)

    print(f"{to_predict.shape[0]} samples available for prediction")
    print("Summary statistics of prediction data:")
    print(to_predict.describe())

    if to_predict.shape[0] == 0:
        raise ValueError("No samples with complete feature data detected!")

    # Make predictions
    predictions = model.predict(to_predict)
    probabilities = model.predict_proba(to_predict)

    # Convert numeric predictions back to original class names
    predicted_classes = [class_names[pred] for pred in predictions]

    # Create results DataFrame
    results = pd.DataFrame({
        'pred_magnetite_type': predicted_classes,
        'H_proba': probabilities[:, 0],
        'I_proba': probabilities[:, 1],
        'S_proba': probabilities[:, 2]
    })

    # Display prediction summary
    prediction_counts = Counter(predicted_classes)
    print(f"\nPrediction summary: {dict(prediction_counts)}")

    return results, to_predict

def save_prediction_results(df, prediction_data, results, original_file_path):
    """
    Save prediction results to Excel file.

    Parameters:
    df: Original DataFrame
    prediction_data: Data used for prediction
    results: Prediction results
    original_file_path: Path to original data file

    Returns:
    str: Path to saved results file
    """
    # Generate output filename
    base_filename = os.path.basename(original_file_path)
    prefix, _ = os.path.splitext(base_filename)
    save_name = prefix + '_result.xlsx'

    # Merge results with original data using 'V' as key
    results_with_key = pd.concat([prediction_data['V'], results], axis=1)
    output = df.join(results_with_key.set_index('V'), on='V')

    # Save to Excel
    output.to_excel(save_name, index=False)
    print(f"Results saved to {save_name}")

    return save_name
# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Main function to excute the complete workflow.
    """
    print("Random Forest Classifier for Magnetite Genetic Type Classification")
    print("="*50)

    try:
        # Step 1: Load and preprocess training data
        X, y, class_names = load_training_data(Traning_DATA_PATH)

        # Step 2: Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )

        # Step 3: Handle class imbalance with SMOTE
        X_train, y_train = handle_class_imbalance(X_train, y_train)

        # Step 4: Perform initial cross-validation
        print("\nInitial Cross-validation Performance:")
        models = (RandomForestClassifier(),)
        for clf in models:
            scores = cross_val_score(clf, X_train, y_train, cv=5,
                                     scoring='f1_macro', n_jobs=-1)
            print(f'F1-score: {scores.mean():.2f} ± {scores.std():.2f}')

        # Step 5: Train final model with hyperparameter tuning
        best_model, grid_search = train_random_forest(X_train, y_train)

        # Display best hyperparameters
        print("\nBest Hyperparameters:")
        for param, value in grid_search.best_params_.items():
            param_name = param.split('__')[-1]
            print(f"{param_name:>20}: {value}")

        print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

        # Step 6: Evaluate model
        y_train_pred, y_test_pred = evaluate_model(
            best_model, X_train, y_train, X_test, y_test, class_names
        )

        # Step 7: Create visualizations
        plot_feature_importance(best_model, list(X.columns), FEATURE_IMPORTANCE_PLOT)
        plot_confusion_matrix(
            y_test, y_test_pred, class_names,
            (CONFUSION_MATRIX_SVG, CONFUSION_MATRIX_PDF)
        )

        # Step 8: Predict new samples
        results, prediction_data = predict_new_samples(
            grid_search, PREDICTION_DATA_PATH, class_names
        )

        # Display detailed results
        print("\nDetailed prediction results (first 10 samples):")
        pd.set_option('display.max_columns', 10)
        print(results.head(10))

        # Step 9: Save results
        save_prediction_results(
            pd.read_excel(PREDICTION_DATA_PATH),
            prediction_data,
            results,
            PREDICTION_DATA_PATH
        )

        print("\n"+"="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*50)
        print("Output files generated:")
        print(f"- {FEATURE_IMPORTANCE_PLOT} (feature importance plot)")
        print(f"- {CONFUSION_MATRIX_SVG} (confusion matrix)")
        print(f"- {CONFUSION_MATRIX_PDF} (confusion matrix, PDF format)")
        print(f"-[filename]_results.xlsx (prediction results)")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")
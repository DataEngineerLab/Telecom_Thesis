import pandas as pd
import numpy as np
import xgboost as xgb # Keep for potential future use or if you want to compare
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Install SHAP and LIME if you haven't already:
# pip install shap
# pip install lime

import shap # SHapley Additive exPlanations
import lime
import lime.lime_tabular # For tabular data with LIME

# --- 1. Load Your Actual Preprocessed Data from dataSetLast.data.csv ---
try:
    # Load the dataset as numpy array as done in RandomForest.py
    dataset = np.loadtxt('dataSetLast.data.new.csv', delimiter=',')
    print("Dataset 'dataSetLast.data.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'dataSetLast.data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Define features (x) and target (y) as per RandomForest.py
# Assuming x is from columns 1 to 14 (inclusive) and y is from column 16 (0-indexed 15)
# Ensure these indices match your actual data columns.
x = dataset[:1500, 1:15] # Features
y = dataset[:1500, 15].ravel() # Target, converted to 1D array

# Create generic feature names for interpretability tools since original data is numeric indices
# feature_names_original = [f'feature_{i}' for i in range(x.shape[1])]

feature_names_original = [
    'Subno', 'Device ID', 'Device_Model', 'Area', 'PSTN_ID_Age', 'Title',
    'Local_Call', 'Roaming_Call', 'AVG voice Call duration / cycle',
    'AVG data Call Duration / cycle', 'Nbr voice Call / cycle',
    'Nbr data Call / cycle', 'Call / cycle Type', 'Speed'
]

# Convert numpy arrays to pandas DataFrames for easier handling with SHAP/LIME
# This also helps in retaining feature names
X = pd.DataFrame(x, columns=feature_names_original)
y = pd.Series(y, name='Complaint')

# Ensure target variable is binary (0 or 1)
y = y.astype(int)

# Train/test split as per RandomForest.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y) # Added stratify for balanced splits

# Apply SMOTE as per RandomForest.py
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"Original Dataset shape: {X.shape}")
print(f"Training data shape after SMOTE: {X_train_sm.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Target distribution (y_train_sm): {y_train_sm.value_counts(normalize=True)}")
print(f"Target distribution (y_test): {y_test.value_counts(normalize=True)}")
print("-" * 50)

# --- 2. Train the Random Forest Model ---

# --- Hyperparameter Tuning for Random Forest to improve performance ---
# This step aims to find the best parameters for your model to increase prediction percentage for 'Complaint'
print("Starting Hyperparameter Tuning for Random Forest (this may take a while)...")
param_grid = {
    'n_estimators': [200, 500], # Reduced for faster execution, consider larger range [100, 200, 500, 1000]
    'max_depth': [10, 17, 25],  # Explore different depths
    'min_samples_split': [2, 6, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['log2', 'sqrt'] # Common values
}

# Use F1-score as the scoring metric, as it balances precision and recall,
# which is often crucial for imbalanced datasets like complaint prediction.
# 'recall' could also be a good choice if minimizing false negatives is paramount.
grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
                           param_grid=param_grid,
                           scoring='f1', # Optimize for F1-score
                           cv=3,         # Use 3-fold cross-validation for speed, consider 5 or 10 for more robustness
                           verbose=1,
                           n_jobs=-1)    # Use all available CPU cores

grid_search.fit(X_train_sm, y_train_sm)

model = grid_search.best_estimator_
print(f"Best Random Forest parameters found: {grid_search.best_params_}")
print("Random Forest model trained with best parameters.")


# Predict on the original (non-SMOTEd) test set
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# Print standard evaluation metrics as in RandomForest.py
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Complaint', 'Complaint']))
print("-" * 50)

# --- Adjusting Prediction Threshold to potentially increase 'Complaint' predictions ---
# By default, a probability >= 0.5 is classified as 'Complaint'.
# You can lower this threshold to get more 'Complaint' predictions (increase recall, potentially decrease precision).

# Example: Set a new threshold (e.g., 0.3 or 0.4)
new_threshold = 0.4 # Adjust this value based on your desired balance of precision/recall

y_pred_new_threshold = (y_scores >= new_threshold).astype(int)

print(f"\nEvaluating model with a new prediction threshold of {new_threshold}:")
new_accuracy = accuracy_score(y_test, y_pred_new_threshold)
new_precision = precision_score(y_test, y_pred_new_threshold)
new_recall = recall_score(y_test, y_pred_new_threshold)
new_f1 = f1_score(y_test, y_pred_new_threshold)

print(f"New Threshold Accuracy: {new_accuracy:.4f}")
print(f"New Threshold Precision: {new_precision:.4f}")
print(f"New Threshold Recall: {new_recall:.4f}")
print(f"New Threshold F1 Score: {new_f1:.4f}")
print("\nClassification Report (New Threshold):\n", classification_report(y_test, y_pred_new_threshold, target_names=['No Complaint', 'Complaint']))
print("-" * 50)


# --- 3. Implement SHAP for Model Interpretability ---
# SHAP provides "Shapley values" which quantify the contribution of each feature to the prediction.

# Initialize a SHAP explainer for tree-based models (RandomForestClassifier)
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set
# This might take some time for larger datasets
print("Calculating SHAP values (this may take a moment)...")
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values will be a list of two arrays (for class 0 and class 1).
# We usually focus on the SHAP values for the positive class (class 1 - 'Complaint').
shap_values_for_positive_class = shap_values[0]

print("SHAP values calculated.")

# --- 3.1. Visualize Global Feature Importance (SHAP Summary Plot) ---
# This plot shows the overall impact of each feature on the model's output.
# Each dot is an observation. Its position on the x-axis shows the SHAP value for that feature.
# Color indicates the feature value (red=high, blue=low).

print("\nGenerating SHAP Summary Plot (Global Feature Importance)...")
shap.summary_plot(shap_values_for_positive_class, X_test, feature_names=feature_names_original, show=False)
# If running in a local environment (e.g., Jupyter Notebook), you would typically use:
# import matplotlib.pyplot as plt
# plt.tight_layout()
# plt.show()
print("SHAP Summary Plot generated. (Check your plot output/viewer if running locally)")

# --- 3.2. Visualize Individual Prediction Explanation (SHAP Force Plot) ---
# This plot explains a single prediction, showing how each feature pushes the prediction
# from the base value (average prediction) to the model's output for that instance.

# Let's pick an arbitrary instance from the test set (e.g., the first one)
instance_index = 0
print(f"\nExplaining prediction for test instance {instance_index}...")
print(f"Actual label: {y_test.iloc[instance_index]}, Predicted label: {y_pred[instance_index]}")

# For force plot, you need the explainer's expected value (base value)
# and the SHAP values for that specific instance.
shap.initjs() # Initialize JavaScript for interactive plots (important for notebooks)
shap.force_plot(explainer.expected_value, shap_values_for_positive_class[instance_index,:], X_test.iloc[instance_index,:], feature_names=feature_names_original)
# Note: The force plot often works best in Jupyter notebooks due to its interactive nature.
# If you're not in a notebook, you might just see a static representation or need to save it.
print("SHAP Force Plot generated. (Check your plot output/viewer if running locally)")

# --- 3.3. Visualize Feature Dependence (SHAP Dependence Plot) ---
# Shows how the model output changes as a single feature changes,
# and how that interaction is influenced by another feature.

print("\nGenerating SHAP Dependence Plot (e.g., for feature_0)...")
# Pick a feature from feature_names_original to plot. 'feature_0' is just an example.
# You might want to analyze features that appear important in the summary plot.
# Example: shap.dependence_plot("feature_0", shap_values_for_positive_class, X_test, interaction_index="feature_1", show=False)
shap.dependence_plot("Area", shap_values_for_positive_class, X_test, interaction_index="Device_Model", show=False) # Simplified without interaction for general case
plt.tight_layout()
plt.show()
print("SHAP Dependence Plot generated. (Check your plot output/viewer if running locally)")
print("-" * 50)


# --- 4. Implement LIME for Model Interpretability ---
# LIME explains individual predictions by fitting a simple, interpretable model
# (like linear regression) around the neighborhood of the prediction.

# Initialize a LIME explainer for tabular data
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_sm.values, # Use SMOTEd training data for LIME background
    feature_names=feature_names_original, # Use the generic feature names
    class_names=['No Complaint', 'Complaint'], # Based on classification_report target names
    mode='classification'
)

print("LIME explainer initialized.")

# --- 4.1. Explain an Individual Prediction (LIME Explanation) ---

# Let's use the same instance as for SHAP (first instance in X_test)
instance_to_explain_lime = X_test.iloc[instance_index]

# Function to get probabilities from your model (LIME needs this)
predict_fn_lime = model.predict_proba

print(f"\nExplaining prediction for test instance {instance_index} using LIME...")
exp = explainer_lime.explain_instance(
    data_row=instance_to_explain_lime.values,
    predict_fn=predict_fn_lime,
    num_features=10 # Number of top features to show in the explanation
)

# Print the explanation as a list of (feature, weight) tuples
print("\nLIME Explanation (Feature weights for this specific prediction):")
print(exp.as_list())

# --- 4.2. Visualize LIME Explanation ---
# LIME can generate an HTML plot for interactive viewing.

print("\nGenerating LIME Explanation Plot (HTML)...")
# The explanation will be displayed in your output if in a notebook, or can be saved to HTML.
# exp.show_in_notebook(show_table=True, show_all=False) # For Jupyter notebooks
exp.save_to_file('lime_explanation_instance.html')
print("LIME explanation saved to lime_explanation_instance.html")
print("LIME Explanation Plot generated. (Check your plot output/viewer or saved HTML file)")
print("-" * 50)

print("\nInterpretability analysis complete. Review the generated plots and printed explanations.")
print("Remember to replace the sample data and model with your actual thesis components.")

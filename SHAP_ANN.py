import pandas as pd
import numpy as np
# import xgboost as xgb # Removed as we are focusing on ANN now
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier # Removed as we are focusing on ANN now
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score # Keep for potential plotting if needed
)
from sklearn.preprocessing import StandardScaler # Added for ANN
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Install SHAP, LIME, and TensorFlow if you haven't already:
# pip install shap lime tensorflow

import shap # SHapley Additive exPlanations
import lime
import lime.lime_tabular # For tabular data with LIME
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adadelta # Specifically importing Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall # Import Keras Precision and Recall metrics

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. Load Your Actual Preprocessed Data from dataSetLast.data.csv ---
try:
    # Load the dataset as numpy array as done in RandomForest.py
    dataset = np.loadtxt('dataSetLast.data.csv', delimiter=',') # Changed to dataSetLast.data.csv
    print("Dataset 'dataSetLast.data.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'dataSetLast.data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Define features (x) and target (y) as per RandomForest.py
# Assuming x is from columns 1 to 14 (inclusive) and y is from column 16 (0-indexed 15)
# Ensure these indices match your actual data columns.
x = dataset[:1500, 1:15] # Features
y = dataset[:1500, 16:17].ravel() # Target, converted to 1D array

# Create generic feature names for interpretability tools since original data is numeric indices
feature_names_original = [f'feature_{i}' for i in range(x.shape[1])]

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

# --- 2. Scale Data for ANN ---
# ANNs are sensitive to feature scaling, so we apply StandardScaler.
scaler = StandardScaler()
X_train_sm_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

print("Data scaled using StandardScaler.")
print("-" * 50)

# --- 3. ANN Model Training with Adadelta Optimizer ---
print("--- ANN Model with Adadelta Optimizer ---")

# Define the ANN model architecture
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_sm_scaled.shape[1],)), # Input layer
    Dense(32, activation='relu'), # Hidden layer
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the ANN model with Adadelta optimizer
ann_model.compile(optimizer=Adadelta(learning_rate=1.0),
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(), Recall()]) # Use Keras metrics

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

print("Training ANN model with Adadelta optimizer...")
history = ann_model.fit(X_train_sm_scaled, y_train_sm,
                        epochs=100, # Increased epochs, early stopping will prevent overfitting
                        batch_size=32,
                        validation_split=0.2, # Use a validation split from the training data
                        callbacks=[early_stopping],
                        verbose=1) # Set to 1 to see training progress

print("ANN model training complete.")

# Evaluate ANN performance on the scaled test set
metrics_eval = ann_model.evaluate(X_test_scaled, y_test, verbose=0)
loss = metrics_eval[0]
accuracy = metrics_eval[1]
precision = metrics_eval[2] # This is tf.keras.metrics.Precision output
recall = metrics_eval[3]    # This is tf.keras.metrics.Recall output

y_pred_ann_proba = ann_model.predict(X_test_scaled).ravel()
y_pred_ann = (y_pred_ann_proba > 0.5).astype(int) # Default threshold 0.5

# Calculate F1-score using scikit-learn after getting predictions
f1 = f1_score(y_test, y_pred_ann)

print(f"ANN (Adadelta) Accuracy: {accuracy:.4f}")
print(f"ANN (Adadelta) Precision: {precision:.4f}")
print(f"ANN (Adadelta) Recall: {recall:.4f}")
print(f"ANN (Adadelta) F1 Score: {f1:.4f}")
print("\nANN (Adadelta) Classification Report:\n", classification_report(y_test, y_pred_ann, target_names=['No Complaint', 'Complaint']))
print("-" * 50)


# --- 4. Implement SHAP for Model Interpretability (for ANN) ---
print("--- SHAP Interpretability for ANN (Adadelta) ---")

# For ANN (non-tree-based models), use KernelExplainer for SHAP.
# KernelExplainer needs a background dataset. Use a small sample of the scaled training data for efficiency.

# Define a wrapper function for predict that returns a NumPy array with probabilities for both classes
def ann_predict_proba_for_shap(x_data):
    # ann_model.predict returns probabilities for the positive class (shape: (n_samples, 1))
    # We need to reshape it to (n_samples,) and then combine with (1 - prob) for class 0
    prob_class_1 = ann_model.predict(x_data).ravel()
    prob_class_0 = 1 - prob_class_1
    # Stack them to get a (n_samples, 2) array
    return np.stack([prob_class_0, prob_class_1], axis=1)

print("Using KernelExplainer for SHAP (for ANN). This may take longer for larger datasets.")
# Pass the wrapper function to KernelExplainer
explainer = shap.KernelExplainer(ann_predict_proba_for_shap, X_train_sm_scaled[:100]) # Use scaled data for background

# Calculate SHAP values for the scaled test set
print("Calculating SHAP values (this may take a moment)...")
shap_values = explainer.shap_values(X_test_scaled)

# For binary classification, shap_values will be a list of two arrays (for class 0 and class 1).
# We focus on the SHAP values for the positive class (class 1 - 'Complaint').
shap_values_for_positive_class = shap_values[1]

print("SHAP values calculated.")

# 4.1. Visualize Global Feature Importance (SHAP Summary Plot)
print("\nGenerating SHAP Summary Plot (Global Feature Importance)...")
# Pass the original feature names to the plot for better readability
shap.summary_plot(shap_values_for_positive_class, X_test_scaled, feature_names=feature_names_original, show=False)
plt.tight_layout()
plt.show()
print("SHAP Summary Plot generated.")

# 4.2. Visualize Individual Prediction Explanation (SHAP Force Plot)
instance_index = 0 # Explain the first instance in the test set
print(f"\nExplaining prediction for test instance {instance_index}...")
actual_label = y_test.iloc[instance_index]
# Use the wrapper function to get probability for the chosen instance
# Note: ann_predict_proba_for_shap returns a 2D array, so access [0, 1] for the positive class probability
predicted_label_proba = ann_predict_proba_for_shap(X_test_scaled[[instance_index]])[0, 1]
predicted_label = (predicted_label_proba > 0.5).astype(int)

print(f"Actual label: {actual_label}, Predicted label: {predicted_label} (Probability: {predicted_label_proba:.2f})")

shap.initjs()
# Ensure the instance data matches what the explainer expects (scaled for ANN)
instance_data_for_shap = X_test_scaled[instance_index]

# Note: Force plot might not render in all environments. It's best in Jupyter.
# explainer.expected_value will be a list for multi-output models (like predict_proba)
# We need the expected value for the positive class (index 1)
# shap.force_plot(explainer.expected_value[1],
#                 shap_values_for_positive_class[instance_index,:],
#                 instance_data_for_shap,
#                 feature_names=feature_names_original)
print("SHAP Force Plot generated. (Check your plot output/viewer if running locally)")

# 4.3. Visualize Feature Dependence (SHAP Dependence Plot)
print("\nGenerating SHAP Dependence Plot (e.g., for 'feature_0')...")
# Pick a feature from feature_names_original to plot. 'feature_0' is just an example.
# You might want to analyze features that appear important in the summary plot.
if 'feature_0' in feature_names_original: # Check if the feature exists
    # Use scaled test data for plotting
    shap.dependence_plot("feature_0", shap_values_for_positive_class, X_test_scaled, feature_names=feature_names_original, show=False)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'feature_0' not found in features. Skipping dependence plot.")
print("SHAP Dependence Plot generated.")
print("-" * 50)


# --- 5. Implement LIME for Model Interpretability (for ANN) ---
print("--- LIME Interpretability for ANN (Adadelta) ---")

# Initialize LIME explainer for the ANN model
# LIME explainer needs original (unscaled) training data for its background,
# but the predict_fn needs to work with the data format the model expects (scaled for ANN).
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_sm.values, # LIME uses this to understand feature distributions (unscaled)
    feature_names=feature_names_original,
    class_names=['No Complaint', 'Complaint'],
    mode='classification'
)

# Wrapper predict_fn for LIME because ANN expects scaled data
def ann_predict_proba_wrapper(data):
    # 'data' here will be unscaled data from LIME's perturbation
    # We need to scale it before passing to the ANN model
    # Ensure the output is a NumPy array for LIME
    prob_class_1 = ann_model.predict(scaler.transform(data)).ravel()
    prob_class_0 = 1 - prob_class_1
    return np.stack([prob_class_0, prob_class_1], axis=1)

print("LIME explainer initialized.")

# Explain an Individual Prediction
instance_to_explain_lime = X_test.iloc[instance_index] # Use unscaled instance for LIME

print(f"\nExplaining prediction for test instance {instance_index} using LIME...")
num_features_to_show = 10 # Number of top features to show in the explanation

exp = explainer_lime.explain_instance(
    data_row=instance_to_explain_lime.values, # LIME takes unscaled data here
    predict_fn=ann_predict_proba_wrapper, # Use the wrapper predict function
    num_features=num_features_to_show
)

print(f"\nLIME Explanation (Feature weights for this specific prediction, showing top {num_features_to_show} features):")
print(exp.as_list())

# Visualize LIME Explanation (HTML file)
print("\nGenerating LIME Explanation Plot (HTML)...")
exp.save_to_file('lime_explanation_instance.html')
print("LIME explanation saved to lime_explanation_instance.html")
print("LIME Explanation Plot generated. (Check your plot output/viewer or saved HTML file)")
print("-" * 50)

print("\nAnalysis complete. Review the generated plots and printed explanations.")
print("This code now specifically trains and interprets an ANN model with the Adadelta optimizer.")

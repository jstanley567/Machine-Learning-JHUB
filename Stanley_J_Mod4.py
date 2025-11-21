# ===========================================
# BREAST CANCER RECURRENCE PREDICTION
# Challenge B & C - Alternative Implementation
# ===========================================

print("Initialising environment...")

import os
import warnings
warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

# Neural network / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Colab helpers
from google.colab import drive, files
import shutil

# -------------------------------------------------------------------
# CONSTANTS / GLOBALS
# -------------------------------------------------------------------

CORRUPTED_TOKENS = ['41913', '43588', '43683', '43778', '43713', '41974', '?', '�']

DEFAULT_SEARCH_PATHS = [
    '/content/drive/MyDrive/breast_cancer.xls',
    '/content/drive/MyDrive/breast-cancer.xls',
    '/content/drive/MyDrive/breast-cancer (2).xls'
]

NEW_PATIENT_RAW = {
    'age': ['50-59'],
    'menopause': ['premeno'],
    'tumor-size': ['20-24'],
    'inv-nodes': ['0-2'],
    'node-caps': ['no'],
    'deg-malig': [2],
    'breast': ['left'],
    'breast-quad': ['left_low'],
    'irradiat': ['yes']
}

# -------------------------------------------------------------------
# FILE HANDLING
# -------------------------------------------------------------------

def mount_and_locate_file():
    """
    Mount Google Drive, search for expected files, and if not present,
    prompt the user to upload. Returns absolute path to the dataset.
    """
    print("\n" + "=" * 60)
    print("STEP 0: DATA ACCESS SETUP")
    print("=" * 60)

    print("Mounting Google Drive at /content/drive ...")
    drive.mount('/content/drive', force_remount=False)

    dataset_path = None
    print("\nSearching for dataset in Google Drive...")
    for candidate in DEFAULT_SEARCH_PATHS:
        if os.path.exists(candidate):
            dataset_path = candidate
            print(f"  ✓ Found dataset: {candidate}")
            break

    if dataset_path is None:
        print("\nNo known dataset file found.")
        print("Please upload the breast cancer .xls file from your computer.")
        uploaded_files = files.upload()
        if uploaded_files:
            uploaded_name = list(uploaded_files.keys())[0]
            # Move uploaded file into MyDrive and use that path
            src = f"/content/{uploaded_name}"
            dst = f"/content/drive/MyDrive/{uploaded_name}"
            shutil.move(src, dst)
            dataset_path = dst
            print(f"  ✓ File uploaded and moved to: {dataset_path}")
        else:
            raise FileNotFoundError("No file was uploaded. Cannot proceed.")

    return dataset_path

# -------------------------------------------------------------------
# DATA LOADING & CLEANING
# -------------------------------------------------------------------

def read_and_clean_dataset(path_to_file):
    """
    Load the Excel file, remove known corrupted tokens, drop rows with NaNs.
    """
    print("\n" + "=" * 60)
    print("STEP 1: LOADING & CLEANING DATA")
    print("=" * 60)

    data = pd.read_excel(path_to_file)
    print(f"Original dataset shape: {data.shape}")
    print(f"Replacing corrupted values: {CORRUPTED_TOKENS}")

    # Replace corrupted tokens only in object (string) columns
    for col_name in data.select_dtypes(include=['object']).columns:
        data[col_name] = data[col_name].replace(CORRUPTED_TOKENS, np.nan)

    cleaned = data.dropna()
    removed_count = data.shape[0] - cleaned.shape[0]
    print(f"Cleaned dataset shape: {cleaned.shape}")
    print(f"Rows removed due to corrupted / missing data: {removed_count}")

    return cleaned

# -------------------------------------------------------------------
# ENCODING & SPLITTING
# -------------------------------------------------------------------

def encode_features_and_target(df):
    """
    Encode all categorical predictors and the target (Class) column.
    """
    print("\n" + "=" * 60)
    print("STEP 2: ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)

    features = df.drop(columns=['Class'])
    labels = df['Class']

    encoders_map = {}

    # Encode non-numeric (object) columns
    categorical_cols = features.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col].astype(str))
        encoders_map[col] = encoder
        print(f"  ✓ Encoded {col}: {len(encoder.classes_)} categories")

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(labels)
    class_map = {idx: cls for idx, cls in enumerate(target_encoder.classes_)}
    print(f"  ✓ Encoded target 'Class': {class_map}")

    return features, y_encoded, encoders_map, target_encoder

def make_train_val_test_split(X, y):
    """
    Create train, validation, and test splits with same proportions as original script.
    """
    print("\n" + "=" * 60)
    print("STEP 3: CREATING TRAIN / VAL / TEST SPLITS")
    print("=" * 60)

    # 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 20% of remaining for validation (i.e. 16% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full
    )

    print(f"  Training set size:    {X_train.shape[0]} samples")
    print(f"  Validation set size:  {X_val.shape[0]} samples")
    print(f"  Test set size:        {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test

# -------------------------------------------------------------------
# CHALLENGE B - RANDOM FOREST
# -------------------------------------------------------------------

def fit_random_forest_with_grid_search(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning on a RandomForestClassifier using GridSearchCV.
    """
    print("\n" + "🔍 CHALLENGE B: MACHINE LEARNING MODEL".center(60, "="))

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }

    rf_base = RandomForestClassifier(random_state=42)

    print("  Running GridSearchCV for Random Forest (scoring: roc_auc, 5-fold CV)...")
    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print(f"\n  ✓ Best hyperparameters: {grid.best_params_}")

    # Validation AUC
    val_probs = best_rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"  ✓ Validation AUC (Random Forest): {val_auc:.4f}")

    return best_rf

def run_stratified_cv(model, X, y):
    """
    Perform 10-fold stratified cross-validation using roc_auc scoring.
    """
    print("\n" + "=" * 60)
    print("STEP 4: 10-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1
    )

    scores_str = [f"{s:.4f}" for s in cv_scores]
    print(f"  Individual AUC scores: {scores_str}")

    mean_auc = cv_scores.mean()
    spread = cv_scores.std() * 2
    print(f"  Mean AUC: {mean_auc:.4f} (+/- {spread:.4f})")

    return cv_scores

def evaluate_on_test_set(model, X_test, y_test, target_le, label="Model"):
    """
    Compute accuracy, AUC and print classification report on the test set.
    """
    print("\n" + "FINAL TEST EVALUATION".center(60, "-"))

    y_hat = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_prob)

    print(f"  {label} Accuracy: {acc:.4f}")
    print(f"  {label} AUC:       {auc:.4f}\n")

    print("  Classification Report:")
    print(classification_report(y_test, y_hat, target_names=target_le.classes_))

    return acc, auc

# -------------------------------------------------------------------
# CHALLENGE C - NEURAL NETWORK
# -------------------------------------------------------------------

def build_and_tune_neural_network(X_train, y_train, X_val, y_val):
    """
    Train several neural networks with different learning rates and keep the best by validation AUC.
    """
    print("\n" + "🧠 CHALLENGE C: NEURAL NETWORK WITH LEARNING RATE TUNING".center(60, "="))

    candidate_lrs = [0.01, 0.001, 0.0001]
    best_auc = 0.0
    best_nn = None

    for lr in candidate_lrs:
        print(f"\n  Testing learning rate: {lr}")

        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )

        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=0
        )

        print("    Training neural network...")
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
        print(f"    Validation AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_nn = model

    print(f"\n  ✅ Best NN validation AUC: {best_auc:.4f}")
    return best_nn

# -------------------------------------------------------------------
# UNSEEN EXAMPLE PREDICTION
# -------------------------------------------------------------------

def predict_single_patient(model, encoders_map, target_le, model_kind="ML"):
    """
    Use either the RF model ('ML') or NN ('NN') to predict an unseen example.
    """
    print("\n" + "UNSEEN EXAMPLE PREDICTION".center(60, "-"))

    new_patient_df = pd.DataFrame(NEW_PATIENT_RAW)

    print("\n  Input patient record:")
    print(new_patient_df.to_string(index=False))

    # Apply the same label encoders used in training
    for col in new_patient_df.columns:
        if col in encoders_map:
            new_patient_df[col] = encoders_map[col].transform(
                new_patient_df[col].astype(str)
            )

    if model_kind == "ML":
        # Random Forest path
        predicted_class_num = model.predict(new_patient_df)[0]
        predicted_probs = model.predict_proba(new_patient_df)[0]
        predicted_label = target_le.inverse_transform([predicted_class_num])[0]
        confidence_pct = predicted_probs[predicted_class_num] * 100.0
    else:
        # Neural network path
        raw_output = model.predict(new_patient_df, verbose=0)[0][0]
        predicted_class_num = int(raw_output > 0.5)
        predicted_label = target_le.inverse_transform([predicted_class_num])[0]
        confidence_pct = float(raw_output) * 100.0

    print(f"\n  🔮 PREDICTION: {predicted_label}")
    print(f"  📊 CONFIDENCE: {confidence_pct:.2f}%")

    return predicted_label

# -------------------------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------------------------

def run_full_pipeline():
    print("=" * 60)
    print("🚀 BREAST CANCER RECURRENCE PREDICTION WORKFLOW")
    print("=" * 60)

    # 0. Data access (Drive / upload)
    data_path = mount_and_locate_file()

    # 1. Load & clean
    df_clean = read_and_clean_dataset(data_path)

    # 2. Encode
    X, y, encoders_map, target_le = encode_features_and_target(df_clean)

    # 3. Splits
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(X, y)

    # 4. Challenge B - Machine Learning (Random Forest)
    print("\n" + "🔍 CHALLENGE B: MACHINE LEARNING".center(60, "="))
    rf_best = fit_random_forest_with_grid_search(X_train, y_train, X_val, y_val)

    # 5. Cross-validation (10-fold)
    run_stratified_cv(rf_best, X, y)

    # 6. Test set evaluation for RF
    evaluate_on_test_set(rf_best, X_test, y_test, target_le, label="Random Forest")

    # 7. Predict unseen example with RF
    predict_single_patient(rf_best, encoders_map, target_le, model_kind="ML")

    # 8. Challenge C - Neural Network
    print("\n" + "🧠 CHALLENGE C: NEURAL NETWORK".center(60, "="))
    nn_best = build_and_tune_neural_network(X_train, y_train, X_val, y_val)

    # 9. Evaluate NN on test set
    print("\n" + "NEURAL NETWORK TEST EVALUATION".center(60, "-"))
    nn_test_loss, nn_test_acc, nn_test_auc = nn_best.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss:     {nn_test_loss:.4f}")
    print(f"  Accuracy: {nn_test_acc:.4f}")
    print(f"  AUC:      {nn_test_auc:.4f}")

    # 10. Predict unseen example with NN
    predict_single_patient(nn_best, encoders_map, target_le, model_kind="NN")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE - READY FOR SCREENSHOTS & SUBMISSION")
    print("=" * 60)

# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    run_full_pipeline()

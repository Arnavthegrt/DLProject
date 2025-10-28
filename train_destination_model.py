import pandas as pd
import numpy as np
# Matplotlib and Seaborn are removed as we are not plotting in this script
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # --- 1. Dataset Loading ---
    try:
        # Update this path to where your file is located
        df = pd.read_csv("travel_preferences_augmented.csv")
    except FileNotFoundError:
        print("Error: 'travel_preferences_augmented.csv' not found.")
        print("Please make sure the file is in the same directory or update the file path.")
        return # Exit the script if file not found

    print("--- Data Loaded Successfully ---")

    # --- 2. Feature Engineering ---
    print("--- Starting Feature Engineering ---")

    # 2.1. Handle 'interests' column
    if 'interests' in df.columns:
        interest_dummies = df['interests'].str.get_dummies(sep='|')
        interest_dummies.columns = [f'interest_{col}' for col in interest_dummies.columns]
        df = pd.concat([df, interest_dummies], axis=1)
        df = df.drop('interests', axis=1)

    # 2.2. Handle 'start_date' column
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['start_month'] = df['start_date'].dt.month
        df = df.drop('start_date', axis=1)

    print("--- Feature Engineering Complete ---")

    # --- 3. Define Features (X) and Target (y) ---
    X = df.drop("destination", axis=1)
    y = df["destination"]

    # --- 4. Split Data BEFORE Preprocessing ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # --- 5. Preprocessing with ColumnTransformer ---
    
    # Identify column types from the feature-engineered X_train
    interest_features = [col for col in X_train.columns if col.startswith('interest_')]
    
    numeric_features = ['duration_days', 'num_people', 'budget_per_person_inr', 'start_month']
    numeric_features.extend(interest_features)

    categorical_features = ['departure_city', 'preferred_pace', 'accommodation_style']
    
    # Create the preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 6. Apply Preprocessing ---
    print("--- Preprocessing Data ---")
    # Fit the preprocessor ONLY on the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)

    # Convert from sparse matrix to dense array if necessary
    if not isinstance(X_train_processed, np.ndarray):
        X_train_processed = X_train_processed.toarray()
    if not isinstance(X_test_processed, np.ndarray):
        X_test_processed = X_test_processed.toarray()

    # --- 7. Encode Labels ---
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} destination classes.")
    # --- > SAVE THE LABEL ENCODER < ---
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("Label encoder saved to 'label_encoder.joblib'")
    # --- > SAVE THE CLASS NAMES < ---
    # Save the actual destination names in the correct order
    np.save('destination_classes.npy', label_encoder.classes_)
    print("Destination class names saved to 'destination_classes.npy'")

    # --- 6. Apply Preprocessing ---
    print("--- Preprocessing Data ---")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    # --- > SAVE THE PREPROCESSOR < ---
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Preprocessor saved to 'preprocessor.joblib'")

    # --- 8. Build and Train Model ---
    model = Sequential([
        Dense(64, activation='relu',
              input_shape=(X_train_processed.shape[1],),
              kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary() # Print model summary

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train_processed, y_train_encoded,
        epochs=150,
        validation_split=0.2, # Use 20% of training data for validation
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2 # Use verbose=2 for cleaner log output
    )

    # --- 9. Evaluate Model ---
    print("\n--- Evaluating on Test Set (using best weights) ---")
    loss, accuracy = model.evaluate(X_test_processed, y_test_encoded, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Detailed Classification Report
    y_pred_probs = model.predict(X_test_processed)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- Classification Report ---")
    # Get original class names for the report
    target_names = label_encoder.classes_
    print(classification_report(y_test_encoded, y_pred, target_names=target_names))
    
   # --- 10. (Optional) Save the Model ---
    model.save('destination_predictor_model.h5')
    print("\nModel saved to 'destination_predictor_model.h5'")


# This makes the script runnable from the command line
if __name__ == "__main__":
    main()

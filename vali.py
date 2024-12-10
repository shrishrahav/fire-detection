import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to dataset
data_dir = r'C:\Users\SHRISHRAHAV\Downloads\dataset'

# Map class indices
label_map = {0: 'no_fire', 1: 'fire'}

# Generate a list of file paths and labels
data_files = []
data_labels = []

for label, class_name in label_map.items():
    class_dir = os.path.join(data_dir, class_name)
    if os.path.exists(class_dir):
        for file_name in os.listdir(class_dir):
            data_files.append(os.path.join(class_dir, file_name))
            data_labels.append(label)

# Convert file paths and labels to numpy arrays
data_files = np.array(data_files)
data_labels = np.array(data_labels)

# Initialize the data generator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_index, val_index) in enumerate(kfold.split(data_files, data_labels), 1):
    print(f"\n--- Fold {fold} ---")
    
    # Split data into training and validation sets
    train_files, val_files = data_files[train_index], data_files[val_index]
    train_labels, val_labels = data_labels[train_index], data_labels[val_index]
    
    # Convert numeric labels to string labels for flow_from_dataframe
    train_labels = np.array([label_map[label] for label in train_labels])
    val_labels = np.array([label_map[label] for label in val_labels])
    
    # Create DataFrames for train and validation data
    train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
    val_df = pd.DataFrame({'filename': val_files, 'class': val_labels})
    
    # Data generators
    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )
    val_generator = datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary'
    )
    
    # Define the model
    base_model = VGG16(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(train_generator, epochs=5, validation_data=val_generator, verbose=1)
    
    # Evaluate on validation data
    val_generator.reset()  # Ensure correct alignment
    val_predictions = (model.predict(val_generator) > 0.5).astype(int).flatten()
    val_true_labels = val_generator.classes
    
    # Calculate AUC-ROC
    auc = roc_auc_score(val_true_labels, val_predictions)
    auc_scores.append(auc)
    print(f"Fold {fold} AUC-ROC: {auc:.2f}")

# Average AUC-ROC across all folds
print(f"\nAverage AUC-ROC: {np.mean(auc_scores):.2f}")

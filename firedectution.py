import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Paths to dataset folders
train_dir = r'C:\Users\SHRISHRAHAV\Downloads\dataset'
test_dir = r'C:\Users\SHRISHRAHAV\Downloads\Fire_Dataset'

# Verify class labels in dataset folders
classes = os.listdir(train_dir)
print(f"Classes in training set: {classes}")

# Data Generators for Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split for validation
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Train and Validation Data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test Data
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights}")

# CNN Model Definition
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model with Class Weights
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    class_weight=class_weights
)

# Save the Model


# Evaluate Model on Test Data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate Predictions and Classification Report
test_data.reset()
predictions = (model.predict(test_data) > 0.5).astype("int32")
true_labels = test_data.classes

print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=list(test_data.class_indices.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predictions))
model.save("fire_detection_model.keras")  # Save in HDF5 format

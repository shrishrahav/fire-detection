from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_data = train_datagen.flow_from_directory(
    'path/to/train/dataset',  # e.g., C:\path\to\dataset
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'  # Binary classification
)

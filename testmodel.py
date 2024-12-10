from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model("fire_detection_model.keras")
img_path = r"C:\Users\SHRISHRAHAV\Downloads\dataset\non_fire_images\non_fire.99.png"
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
print("Prediction:", "Fire" if prediction[0][0] > 0.5 else "No Fire")

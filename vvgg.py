from tensorflow.keras.applications import MobileNet

base_model = MobileNet(input_shape=(64, 64, 3), include_top=False, weights='imagenet')

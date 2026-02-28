import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os


MODEL_PATH = "solar_fault_model_saved"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Bird_drop", "Clean", "Dusty", "Electrical-Damage",
               "Physical-Damage", "Snow-Covered"]


model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")


img_path = sys.argv[1]   
if not os.path.exists(img_path):
    print("Image not found!")
    sys.exit()

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)


predictions = model.predict(img_array)
predicted_class = CLASS_NAMES[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print("\nPrediction:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))
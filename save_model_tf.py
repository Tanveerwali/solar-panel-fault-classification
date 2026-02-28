import tensorflow as tf

# Load your existing keras model (weights + architecture)
model = tf.keras.models.load_model("solar_fault_model.keras", compile=False)

# Save it in TensorFlow SavedModel format
model.save("solar_fault_model_saved", save_format="tf")

print("Model saved successfully in 'solar_fault_model_saved' folder")
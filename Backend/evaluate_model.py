import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# Path to your dataset
DATASET_DIR = r'C:\Users\Yashashwini\Downloads\Medicinal plant dataset'
IMG_SIZE = 224
BATCH_SIZE = 32

# Data generator for validation set
val_datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)
val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class names
class_names = list(val_gen.class_indices.keys())
print("Class names:", class_names)

# Load the trained model
model = load_model("model/xception_plant_model.h5")

# Performance Evaluation on Validation Set
val_gen.reset()
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_gen.classes

print("\nClassification Report (Validation Set):")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix (Validation Set):")
print(confusion_matrix(y_true, y_pred))

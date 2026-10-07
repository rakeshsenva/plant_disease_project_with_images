import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 123
EPOCHS = 10

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

class_dirs = [d for d in os.listdir(DATA_DIR)
              if os.path.isdir(os.path.join(DATA_DIR, d))]
if len(class_dirs) < 2:
    raise ValueError("At least 2 class folders are required inside dataset.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
with open(CLASS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

loss, accuracy = model.evaluate(val_ds, verbose=0)
print("\nTraining completed!")
print("Classes:", class_names)
print(f"Validation accuracy: {accuracy:.4f}")
print("Model saved:", MODEL_PATH)
print("Class names saved:", CLASS_PATH)

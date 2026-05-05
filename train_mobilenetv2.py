import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# -----------------------------
# Dataset Path
# -----------------------------
dataset_path = r"C:\final year project\strabismus risk assessment\image_dataset_split"

# -----------------------------
# Plot Output Path
# -----------------------------
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Image Parameters
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16

# -----------------------------
# Data Generator
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -----------------------------
# Load MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False   # Freeze base layers

# -----------------------------
# Add Custom Layers
# -----------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Early Stopping (IMPORTANT)
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# Learning Rate Logger
# -----------------------------
class LrLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

lr_logger = LrLogger()

# -----------------------------
# Model Summary (for viva)
# -----------------------------
model.summary()

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stop, lr_logger]
)

# -----------------------------
# Save Model
# -----------------------------
model.save("strabismus_mobilenetv2_model.h5")

print("✅ Model training completed and saved successfully!")

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "image_accuracy_curve.png"), dpi=300)
plt.show()

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "image_loss_curve.png"), dpi=300)
plt.show()

# -----------------------------
# Plot Learning Rate
# -----------------------------
if 'lr' in history.history:
    plt.plot(history.history['lr'], label='Learning Rate')
    plt.legend()
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.show()
# main.py

# -----------------------------------
# SECTION 1: IMPORT LIBRARIES
# -----------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, RandomFlip, RandomRotation
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

print("TensorFlow Version:", tf.__version__)

# -----------------------------------
# SECTION 2: LOAD AND PREPARE DATA
# -----------------------------------
print("\nLoading and preparing CIFAR-100 data...")

# Load the CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# --- Data Preprocessing ---
# One-hot encode the integer labels (100 classes)
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

print("Data loaded successfully.")
print("Training data shape:", x_train.shape)
print("Number of classes:", y_train.shape[1])

# --- Define Image Size ---
# Pre-trained models like MobileNetV2 expect a larger input size than CIFAR's 32x32.
# We will resize the images. 96x96 is a good compromise.
IMG_SIZE = 96

# -----------------------------------
# SECTION 3: BUILD THE DATA PIPELINE WITH AUGMENTATION
# -----------------------------------
print("\nBuilding data augmentation and preprocessing pipeline...")

# Create a small data augmentation model
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
])

# Define the input layer
inputs = Input(shape=(32, 32, 3))

# Define the workflow for preparing an image
# 1. Resize images to the size MobileNetV2 expects
resized_images = tf.keras.layers.UpSampling2D(size=(IMG_SIZE//32, IMG_SIZE//32))(inputs)
# 2. Apply data augmentation (only during training)
augmented_images = data_augmentation(resized_images)
# 3. Apply the specific preprocessing required by MobileNetV2
preprocessed_images = preprocess_input(augmented_images)

# -----------------------------------
# SECTION 4: BUILD THE TRANSFER LEARNING MODEL
# -----------------------------------
print("\nBuilding the Transfer Learning model with MobileNetV2...")

# Load MobileNetV2 pre-trained on ImageNet, without its final classification layer
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False, # Don't include the ImageNet classifier
                         weights='imagenet')

# Freeze the convolutional base to prevent its weights from being updated during initial training
base_model.trainable = False

# --- Create the full model ---
# 1. Pass the preprocessed images through the frozen base_model
x = base_model(preprocessed_images, training=False)
# 2. Pool the features to a single vector
x = GlobalAveragePooling2D()(x)
# 3. Add our custom classifier head
# A Dense layer with 100 neurons (for 100 classes) and softmax activation
outputs = Dense(100, activation='softmax')(x)

# Combine everything into a single model
model = Model(inputs, outputs)

# Display the model's architecture
model.summary()

# -----------------------------------
# SECTION 5: INITIAL TRAINING (FEATURE EXTRACTION)
# -----------------------------------
print("\nPhase 1: Initial Training (with frozen base model)...")

# Compile the model for the first phase
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (only the new classifier head is being trained)
initial_epochs = 10
history = model.fit(x_train, y_train,
                    epochs=initial_epochs,
                    batch_size=64,
                    validation_split=0.2)

# -----------------------------------
# SECTION 6: FINE-TUNING THE MODEL
# -----------------------------------
print("\nPhase 2: Fine-Tuning (unfreezing top layers)...")

# Unfreeze the base model to allow fine-tuning
base_model.trainable = True

# We'll only fine-tune the top layers. Let's freeze all layers before layer 100.
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a very low learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training the model
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(x_train, y_train,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1], # Start from where we left off
                         batch_size=64,
                         validation_split=0.2)

# -----------------------------------
# SECTION 7: FINAL EVALUATION AND INTERPRETATION
# -----------------------------------
print("\nEvaluating final model performance...")

# Evaluate the final model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# --- Plotting Combined Training History ---
# Append the fine-tuning history to the initial history
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(initial_epochs - 1, linestyle='--', color='k', label='Start Fine-Tuning') # Separator line
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(initial_epochs - 1, linestyle='--', color='k', label='Start Fine-Tuning') # Separator line
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('advanced_training_history.png')
print("\nSaved training history plot as 'advanced_training_history.png'")
plt.show()
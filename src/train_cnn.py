import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

img_height, img_width = 224, 224
batch_size = 16

train_ds  = tf.keras.preprocessing.image_dataset_from_directory(
  "./spectrograms/train",
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./spectrograms/val",
  image_size=(img_height, img_width),
  batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./spectrograms/test",
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False
)

class_names = train_ds.class_names

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_accuracy'], label='validation accuracy')
# plt.legend()
# plt.savefig("../plots/accuracy_plot.png")
# plt.show()

# loss, accuracy = model.evaluate(val_ds)
# print(f"Validation accuracy: {accuracy:.4f}")

# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Final test accuracy: {test_acc:.4f}")

# model.save('../models/spectrogram_classifier.keras')

# --- PLOTTING ACCURACY ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.savefig("./plots/training_accuracy.png") # Save BEFORE show
# plt.show() 
plt.close()

# --- EVALUATION ---
print("\nEvaluating on Test Set...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy:.4f}")

model.save('./models/spectrogram_classifier.keras')

# --- CONFUSION MATRIX & REPORT ---
print("\nGenerating Confusion Matrix...")
y_true = []
y_pred = []

# Iterate over the test dataset to get predictions
for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    # Convert probabilities to binary predictions (0 or 1)
    y_pred.extend((predictions > 0.5).astype(int).flatten())

# Compute Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig("./plots/confusion_matrix.png")
plt.close()

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
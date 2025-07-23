import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

img_height, img_width = 224, 224
batch_size = 16

train_ds  = tf.keras.preprocessing.image_dataset_from_directory(
  "../spectrograms",
  validation_split=0.3,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "../spectrograms",
  validation_split=0.3,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

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

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy:.4f}")

model.save('../models/spectrogram_classifier.keras')
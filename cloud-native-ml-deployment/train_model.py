import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

print("===== TRAINING MODEL =====")

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype('float32') / 255

# Fast CPU subset
x_train = x_train[:10000]
y_train = y_train[:10000]

model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=32
)

model.save('mnist_model.keras')

print("Model training complete!")

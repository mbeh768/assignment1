# tensorflow implementation of FC neural network
import os
from datetime import datetime

import tensorflow as tf


batch_size = 4096

# initialize dataset
dataset = tf.keras.datasets.fashion_mnist

# load training and testing portions of dataset
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# normalize images [0-1]
x_train, x_test = x_train/255.0, x_test/255.0

# place into datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = val_ds.batch(batch_size)

# define model
model = tf.keras.Sequential([
    # flatten 28x28 characters into 784x1 vector
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # hidden layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    # output layer
    tf.keras.layers.Dense(10)   # 10 output classes for 10 digits
])  

# compile model 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

# training
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    train_ds,
    epochs=50,
    batch_size=batch_size,
    validation_data=val_ds,
    callbacks=[tensorboard_callback],
)

# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

print(model.summary())
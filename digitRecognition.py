import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

train_labels = train_labels
test_labels = test_labels

train_images = train_images.reshape(-1,784)/255.0
test_images = test_images.reshape(-1,784)/255.0
'''
def create_model():
	model = tf.keras.models.Sequential([
		keras.layers.Dense(512, activation='relu', input_shape=(784,)),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(10)
	])
	model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[tf.metrics.SparseCategoricalAccuracy()])
	return model

model = create_model()

model.summary()

checkpoint_path = "models/model3/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

model.fit(train_images,train_labels,epochs=8,validation_data=(test_images, test_labels),callbacks=[cp_callback])

os.listdir(checkpoint_dir)

model.save('models/model3/model')
'''

new_model = tf.keras.models.load_model('models/model3/model')

new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

import tensorflow as tf 
import numpy as np 
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs={}):
        if(logs.get('accuracy')>0.6):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training=True

mnist=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0

callbacks=myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10,callbacks=[callbacks])
model.evaluate(x_test,y_test)


# just to visualize.
# import matplotlib.pyplot as plt
# f, axarr = plt.subplots(3,4)
# FIRST_IMAGE=0
# SECOND_IMAGE=7
# THIRD_IMAGE=26
# CONVOLUTION_NUMBER = 1
# from tensorflow.keras import models
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
# for x in range(0,4):
#   f1 = activation_model.predict(x_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[0,x].grid(False)
#   f2 = activation_model.predict(x_test[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[1,x].grid(False)
#   f3 = activation_model.predict(x_test[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[2,x].grid(False)
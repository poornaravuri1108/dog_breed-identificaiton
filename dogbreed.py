import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.layers import Dense,Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB3
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

image_size = (300, 300)
img_height = 300
img_width = 300
batch_size = 16
NUM_CLASSES = 120

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/home/kranthi/my_prog/2021-22/kranthi/Dogbreed_classification/dataset/",
  validation_split=0.2,
  label_mode='categorical',
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/home/kranthi/my_prog/2021-22/kranthi/Dogbreed_classification/dataset/",
  validation_split=0.2,
  label_mode='categorical',
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names[:5])

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

print(labels_batch)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis("off")
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(num_classes):
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = img_augmentation(inputs)
    base_model = EfficientNetB3(include_top=False, input_tensor=x, drop_connect_rate=0.3, weights="imagenet")

    # Freeze the pretrained weights
    base_model.trainable = True
##    for layer in base_model.layers[:265]:
##      layer.trainable = False
##    for layer in base_model.layers[265:]:
##      layer.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
    
    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    
    # Load weights from second time onwards by selecting the file name properly
    model.load_weights("save_at_19.h5")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
  
##    return model,base_model
##model,base_model = build_model(num_classes=NUM_CLASSES)
##for i, layer in enumerate(base_model.layers):
##   print(i, layer.name)
##exit()

NUM_CLASSES = 120
model = build_model(num_classes=NUM_CLASSES)

tbCallBack = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True, write_images=False)
checkpoint_filepath = 'save_at_{epoch}.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
callbacks_list = [tbCallBack, model_checkpoint_callback]

epochs = 100
hist = model.fit(train_ds, epochs=epochs, initial_epoch=19, validation_data=val_ds, 
                 callbacks=callbacks_list, verbose=1)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


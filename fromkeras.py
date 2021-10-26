from tensorflow.keras.applications import EfficientNetB0

model = EfficientNetB0(include_top=False, weights='imagenet')
# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224

import tensorflow_datasets as tfds

batch_size = 16

dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(dataset_name, split=["train", "test"],
                                         with_info=True, as_supervised=True)
NUM_CLASSES = ds_info.features["label"].num_classes

size = (IMG_SIZE, IMG_SIZE)
import tensorflow as tf
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

import matplotlib.pyplot as plt

def format_label(label):
    string_label = label_info.int2str(label)
    return string_label.split("-")[1]


import numpy as np
label_info = ds_info.features["label"]
for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(format_label(label)))
    plt.axis("off")
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

##for image, label in ds_train.take(1):
##    for i in range(9):
##        ax = plt.subplot(3, 3, i + 1)
##        aug_img = img_augmentation(tf.expand_dims(image, axis=0))
##        plt.imshow(aug_img[0].numpy().astype("uint8"))
##        plt.title("{}".format(format_label(label)))
##        plt.axis("off")
##
##plt.show()
####exit()

# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    print(label)
    return image, label


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)

ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


import matplotlib.pyplot as plt
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

model = build_model(num_classes=NUM_CLASSES)

epochs = 25 
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
plot_hist(hist)

##def unfreeze_model(model):
##    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
##    for layer in model.layers[-20:]:
##        if not isinstance(layer, layers.BatchNormalization):
##            layer.trainable = True
##
##    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
##    model.compile(
##        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
##    )
##
##
##unfreeze_model(model)
##
##epochs = 10  # @param {type: "slider", min:8, max:50}
##hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
##plot_hist(hist)

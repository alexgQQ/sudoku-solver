import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


def train_model():

    # the data, split between train and test sets
    (_x_train, _y_train), (_x_test, _y_test) = keras.datasets.mnist.load_data()

    x_train = []
    y_train = []

    for data, label in zip(_x_train, _y_train):
        if label != 0:
            x_train.append(data)
            y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    x_test = []
    y_test = []

    for data, label in zip(_x_test, _y_test):
        if label != 0:
            x_test.append(data)
            y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    model.save("digits_model.h5")


# im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
# gray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
# gray = gray.reshape((1, 28, 28, 1))
# gray /= 255
# print(gray.shape)

# model = load_model("test_model.h5")
# prediction = model.predict(gray)
# print(prediction.argmax())

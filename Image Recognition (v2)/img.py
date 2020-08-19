from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype("float32") / 255

train_labels = to_categorical(train_labels)

model.fit(train_data, train_labels, batch_size=3000, epochs=5)


test_data = test_data.reshape(10000, 28, 28, 1)
test_data = test_data.astype("float32") / 255

test_labels = to_categorical(test_labels)

loss, accuracy = model.evaluate(test_data, test_labels)

print("Loss:", loss, "\nAccuracy:", accuracy)

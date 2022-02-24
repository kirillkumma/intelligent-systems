import pandas
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.callbacks import History
from sklearn.preprocessing import LabelEncoder

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

model1 = Sequential()
model_history1 = History()
model1.add(Dense(4, activation='relu'))
model1.add(Dense(3, activation='softmax'))

model1.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['accuracy'])

model1.fit(X, dummy_y, epochs=75, batch_size=10,
           validation_split=0.1, callbacks=[model_history1])

model2 = Sequential()
model_history2 = History()
model2.add(Dense(4, activation='relu'))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(3, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['accuracy'])

model2.fit(X, dummy_y, epochs=180, batch_size=10,
           validation_split=0.1, callbacks=[model_history2])

model3 = Sequential()
model_history3 = History()
model3.add(Dense(4, activation='relu'))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(3, activation='softmax'))

model3.compile(optimizer='sgd', loss='categorical_crossentropy',
               metrics=['accuracy'])

model3.fit(X, dummy_y, epochs=200, batch_size=10,
           validation_split=0.1, callbacks=[model_history3])


def create_diagram(n, title, model_history):
    N = np.arange(0, n)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, model_history.history["loss"], label="train_loss")
    plt.plot(N, model_history.history["val_loss"], label="val_loss")
    plt.plot(N, model_history.history["accuracy"], label="train_accuracy")
    plt.plot(N, model_history.history["val_accuracy"], label="val_accuracy")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


create_diagram(75, "Model 1", model_history1)
create_diagram(180, "Model 2", model_history2)
create_diagram(200, "Model 3", model_history3)

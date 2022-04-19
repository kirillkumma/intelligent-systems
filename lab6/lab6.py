import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.datasets import imdb

(training_data, training_targets), (testing_data,
testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets),
axis=0)

print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))
length = [len(i) for i in data]
print("Average Review length:", np.mean(length))

print("Label:", targets[0])
print(data[0])

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]])
print(decoded)

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1    
    return results

data1 = vectorize(data)
targets = np.array(targets).astype("float32")

test_x1 = data1[:10000]
test_y1 = targets[:10000]
train_x1 = data1[10000:]
train_y1 = targets[10000:]

model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

results = model.fit(train_x1, train_y1, epochs= 2, batch_size = 500, validation_data = (test_x1, test_y1))
print(np.mean(results.history['val_accuracy']))

data2 = vectorize(data, 12000)
test_x2 = data2[:10000]
test_y2 = targets[:10000]
train_x2 = data2[10000:]
train_y2 = targets[10000:]
model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(12000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
results = model.fit(train_x2, train_y2, epochs= 2, batch_size = 500, validation_data = (test_x2, test_y2))
print(np.mean(results.history['val_accuracy']))

data3 = vectorize(data)
test_x3 = data3[:500]
test_y3 = targets[:500]
train_x3 = data3[:]
train_y3 = targets[:]
model = models.Sequential()
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
results = model.fit(train_x3, train_y3, epochs= 3, batch_size = 500, validation_data = (test_x3, test_y3))
print(np.mean(results.history['val_accuracy']))

predict = input()
word_index_org = imdb.get_word_index()
word_index = dict()

for word,number in word_index_org.items():
    word_index[word] = number + 3
str = predict.lower().split(' ')

def parseNumber(str):
    arr = []
    for word in str:
        arr.append(word_index[word])
    result = vectorize(arr)
    return result

prediction = model.predict(parseNumber(str))
print('Positive:' if np.mean(prediction)>0.5 else 'Negative:',np.mean(prediction))

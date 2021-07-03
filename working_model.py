import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df=pd.read_csv("XRP-USD.csv")
df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

training_set = df.iloc[:1800, 1:2].values
test_set = df.iloc[1800:, 1:2].values

# Creating a data structure with 90 time-steps and 1 output
x_train = []
y_train = []
for i in range(90, 1800):
    x_train.append(training_set[i-90:i, 0])
    y_train.append(training_set[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = tf.keras.models.Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(tf.keras.layers.LSTM(units = 50))
model.add(tf.keras.layers.Dropout(0.2))
# Adding the output layer
model.add(tf.keras.layers.Dense(units = 1))

# Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the model to the training set
model.fit(x_train, y_train, epochs = 10, batch_size = 32)

# Getting the predicted price
dataset_train = df.iloc[:1800, 1:2]
dataset_test = df.iloc[1800:, 1:2]
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 90:].values
inputs = inputs.reshape(-1,1)

x_test = []
for i in range(90, 506):
    x_test.append(inputs[i-90:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_test.shape)

predicted_XRP_price = model.predict(x_test)
training_XRP_price = model.predict(x_train)

# Visualising the results

print(len(predicted_XRP_price))
print (len(dataset_total))

plt.plot(df.loc[:1799, 'Time'], dataset_train.values, color = "red", label = "Training XRP-USD Price")
plt.plot(df.loc[1800:, 'Time'], predicted_XRP_price, color = "blue", label = "Predicted XRP-USD Price")
plt.plot(df.loc[1800:, 'Time'], dataset_test.values, color = "green", label = "Testing XRP-USD Price")
plt.xticks(np.arange(0, len(df.loc[:,'Time']), 30))
plt.title('XRP-USD model')
plt.xlabel('Time')
plt.ylabel('XRP-USD Price')
plt.legend()
plt.show()
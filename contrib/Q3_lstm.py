import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split

# %% Load data.
x, y = pd.read_pickle('raw/Q3_pre_processed.pkl')
std_ = pd.read_pickle('raw/Q3_standard_scaler.pkl')

# %% Make train and valid set.
x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=482963)

# %% Construct the model.
l0 = Input(shape=(20, 1))
l1, _, _ = LSTM(32, return_state=True)(l0)
h2 = Dense(32, activation="relu")(l1)
# h3 = Dense(8, activation="relu")(h2)
l6 = Dense(1, activation="relu")(h2)
my_model = tf.keras.models.Model(l0, l6)
my_model.compile(optimizer="sgd", loss="mse")
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)

# %% Fit the model.
save_best = tf.keras.callbacks.ModelCheckpoint(f'raw/Q3_lstm.h5', monitor="val_loss", save_best_only=True)
my_model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=1000, batch_size=100,
             callbacks=[stop_early, save_best])

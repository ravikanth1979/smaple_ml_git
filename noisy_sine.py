#!/usr/bin/env python3

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.objectives import mean_absolute_error, cosine_proximity
import matplotlib.pyplot as plt

POINTS_PER_WF = int(1e4)
X_SPACE = np.linspace(0, 100, POINTS_PER_WF)

def make_waveform_with_noise():
    def add_noise(vec):
        stdev = float(np.random.uniform(0.01, 0.2))
        return vec + np.random.normal(0, stdev, size=len(vec))

    f = np.random.choice((np.sin, np.cos))
    wf = f(X_SPACE * np.random.normal(scale=5)) *\
         np.random.normal(scale=5) + np.random.normal(scale=50)
    return wf, add_noise(wf)

RESCALING = 1e-3
BATCH_SHAPE = (1, POINTS_PER_WF, 1)

model = Sequential([
    TimeDistributed(Dense(5, activation='tanh'), batch_input_shape=BATCH_SHAPE),
    LSTM(20, activation='tanh', inner_activation='sigmoid', return_sequences=True),
    LSTM(20, activation='tanh', inner_activation='sigmoid', return_sequences=True),
    TimeDistributed(Dense(1, activation='tanh'))
])

def compute_loss(y_true, y_pred):
    skip_first = POINTS_PER_WF // 2
    y_true = y_true[:, skip_first:, :] * RESCALING
    y_pred = y_pred[:, skip_first:, :] * RESCALING
    me = mean_absolute_error(y_true, y_pred)
    cp = cosine_proximity(y_true, y_pred)
    return me + cp

model.summary()
model.compile(optimizer='adam', loss=compute_loss,
              metrics=['mae', 'cosine_proximity'])

NUM_ITERATIONS = 30000

for iteration in range(NUM_ITERATIONS):
    wf, noisy_wf = make_waveform_with_noise()
    y = wf.reshape(BATCH_SHAPE) * RESCALING
    x = noisy_wf.reshape(BATCH_SHAPE) * RESCALING
    info = model.train_on_batch(x, y)

model.save_weights('final.hdf5')

wf, noisy_wf = make_waveform_with_noise()
y = wf.reshape(BATCH_SHAPE) * RESCALING
x = noisy_wf.reshape(BATCH_SHAPE) * RESCALING

plt.plot(x[0],color='red',label='Real rental demand',linewidth=4.0)
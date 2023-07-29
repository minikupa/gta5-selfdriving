from datetime import datetime

import numpy as np
from keras import Input, Model
from keras.src.layers import Conv2D, Flatten, Dense, Concatenate, Dropout, MaxPooling2D
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def nvidia():
    screen_input = Input(shape=(66, 200, 3))
    minimap_input = Input(shape=(50, 50, 3))

    screen = Conv2D(filters=24, kernel_size=5, strides=2, activation='relu')(screen_input)
    screen = Conv2D(filters=36, kernel_size=5, strides=2, activation='relu')(screen)
    screen = Conv2D(filters=48, kernel_size=5, strides=2, activation='relu')(screen)
    screen = Conv2D(filters=64, kernel_size=3, activation='relu')(screen)
    screen = Conv2D(filters=64, kernel_size=3, activation='relu')(screen)
    screen = Dropout(0.3)(screen)
    screen = Flatten()(screen)

    minimap = Conv2D(filters=32, kernel_size=5, activation='relu')(minimap_input)
    minimap = MaxPooling2D(strides=(2, 2))(minimap)
    minimap = Conv2D(filters=64, kernel_size=5, activation='relu')(minimap)
    minimap = Dropout(0.2)(minimap)
    minimap = MaxPooling2D(strides=(2, 2))(minimap)
    minimap = Flatten()(minimap)

    merged = Concatenate()([screen, minimap])

    x = Dense(1024, activation='relu')(merged)
    x = Dense(100, activation='relu')(x)

    output = Dense(2)(x)

    model = Model(inputs=[screen_input, minimap_input], outputs=output)
    model.summary()
    return model


file = np.load('final_data.npy', allow_pickle=True)

screen, minimap, output = [], [], []

for i in file:
    screen.append(i['screen'])
    minimap.append(i['minimap'])
    output.append(i['output'])

data = train_test_split(screen, minimap, output)
screen_train, screen_test, minimap_train, minimap_test, y_train, y_test = data

screen_train = np.array(screen_train, dtype=np.float32)
screen_test = np.array(screen_test, dtype=np.float32)
minimap_train = np.array(minimap_train, dtype=np.float32)
minimap_test = np.array(minimap_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

screen_train *= 1 / 255.
screen_test *= 1 / 255.
minimap_train *= 1 / 255.
minimap_test *= 1 / 255.

print(screen_train.shape)
print(screen_test.shape)
print(minimap_train.shape)
print(minimap_test.shape)
print(y_train.shape)
print(y_test.shape)

model = nvidia()
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

print(f"학습 시작 - {datetime.now()}")
history = model.fit(x=[screen_train, minimap_train], y=y_train, epochs=20, batch_size=32)
print(f"학습 끝 - {datetime.now()}")

print("모델 성능")
model.evaluate(x=[screen_test, minimap_test], y=y_test)
model.save("gta5-selfdriving.keras")
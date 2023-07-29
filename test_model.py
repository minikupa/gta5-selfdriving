import random
import cv2
from keras import backend as K
import numpy as np
from keras.src.saving.saving_api import load_model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = load_model("gta5-selfdriving.keras", custom_objects={'root_mean_squared_error': root_mean_squared_error})
data = np.load("final_data.npy", allow_pickle=True)

for i in range(100000):
    index = random.randint(0, len(data) - 1)

    screen = np.array(data[index]['screen'], dtype=np.float32).reshape(1, 66, 200, 3)
    minimap = np.array(data[index]['minimap'], dtype=np.float32).reshape(1, 50, 50, 3)

    screen *= 1 / 255.
    minimap *= 1 / 255.

    cv2.imshow('test', data[index]['screen'])
    print(model.predict(x=[screen, minimap]))
    cv2.waitKey(3000)

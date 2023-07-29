import cv2
from keras import backend as K
import numpy as np
from keras.src.saving.saving_api import load_model
from screenshot import grab_screen
from vjoy import vj, setJoy



def setJoy_Steer_Throttle (value_steerX, value_throttleX, button_state,scale = 16384):
    value_steerX = value_steerX +1
    value_throttleX = value_throttleX +1
    xPos_steering = int(value_steerX*scale)
    xPos_throttle = int(value_throttleX*scale)
    joystickPosition = vj.generateJoystickPosition(wAxisX = xPos_steering, wAxisY = int(scale/2), wAxisZRot = xPos_throttle, lButtons = button_state)
    vj.update(joystickPosition)


model = load_model("gta5-selfdriving.keras")

window_name = "GTA5"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

start_loop, data_loop = True, True

vj.open()

while start_loop:
    key = cv2.waitKey(1)
    if key == ord('s'):
        for i in range(5):
            img = np.full(shape=(750, 2000, 3), fill_value=255, dtype=np.uint8)

            cv2.putText(img, f"{5 - i}sec", (30, 130),
                cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), cv2.LINE_8)
            cv2.imshow(window_name, img)
            cv2.waitKey(1000)
        
        start_loop = False

cv2.destroyAllWindows()
while data_loop:
    screen = grab_screen(region=(0, 80, 1270, 830))
    screen = np.array(screen)
    screen = cv2.resize(screen, (200, 66))
    screen = screen[:, :, :3]

    minimap = grab_screen(region=(20, 670, 210, 810))
    minimap = np.array(minimap)
    minimap = cv2.resize(minimap, (50, 50))
    minimap = minimap[:, :, :3]

    screen = np.array(screen, dtype=np.float32).reshape(1, 66, 200, 3)
    minimap = np.array(minimap, dtype=np.float32).reshape(1, 50, 50, 3)

    screen *= 1 / 255.
    minimap *= 1 / 255.

    output = model.predict(x=[screen, minimap])[0]
    steering = output[0]
    throttle = output[1]

    print(steering, throttle)
    setJoy_Steer_Throttle(steering, throttle, 1)
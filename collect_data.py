import time
import re
import os
import cv2
import numpy as np
from screenshot import grab_screen
import get_joystick
from PIL import ImageFont, ImageDraw, Image

window_name = "GTA5"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

joystick = get_joystick.init_joystick()
get_joystick.read_joystick(joystick)
get_joystick.x = 0
get_joystick.r2 = 0

start_loop, data_loop = True, True
training_data = []

while start_loop:
    key = cv2.waitKey(1)
    if key == ord('s'):
        for i in range(7):
            img = np.full(shape=(750, 2000, 3), fill_value=255, dtype=np.uint8)

            cv2.putText(img, f"{7 - i}sec", (30, 130),
                cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), cv2.LINE_8)
            cv2.imshow(window_name, img)
            cv2.waitKey(1000)
        
        start_loop = False

direct = "./data"

file_list = os.listdir(direct)
jpg_list = []


for file in file_list:
    jpg_list.append(file)

jpg_list.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
num = jpg_list[-1].replace('gta_ai.npy', '')

while data_loop:
    get_joystick.read_joystick(joystick)
    img = np.full(shape=(750, 2000, 3), fill_value=255, dtype=np.uint8)

    cv2.putText(img, f"angle : {get_joystick.x}", (30, 130),
                cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), cv2.LINE_8)
    cv2.putText(img, f"throttle : {get_joystick.r2}", (30, 300),
                cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), cv2.LINE_8)
    cv2.putText(img, f"index : {len(training_data)}", (30, 470),
                cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), cv2.LINE_8)
    
    cv2.imshow(window_name, img)

    screen = grab_screen(region=(0, 80, 1270, 830))
    screen = np.array(screen)
    screen = cv2.resize(screen, (200, 66))

    minimap = grab_screen(region=(20, 670, 210, 810))
    minimap = np.array(minimap)
    minimap = cv2.resize(minimap, (50, 50))

    output = [get_joystick.x, get_joystick.r2]
    training_data.append({'screen': screen, 'minimap' : minimap, 'output': output})

    if len(training_data) == 500:
        data_loop = False
        np.save(f"./data/{int(num)+1}gta_ai.npy", training_data)
        cv2.destroyAllWindows()


    key = cv2.waitKey(70)
    if key == ord('q'):
        data_loop = False
        cv2.destroyAllWindows()

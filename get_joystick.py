import random
import pygame
import time

def init_joystick():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick found.")
        return None
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Initialized joystick:", joystick.get_name())
    return joystick

def read_joystick(joystick):
    global x 
    global r2

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
        if event.type == pygame.JOYAXISMOTION:
            # 조이스틱 축의 움직임을 읽어옵니다.
            if event.axis == 0:  # X 축
                x = round(event.value, 7)             
            elif event.axis == 5:  # R2 버튼 세기
                r2 = round(event.value, 7)
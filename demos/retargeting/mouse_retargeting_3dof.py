import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

import pygame
import numpy as np
import matplotlib.pyplot as plt

from core.kinematics import (
    fk_3dof,
    jacobian_3dof,
    fk_all_3dof

)

from core.controller import damped_pinv

def draw_link(screen, p1, p2):

    pygame.draw.line(
        screen,
        (80,80,80),
        p1,
        p2,
        20
    )

    pygame.draw.line(
        screen,
        (180,180,180),
        p1,
        p2,
        8
    )
def draw_joint(screen, p):

    pygame.draw.circle(
        screen,
        (40,40,40),
        p,
        16
    )

    pygame.draw.circle( 
        screen,
        (220,220,220),
        p,
        8
    )
def draw_ee(screen, p):

    pygame.draw.circle(
        screen,
        (0,120,255),
        p,
        12
    )


def world_to_screen(p):

    return (
        int(p[0] * WIDTH),
        int((1-p[1]) * HEIGHT)
    )


pygame.init()

WIDTH = 800
HEIGHT = 800

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Mouse Retargeting Demo"
)

q = np.array([
    0.5,
    0.3,
    -0.2
])

dt = 0.01
Kp = 10.0

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False
    

    mx, my = pygame.mouse.get_pos()

    target = np.array([
        mx / WIDTH,
        1.0 - my / HEIGHT
    ])


    ee = fk_3dof(q)

    error = target - ee

    xdot = Kp * error

    J = jacobian_3dof(q)

    J_pinv = damped_pinv(
        J,
        lam=0.05
    )

    qdot = J_pinv @ xdot

    qdot = np.clip(
        qdot,
        -2.0,
        2.0
    )

    q += qdot * dt

    max_qdot = 0

    speed = np.linalg.norm(qdot)

    max_qdot = max(
        max_qdot,
        speed
    )

    screen.fill((255,255,255))

    # 鼠标目标点

    pygame.draw.circle(
        screen,
        (255,0,0),
        (mx,my),
        8
    )

    # 机械臂末端

    ex = int(ee[0] * WIDTH)
    ey = int((1-ee[1]) * HEIGHT)

    p0,p1,p2,p3 = fk_all_3dof(q)

    s0 = world_to_screen(p0)
    s1 = world_to_screen(p1)
    s2 = world_to_screen(p2)
    s3 = world_to_screen(p3)

    draw_link(screen,s0,s1)
    draw_link(screen,s1,s2)
    draw_link(screen,s2,s3)

    draw_joint(screen,s0)
    draw_joint(screen,s1)
    draw_joint(screen,s2)

    draw_ee(screen,s3)

    pygame.draw.rect(
        screen,
        (100,100,100),
        (s0[0]-30,
        s0[1]-15,
        60,
        30)
    )
    pygame.display.flip()

pygame.quit()

print(
    "max joint speed =",
    max_qdot
)
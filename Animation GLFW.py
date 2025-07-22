#
# from glfw import (init as glfw_init,
#                   create_window as glfw_create_widow,
#                   window_should_close as glfw_window_should_close,
#                   set_window_close_callback,
#                   poll_events as glfw_poll_events,
#                   terminate as glfw_terminate)

from OpenGL.GL import *

from OpenGL_2D_class_GLFW_old import gl2D, gl2DCircle, gl2DText

import numpy as np

# import the Problem Specific class
from SimpleDrawingClass import SimpleDrawing

mydrawing = SimpleDrawing()


def drawGiant():
    # this is what actually draws the picture
    mydrawing.DrawPicture(True)
    if mydrawing.frame > -1:
        glColor3f(1, 1, 1)  #
        gl2DText('Frame Number: ' + str(mydrawing.frame), 5.5, -0.8)



def AnimationCallback(frame, nframes):
    # calculations needed to configure the picture
    # these could be done here or by calling a class method
    d = mydrawing # very useful shorthand!
    d.giantX = d.giantStart + (d.giantStop-d.giantStart)*frame/nframes
    mydrawing.frame = frame

def main():

    gl2d = gl2D(None,drawGiant)
    gl2d.setViewSize(-1, 6, -1, 6,False)

    nframes = 60
    gl2d.glStartAnimation(AnimationCallback, nframes,delaytime=0.1,
                                       reverse=True, repeat=False, reset=True)
    gl2d.glWait()
    print("hello")

main()


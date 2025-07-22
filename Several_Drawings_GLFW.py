import numpy as np

from glfw import (init as glfw_init,
                  create_window as glfw_create_widow,
                  window_should_close as glfw_window_should_close,
                  set_window_close_callback,
                  poll_events as glfw_poll_events,
                  terminate as glfw_terminate
                  )

from OpenGL.GL import *

from OpenGL_2D_class_GLFW_old import gl2D, gl2DCircle, gl2DText,gl2DArrow,gl2DArc

from HersheyFont import HersheyFont
hf = HersheyFont()

from SimpleDrawingClass import SimpleDrawing
mydrawing = SimpleDrawing()

def drawHouse():

    glColor3f(0, 1, 0)
    glLineWidth(3)
    glBegin(GL_LINE_STRIP)  # begin drawing connected lines
    # use GL_LINE for drawing a series of disconnected lines
    glVertex2f(1, 0)
    glVertex2f(3, 0)
    glVertex2f(3, 2)
    glVertex2f(1, 2)
    glVertex2f(1, 0)
    glEnd()

    # draw the roof
    glBegin(GL_LINE_STRIP)  # begin drawing connected lines
    glVertex2f(1, 2)
    glVertex2f(2, 3)
    glVertex2f(3, 2)
    glEnd()

    # Draw the ground
    glBegin(GL_LINE_STRIP)  # begin drawing connected lines
    glVertex2f(0, 0)
    glVertex2f(5, 0)
    glEnd()

    # Draw sun
    radius = 0.7
    glColor3f(0.9, 0.9, 0)  #
    glLineWidth(1)
    gl2DCircle(4, 4, radius, fill=True)

    glLineWidth(3)
    glBegin(GL_LINES)  # begin drawing disconnedted lines
    theta = np.linspace(0,2*np.pi,10)
    for i in range(len(theta)):
        glVertex2f(4+radius*np.cos(theta[i]), 4+radius*np.sin(theta[i]))
        glVertex2f(4+2*radius*np.cos(theta[i]), 4+2*radius*np.sin(theta[i]))
    glEnd()

    glColor3f(0, 0, 0)  #
    gl2DText('"Our House ..... is a very very very fine house"', -1,5.5)
    gl2DText('... Crosby, Stills, Nash and Young', -1,5)

def drawMachine():
    # this is what actually draws the picture
    glColor3f(1, 1, 1)  #
    glLineWidth(1.5)
    glBegin(GL_LINE_STRIP)  # begin drawing connected lines
    glVertex2f(0,0)
    glVertex2f(120,0)
    glVertex2f(120,44)
    glVertex2f(88,77)
    glVertex2f(33,77)
    glVertex2f(0,44)
    glVertex2f(0,0)
    glEnd()

    gl2DCircle(36,36,18)
    gl2DCircle(84,20,10)

    glColor3f(1, 0, 0)  #
    gl2DArrow(120, 0, 4, angleDeg = -90, widthDeg = 40, toCenter = True)
    gl2DArrow(120, 44, 4, angleDeg = 90, widthDeg = 40, toCenter = True)
    glColor3f(0, 0, 0)  #
    gl2DArrow(33, 77, 2, angleDeg = 45)
    gl2DArrow(0, 44, 4, angleDeg = -135)



    glColor3f(0, 1, 0)  #
    glLineWidth(1.5)
    gl2DArc(36,36,18*0.75,0,180)
    glColor3f(1, 1, 0)  #
    glLineWidth(3)
    gl2DArc(84,20, 10*1.5,  135,45)


    glColor3f(0, 1, 1)  #
    glLineWidth(1.5)
    hf.drawText("hello world",25,60, scale = 10)
    hf.drawText("hello world",55,40, scale = 8, slant = 0.5,angle = 10, center = True)
    hf.drawText("hello again",55,20,scale = 5, angle = -30)

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

    # Draw the house, set the window width and height
    gl2d = gl2D(None,drawHouse,width=1200, height= 600)
    gl2d.setViewSize(-1, 6, -1, 6,False)
    gl2d.glWait()  #wait for the user to close the window

    print("Finished drawing 1")

    # Draw the machine, use the default window width and height
    gl2d = gl2D(None,drawMachine)
    gl2d.setViewSize(-6, 126, -3, 80, allowDistortion=False)
    gl2d.glWait()  #wait for the user to close the window

    print("Finished drawing 2")

    # Animate the Giant, use the default window width and height
    gl2d = gl2D(None,drawGiant)
    gl2d.setViewSize(-1, 6, -1, 6,False)
    nframes = 60
    gl2d.glStartAnimation(AnimationCallback, nframes,delaytime=0.02,
                                       reverse=True, repeat=True, reset=True)
    gl2d.glWait()  #wait for the user to close the window

    print("Finished drawing 3")


main()

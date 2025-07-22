import numpy as np

from OpenGL.GL import *

from OpenGL_2D_class_GLFW_old import gl2D, gl2DCircle, gl2DText, gl2DArc, gl2DArrow

from HersheyFont import HersheyFont
hf = HersheyFont()

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
    hf.drawText("hello world", 25, 60, scale=10)
    hf.drawText("hello world", 55, 40, scale=8, slant=0.5, angle=10, center=True)
    hf.drawText("hello again", 55, 20, scale=5, angle=-30)

def main():
    gl2d = gl2D(None,drawMachine,windowType="glfw")
    gl2d.setViewSize(-6, 126, -3, 80, allowDistortion=False)
    gl2d.glWait()  #wait for the user to close the window

    print("Finished drawing the Machine")

main()

    
 






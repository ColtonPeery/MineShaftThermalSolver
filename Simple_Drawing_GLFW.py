import numpy as np
from OpenGL.GL import *
from OpenGL_2D_class_GLFW_old import gl2D, gl2DCircle, gl2DText, gl2DArrow, gl2DArc

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

def main():
    # Draw the house, set the window width and height
    gl2d = gl2D(None,drawHouse,width=1200, height= 600)
    gl2d.setViewSize(-1, 6, -1, 6,False)
    gl2d.glWait()  #wait for the user to close the window

    print("Finished drawing 1")


main()

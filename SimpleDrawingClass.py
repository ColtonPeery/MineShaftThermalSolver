import numpy as np

from OpenGL.GL import *

from OpenGL_2D_class_GLFW_old import  gl2DText, gl2DCircle


class   SimpleDrawing():

    def __init__(self):
        # define any data (including object variables) your program might need
        self.giantX = -6
        self.giantStart = -6
        self.giantStop = 10
        self.frame = -1

    def DoCalculations(self):
        pass

    def DrawGiant(self,gx):
        glColor3f(0, 0, 0)
        gl2DText('Ho Ho Ho"', 0.6 + gx, 3.5)

        glColor3f(0, 1, 0)
        glLineWidth(3)

        # draw the head
        gl2DCircle(0 + gx, 3.5, 0.5, fill=True)

        # draw the body
        glBegin(GL_LINES)  # begin drawing disconnedted lines
        glVertex2f(0+gx,4)
        glVertex2f(0+gx,1.5)

        glVertex2f(0 + gx, 2.8)
        glVertex2f(1 + gx, 1.5)

        glVertex2f(0 + gx, 2.8)
        glVertex2f(-1 + gx, 1.5)

        glVertex2f(0 + gx, 1.5)
        glVertex2f(1 + gx, 0)

        glVertex2f(0 + gx, 1.5)
        glVertex2f(-1 + gx, 0)
        glEnd()


    def DrawPicture(self, drawgiant):
        # this is what actually draws the picture

        # Draw the house
        glColor3f(1, 0.5, 0.5)
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
        glVertex2f(-1, 0)
        glVertex2f(5, 0)
        glEnd()

        # Draw sun
        radius = 0.7
        glColor3f(1, 1, 0)  #
        glLineWidth(1)
        gl2DCircle(4, 4, radius, fill=True)

        glBegin(GL_LINES)  # begin drawing disconnedted lines
        theta = np.linspace(0, 2 * np.pi, 10)
        for i in range(len(theta)):
            glVertex2f(4 + radius * np.cos(theta[i]), 4 + radius * np.sin(theta[i]))
            glVertex2f(4 + 2 * radius * np.cos(theta[i]), 4 + 2 * radius * np.sin(theta[i]))
        glEnd()

        glColor3f(1, 1, 1)  #
        gl2DText('"Our House ..... is a very very very fine house"', 0.5, 5.5)


        # draw the giant at the current location
        if drawgiant:
            self.DrawGiant(self.giantX)
            glScalef(0.5, 0.5, 1)
            self.DrawGiant(self.giantX+0.75)
            glScalef(1/0.5  , 1/0.5, 1)





'''
Modified 7/3/2025   Added support for GLFW pop up window
    Window remains visible until the use closes it.  The window can be resized
    The program is paused until the user closes the window
    To use a GLFW window, create the gl2D object with QTwindow = None
PyQT is still supported but is no longer the ONLY windowing system
'''

import numpy as np
import sys

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from glfw import (  init as glfw_init,
                    create_window as glfw_create_widow,
                    make_context_current as glfw_make_context_current,
                    get_window_size as glfw_get_window_size,
                    swap_buffers as glfw_swap_buffers,
                    window_should_close as glfw_window_should_close,
                    poll_events as glfw_poll_events,
                    terminate as glfw_terminate,
                    set_scroll_callback as glfw_set_scroll_callback ,
                    set_window_size_callback as glfw_set_window_size_callback,
                    set_cursor_pos_callback as glfw_set_cursor_pos_callback,
                    set_mouse_button_callback as glfw_set_mouse_button_callback
                  )


#from PyQt5.QtCore import Qt, QEvent

from time import sleep
import threading
import sys

class gl2D():

    def __init__(self, QTWindow = None, drawCallback = None, windowType ="PyQt",
                 allowDistortion=False,
                 xmin=0, xmax=1, ymin=0, ymax=1,
                 rotate=0, zoom=1.0,
                 backgroundcolor=(0.65, 0.65, 0.65), width = 1200, height = 600, title = "OpenGL"):
        if QTWindow == None:  #this is a glfw window
            glfw_init()
            glfwWindow = glfw_create_widow(width, height, title, None, None)
            windowType = "glfw"
            self.glWindow = glfwWindow
        else:                  # this is a PyQt window
            self.glWindow = QTWindow


        self.glWindowType = windowType
        self.drawCallback = drawCallback
        self.allowDistortion = allowDistortion
        self.glZoomval = zoom  # zoom factor
        self.glRotateval = rotate  # rotation factor in Degrees
        self.glXmin = xmin  # Width of the object to be drawn
        self.glXmax = xmax  # Height of the object to be drawn
        self.glYmin = ymin  # X-location of the center of the object to be drawn
        self.glYmax = ymax  # Y-location of the center of the object to be drawn
        self.glWidth = xmax - xmin  # Width of the object to be drawn
        self.glHeight = ymax - ymin  # Height of the object to be drawn
        self.glXcenter = (xmax + xmin) / 2  # X-location of the center of the object to be drawn
        self.glYcenter = (ymax + ymin) / 2  # Y-location of the center of the object to be drawn
        self.glZoomX = 0
        self.glZoomY = 0
        self.glRotX = self.glXcenter
        self.glRotY = self.glYcenter
        self.glBackgroundColor = backgroundcolor

        # View control data
        self.glViewReady = False
        self.glModel = None  # storing the model matrix
        self.glProjection = None  # storing the projection matrix
        self.glView = None  # storing the Viewport array

        # Animation control Data
        self.glAnimationIsRunning = False #are we already running animation?
        self.glAnimationCallback = None  #the function that will draw
        self.glAnimationFrameValues = None  #animation control params to send to the callbaxk
        self.glAnimationCurrentFrame = 0  #where are we right now
        self.glAnimationPlayListLength = 0
        self.glAnimationDelayTime = 0  #delay between frames (in seconds)
        self.glAnimationRepeat = False  # repeat when the end is reached?
        self.glAnimationReverse = False  #reverse when the end is reached?
        self.glAnimationReversed = False  #are we currently moving in reverse?
        self.glAnimationReset = True #reset the animation at the end
        self.glRestartDraggingCallback = None

        #Mouse Interaction Data
        self.glMouseTextBox = None
        self.glDragList = None
        self.glDragListIndex = -1
        self.glDragMaxDist = None
        self.glDragCallback = None
        self.glDraggingActive = False
        self.glDraggingHandleSize = 0.05
        self.glDraggingHandleWidth = 2
        self.glDraggingHandleColor = [1,1,1]
        self.glWheelZoom = False
        self.rawMouseX = 0
        self.rawMouseY = 0

        if self.glWindowType == "PyQt":
            self.glWindow.initializeGL = self.glInit  # initialize callback
            self.glWindow.paintGL = self.paintGL  # paint callback
        elif self.glWindowType == "glfw":
            glfw_make_context_current(self.glWindow)
            glfw_set_window_size_callback(self.glWindow, self.glfwSizeChange)
            glfw_set_scroll_callback(self.glWindow, self.glwfWheelZoom)
            glfw_set_cursor_pos_callback(self.glWindow, self.glfwCursorPos)
            glfw_set_mouse_button_callback(self.glWindow, self.glfwMouseButton)
            self.glInit()

        self.glUpdateMatrices()


    def glInit(self):
        glutInit(sys.argv)

    def glUpdate(self):
        self.glUpdateMatrices()
        if self.glWindowType == "PyQt":
            self.glWindow.update()
        if self.glWindowType == "glfw":
            self.paintGL()
            glfw_swap_buffers(self.glWindow)
            glfw_poll_events()
            if glfw_window_should_close(self.glWindow):
                if self.glAnimationIsRunning:
                    self.glStopAnimation()

    def glWait(self):
        if self.glWindowType == "glfw":
            while not glfw_window_should_close(self.glWindow):
                glfw_poll_events()
            glfw_terminate()

    def glfwMouseButton(self, window, button, action, mods):
        self.glUpdateMatrices()
        # reset to original zoom and rotation ... at next update
        if button == 1 and action == 1: # right click to reset translate, zoom and rotate
            self.glWheelZoom = False
            self.glZoomval = 1
            self.glzoomX = 0
            self.glzoomY = 0
            self.glTranslateActive = False
            self.glTranslateXref = 0
            self.glTranslateYref = 0
            self.glTranslateXdist = 0
            self.glTranslateYdist = 0
            self.glViewReady = False
            self.glUpdate()
        # endif button

    def glfwCursorPos(self, window, xpos, ypos):
        self.rawMouseX = xpos
        self.rawMouseY = ypos


    def glwfWheelZoom(self, window, xoffset, yoffset):
        #modify Zoom values to be implemented at the next update
        self.glUpdateMatrices()
        self.glZoomX, self.glZoomY = self.glUnProjectMouse(self.rawMouseX, self.rawMouseY)
        if yoffset > 0:
            self.glZoomval /= 1.25
        else:
            self.glZoomval *= 1.25
        self.glViewReady = False
        self.glWheelZoom = True
        self.glUpdate()

    def glfwSizeChange(self, widow, width, height):
        self.glViewReady = False
        self.glUpdate()


    def glStartDragging(self,dragCallback, dragList, dragMaxDist,
                        handlesize = 0.05, handlewidth = 3, handlecolor = [1,1,1]):
        self.glDragCallback = dragCallback
        self.glDragList = dragList
        self.glDragMaxDist = dragMaxDist
        self.glDragListIndex = -1
        self.glDraggingActive = True
        self.glDraggingHandleSize = handlesize
        self.glDraggingHandleWidth = handlewidth
        self.glDraggingHandleColor = handlecolor

        self.glUpdate()


    def glStopDragging(self):
        self.glDraggingActive = False
        self.glUpdate()

    def glDraggingMouseMove(self,x,y,leftButtonDown):
        if self.glDraggingActive is False: return
        if leftButtonDown and (self.glDragListIndex > -1):
            self.glDragCallback(x, y, self.glDragList, self.glDragListIndex)  # trigger the callback
            self.glUpdate()
        else:
            self.glDragListIndex = self.closestPoint(x,y,self.glDragList, self.glDragMaxDist)
        #endif
        #end method


    def glDraggingMouseButtonPress(self,x,y,):
        if self.glDraggingActive is False: return  #not dragging

        index = self.closestPoint(x,y,self.glDragList, self.glDragMaxDist)
        self.glDragListIndex = index  # remember the point index

        if index > -1: #we found a point that was close enough
            self.glDragCallback(x, y, self.glDragList, self.glDragListIndex)  # trigger the callback
        # end function


    def glDraggingMouseButtonRelease(self,x,y,):
        if self.glDraggingActive is False: return  #not dragging
        self.glDragCallback(x, y, self.glDragList, self.glDragListIndex)  # trigger the callback
        #end function

    def glDraggingShowHandles(self):
        if self.glDraggingActive is False: return  #not dragging
        dl = self.glDragList
        hs = self.glDraggingHandleSize
        hc = self.glDraggingHandleColor
        glColor3f(hc[0],hc[1],hc[2])
        glLineWidth = self.glDraggingHandleWidth
        for i in range(len(dl)):
            if i ==self.glDragListIndex:
                gl2DCircle(dl[i][0], dl[i][1], hs, fill=True, faces = 4)
            else:
                gl2DCircle(dl[i][0], dl[i][1], hs, fill = False, faces =  4)

        #end function

    def closestPoint(self,x,y,pointlist,maxdist):

        distmaxsq = maxdist
        mindistsq = 999999999
        index = -1
        for i in range(len(pointlist)):
            distsq = (x-pointlist[i][0])**2 + (y-pointlist[i][1])**2
            if distsq < mindistsq: # a candidate
                mindistsq = distsq
                if mindistsq < distmaxsq:
                    index = i
        return index

    def glRedraw(self):
        bc = self.glBackgroundColor
        glClearColor(bc[0], bc[1], bc[2], 0)  # set the background color
        glClear(GL_COLOR_BUFFER_BIT)  # clear the drawing
        self.drawCallback()  # draw the user's drawing
        glfw_swap_buffers(self.glWindow)



    def glZoom(self, zoom, xcenter=None, ycenter=None):
        # zoom the image about the chosen center
        if zoom is None: return self.glZoomval  # return the zoom factor
        self.glZoomval = zoom
        if xcenter is not None: self.glZoomX = xcenter
        if ycenter is not None: self.glZoomY = ycenter
        self.glViewReady = False


    def glRotate(self, angle=None, xcenter=None, ycenter=None):
        # rotate the image about the chosen center
        if angle is None: return self.glRotateval  # return the rotation angle
        self.glRotateval = angle
        if xcenter is not None: self.glRotX = xcenter
        if ycenter is not None: self.glRotY = ycenter
        self.glViewReady = False


    def setViewSize(self, xmin, xmax, ymin, ymax, allowDistortion=False):
        self.glXmin = xmin
        self.glXmax = xmax
        self.glYmin = ymin
        self.glYmax = ymax
        self.glWidth = xmax - xmin  # Width of the object to be drawn
        self.glHeight = ymax - ymin  # Height of the object to be drawn
        self.glXcenter = (xmax + xmin) / 2  # X-location of the center of the object to be drawn
        self.glYcenter = (ymax + ymin) / 2  # Y-location of the center of the object to be drawn

        self.allowDistortion = allowDistortion
        self.glViewReady = False
        self.glUpdate()

    def setupGLviewing(self):
        if self.glViewReady is True:  return  # nothing to do

        # setup the drawing window size and scaling
        if self.glWindowType == "PyQt":
            windowWidth = self.glWindow.frameSize().width()
            windowHeight = self.glWindow.frameSize().height()
        elif self.glWindowType == "glfw":
            windowWidth, windowHeight = glfw_get_window_size(self.glWindow)


        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        top = self.glYmax
        bottom = self.glYmin
        right = self.glXmax
        left = self.glXmin

        if self.allowDistortion == False:  # force no shape distortion
            windowShape = windowWidth / windowHeight
            drawingShape = self.glWidth / self.glHeight
            if drawingShape > windowShape:
                newheight = self.glHeight * drawingShape / windowShape
                top = (top + bottom) / 2 + newheight / 2
                bottom = top - newheight
            else:
                newwidth = self.glWidth * windowShape / drawingShape
                right = (right + left) / 2 + newwidth / 2
                left = right - newwidth

        glViewport(1, 1, windowWidth - 1, windowHeight - 1)
        glOrtho(left, right, bottom, top, -1, 1)  # simple 2D projection

        if self.glWheelZoom:
            self.glUpdateMatrices()
            x, y = self.glUnProjectMouse(self.rawMouseX, self.rawMouseY)
            dx,dy =  x - self.glZoomX, y - self.glZoomY
            glTranslatef(dx, dy, 0)
            #self.glWheelZoom = False

        glTranslatef(self.glZoomX, self.glZoomY, 0)
        glScalef(self.glZoomval, self.glZoomval, 1)
        glTranslatef(-self.glZoomX, -self.glZoomY, 0)

        # rotate about the zoom centers
        glTranslatef(self.glRotX, self.glRotY, 0)
        glRotatef(self.glRotateval, 0, 0, 1)
        glTranslatef(-self.glRotX, -self.glRotY, 0)

        # save the transformation matrices to make mouse tracking faster
        self.glUpdateMatrices()
        self.glViewReady = True  # all done ... until things change


    def paintGL(self):
        # this is a fairly generic widow setup code for 2D graphics
        # the specific drawing code should be placed in the drawCallback() function
        # drawCallback() is called on the last line of this function
        self.setupGLviewing()  # what it says!
        bc = self.glBackgroundColor
        glClearColor(bc[0], bc[1], bc[2], 0)  # set the background color
        glClear(GL_COLOR_BUFFER_BIT)  # clear the drawing

        self.drawCallback()  # draw the user's drawing

    def glUpdateMatrices(self):
        self.glModel = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.glProjection = glGetDoublev(GL_PROJECTION_MATRIX)
        self.glView = glGetIntegerv(GL_VIEWPORT)


    def glUnProjectMouse(self, wx, wy):
        vx = GLdouble(wx)
        vy = self.glView[3] - GLdouble(wy)
        vz = GLdouble(0)
        x, y, z = gluUnProject(vx, vy, vz, model=self.glModel, proj=self.glProjection, view=self.glView)
        return x, y

    def glHandleMouseEvents(self,event):
        type = event.type()
        if type in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            pos = event.pos()
            x, y = self.glUnProjectMouse(pos.x(), pos.y())
        else:
            return

        if (type == QEvent.MouseMove):  # to read the mouse location
            if self.glMouseTextBox is not None:
                self.glMouseTextBox.setText("{:.2f}".format(x) + ",  {:.2f}".format(y))
            if event.buttons() & Qt.LeftButton:
                leftButtonDown = True
            else:
                leftButtonDown = False

            if self.glDraggingActive:  self.glDraggingMouseMove(x, y, leftButtonDown)

        elif (type == QEvent.MouseButtonPress):  # to read the mouse location
            if self.glDraggingActive:
                self.glDraggingMouseButtonPress(x,y)
            #print("mouse was clicked at ",x,y)

        elif (type == QEvent.MouseButtonRelease):  # to read the mouse location
            if self.glDraggingActive:  self.glDraggingMouseButtonRelease(x,y)
            #print("mouse was released at ",x,y)
        self.glDraggingShowHandles()
        self.glUpdate()


    def glMouseDisplayTextBox(self, textbox):
        self.glMouseTextBox = textbox


    #def glEnableMouseInteraction(self):

    def glStartAnimation(self, drawfunc, nframes,
                         delaytime=0, repeat=False, reverse=False, reset = True,
                         RestartDraggingCallback = None):

        if self.glAnimationIsRunning is True:  return  # don't want multiple copies

        # save the Animation parameters
        self.glAnimationCallback = drawfunc
        self.glAnimationNFrames = nframes
        self.glAnimationFrameValues = np.linspace(0,nframes-1,nframes)
        self.glAnimationCurrentFrame = 0
        self.glAnimationDelayTime = delaytime
        self.glAnimationRepeat = repeat
        self.glAnimationReverse = reverse
        self.glAnimationReset = reset
        self.glAnimationReversed = False


        #handle Dragging interaction with animation
        if self.glDraggingActive is True:
            self.glRestartDraggingCallback = RestartDraggingCallback
        else:
            self.glRestartDraggingCallback = None

        if self.glRestartDraggingCallback is not None:
            self.glRestartDraggingCallback(False)

        # Start the animation with current parameters
        self.glAnimationIsRunning = True
        self.glAnimate()
        self.glAnimationIsRunning = False

    def glStopAnimation(self):
        if self.glAnimationCallback is None:  return
        if self.glAnimationReset is True: # reset the image to the first frame
            self.glAnimationCallback(0, self.glAnimationNFrames)  # call the callback function

        if self.glRestartDraggingCallback is not None:
            self.glRestartDraggingCallback(True)
            self.glRestartDraggingCallback = None

        self.glAnimationIsRunning = False  # animation ended
        self.glAnimationCallback = None  #

        self.glUpdate()

    def glPauseResumeAnimation(self):

        if self.glAnimationCallback is None:  return

        # Start the animation with current parameters
        if self.glAnimationIsRunning is True:
            self.glAnimationIsRunning = False
        else:
            self.glAnimationIsRunning = True
            self.glAnimate()

    def glAnimate(self):

        # use shorter names ... less typing
        maxIndex = self.glAnimationNFrames
        repeat = self.glAnimationRepeat
        reverse = self.glAnimationReverse

        if self.glAnimationReversed is True:
            step = -1
            theEnd = -1
        else:
            step = 1
            theEnd = maxIndex

        while self.glAnimationCurrentFrame != theEnd:

            if self.glAnimationIsRunning is False:
                return  # the animation was stopped


            self.glAnimationCallback(self.glAnimationCurrentFrame, self.glAnimationNFrames)  # call the callback function
            self.glUpdate()

            sleep(self.glAnimationDelayTime)  # and sleep as directed

            self.glAnimationCurrentFrame += step

            if self.glAnimationCurrentFrame == theEnd:  # at the end, what now??

                if reverse is True:
                    if self.glAnimationReversed is False:  # then move in reverse
                        step = -1
                        theEnd = -1
                        self.glAnimationCurrentFrame = maxIndex - 2  # dont repeat the last step
                        self.glAnimationReversed = True

                if repeat is True:  # want to repeat forever!
                    if reverse is False:  # then we can't be at the beginning point
                        self.glAnimationCurrentFrame = 0  # start over
                    elif reverse is True:  # animation is reversible
                        if self.glAnimationCurrentFrame == -1:  # back at the beginning
                            step = 1
                            theEnd = maxIndex
                            self.glAnimationCurrentFrame = 0
                            self.glAnimationReversed = False
        # end while loop

        self.glStopAnimation() #animation is over

#end of the GL2D class definition

# a few useful drawing functions
def gl2DText(text, x, y, font=GLUT_BITMAP_HELVETICA_18):
    glRasterPos2d(x, y)
    for ch in text:
        glutBitmapCharacter(font, ord(ch))

def gl2DCircle(xcenter, ycenter, radius, fill=False, faces=24):
    theta = 0

    if fill:
        glBegin(GL_POLYGON)
    else:
        glBegin(GL_LINE_STRIP)

    glVertex2f(xcenter + np.cos(theta) * radius, ycenter + np.sin(theta) * radius)
    for i in range(1, faces + 1):
        theta = i / faces * 2 * np.pi
        glVertex2f(xcenter + np.cos(theta) * radius, ycenter + np.sin(theta) * radius)
    glEnd();

def gl2DArc(xcenter, ycenter, radius, startDeg, stopDeg, faces=24):
    start = startDeg * np.pi/180
    delta = 1 / faces * (stopDeg - startDeg) * np.pi/180

    theta = start
    glBegin(GL_LINE_STRIP)
    glVertex2f(xcenter + np.cos(theta) * radius, ycenter + np.sin(theta) * radius);
    for i in range(1, faces + 1):
        theta +=  delta
        glVertex2f(xcenter + np.cos(theta) * radius, ycenter + np.sin(theta) * radius);
    glEnd();

def gl2DArrow(xtip, ytip, size, angleDeg = 0, widthDeg = 60, toCenter = False, fill=True):
    theta = angleDeg * np.pi/180
    delta = (180 - widthDeg) * np.pi/180
    xcenter = xtip - size * np.cos(theta)
    ycenter = ytip - size * np.sin(theta)

    if fill:
        glBegin(GL_POLYGON)
    else:
        glBegin(GL_LINE_STRIP)

    glVertex2f(xcenter + np.cos(theta) * size, ycenter + np.sin(theta) * size)
    glVertex2f(xcenter + np.cos(theta+delta) * size, ycenter + np.sin(theta+delta) * size)
    if toCenter is True:
        glVertex2f(xcenter, ycenter)
    glVertex2f(xcenter + np.cos(theta-delta) * size, ycenter + np.sin(theta-delta) * size)
    glVertex2f(xcenter + np.cos(theta) * size, ycenter + np.sin(theta) * size)

    glEnd()

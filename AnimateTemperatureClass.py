import numpy as np
from scipy.optimize import fsolve
import time

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from OpenGL_2D_class import gl2D, gl2DArrow, gl2DCircle

from HersheyFont import HersheyFont
hf = HersheyFont()



class TemperatureAnimator():
    def __init__(self):
        self.title = None
        self.Temperatures = []
        self.rowSize = 0
        self.nRows = 0
        self.thisRow = 0
        self.spacing = 10
        self.tMin = None
        self.tMax= None

        #required by the animator
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.allowDistortion = False
        self.numberOfAnimationFrames  = None
        self.AnimDelayTime = 0.05
        self.AnimReverse = False
        self.AnimRepeat = False
        self.AnimReset = False


    def ProcessFileData(self, data):
        # from the array of strings, fill the wing dictionary

        for line in data:  # loop over all the lines
            cells = line.strip().replace('(','').replace(')','').split(',')
            keyword = cells[0].strip().lower()

            if keyword == 'title': self.title = cells[1].replace("'", "")

            if keyword == 'range':
                self.tMin = float(cells[1])
                self.tMax = float(cells[2])

            if keyword == 'row':
                temps = []
                for i in range(1,len(cells)): #loop over all temperatures on the row
                    temps.append(float(cells[i]))
                self.Temperatures.append(temps)

        self.ConnectData()


    def ConnectData(self):
        self.rowSize = len(self.Temperatures[0])
        self.nRows = len(self.Temperatures)
        self.numberOfAnimationFrames = len(self.Temperatures) - 1
        self.thisRow = 0 #for the animation frame on startup

        #required for the animator
        self.xmin = -self.spacing * 2
        self.xmax = self.rowSize * self.spacing * 1.05
        self.ymin = -self.spacing
        self.ymax = self.spacing * 6


    def DrawPicture(self):
        # this is what actually draws the picture
        # using data to control what is drawn

        temps = self.Temperatures[self.thisRow]


        size = self.xmax / 8
        # draw the numbers

        for i in range(self.rowSize):  #draw my temperatere at nodes
            colors = temperature_to_rgb(temps[i],self.tMin,self.tMax)
            glColor3f(*colors)  #
            gl2DCircle(i*self.spacing, self.xmin, self.spacing/3,fill=True)

        for i in range(self.rowSize): #draw the temperature scale
            t = float(i) / self.rowSize * (self.tMax - self.tMin)    + self.tMin
            colors = temperature_to_rgb(t, self.tMin, self.tMax)
            glColor3f(*colors)  #
            gl2DCircle(i * self.spacing, self.ymax/2, self.spacing / 1.0, fill=True)

            glColor3f(1,1,1)

            hf.drawText("Temperature Scale",(self.xmax + self.xmin)/2, self.ymax*0.9,
                        center = True, scale=self.spacing*1.5,weight=2)
            hf.drawText(f"{self.tMin:.2f}", self.xmin + self.spacing*1, self.ymax * 0.7,
                                center=False, scale=self.spacing ,weight=2)
            hf.drawText(f"{self.tMax:.2f}", self.xmax-self.spacing*3, self.ymax * 0.7, weight=2,
                                center=False, scale=self.spacing, justify=1 )


    def PrepareNextAnimationFrameData(self, frame, nframes):
        self.thisRow = frame


def temperature_to_rgb(temp, t_min, t_max):
    """
    Maps a temperature value to an RGB color.
    Blue = t_min, Green = middle, Red = t_max
    """
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")

    # Normalize temperature to range [0, 1]
    t_norm = (temp - t_min) / (t_max - t_min)

    if t_norm <= 0.25:
        # Blue (0,0,1) → Cyan (0,1,1)
        ratio = t_norm / 0.25
        r = 0.0
        g = ratio
        b = 1.0

    elif t_norm <= 0.5:
        # Cyan (0,1,1) → Green (0,1,0)
        ratio = (t_norm - 0.25) / 0.25
        r = 0.0
        g = 1.0
        b = 1.0 - ratio

    elif t_norm <= 0.75:
        # Green (0,1,0) → Yellow (1,1,0)
        ratio = (t_norm - 0.5) / 0.25
        r = ratio
        g = 1.0
        b = 0.0

    else:
        # Yellow (1,1,0) → Red (1,0,0)
        ratio = (t_norm - 0.75) / 0.25
        r = 1.0
        g = 1.0 - ratio
        b = 0.0

    return (r, g, b)



# standard PyQt5 imports
from PyQt5.QtWidgets import QDialog, QApplication

# standard OpenGL imports
from OpenGL.GLUT import *

from OpenGL_2D_class_GLFW_old import gl2D

# the ui created by Designer and pyuic
from Simple_GL_ui import Ui_Dialog

# import the Problem Specific class
from SimpleDrawingClass import SimpleDrawing


class main_window(QDialog):
    def __init__(self):
        super(main_window, self).__init__()
        self.ui = Ui_Dialog()
        # setup the GUI
        self.ui.setupUi(self)

        # define any data (including object variables) your program might need
        self.mydrawing = SimpleDrawing()

        # create and setup the GL window object
        self.setupGLWindows()

        # and define any Widget callbacks (buttons, etc) or other necessary setup
        self.assign_widgets()

        # show the GUI
        self.show()

    def assign_widgets(self):  # callbacks for Widgets on your GUI
        self.ui.pushButton_Exit.clicked.connect(self.ExitApp)
        self.ui.pushButton_start.clicked.connect(self.StartAnimation)
        self.ui.horizontalSlider_zoom.valueChanged.connect(self.glZoomSlider)

    # Widget callbacks start here

    def glZoomSlider(self):  # I used a slider to control GL zooming
        zoomval = float((self.ui.horizontalSlider_zoom.value()) / 200 + 0.25)
        self.glwindow1.glZoom(zoomval)  # set the zoom value
        self.glwindow1.glUpdate()  # update the GL image


    def ExitApp(self):
        app.exit()

    # Setup OpenGL Drawing and Viewing
    def setupGLWindows(self):  # setup all GL windows
        # send it the   GL Widget     and the drawing Callback function
        self.glwindow1 = gl2D(self.ui.openGLWidget, self.DrawingCallback)

        # set the drawing space:    xmin  xmax  ymin   ymax
        self.glwindow1.setViewSize(-1, 6, -0.5, 6, allowDistortion=False)


    def DrawingCallback(self):
        # this is what actually draws the picture
        self.mydrawing.DrawPicture(True)  # drawing is done by the DroneCatcher object

    def StartAnimation(self):  # a button to start GL Animation
        nframes = 60
        self.glwindow1.glStartAnimation(self.AnimationCallback, nframes,delaytime=0.1,
                                       reverse=False, repeat=False, reset=True)

    def AnimationCallback(self, frame, nframes):
        # calculations needed to configure the picture
        # these could be done here or by calling a class method

        d = self.mydrawing # very useful shorthand!
        d.giantX = d.giantStart + (d.giantStop-d.giantStart)*frame/nframes
        # the next line is absolutely required for pause, resume, stop, etc !!!
        app.processEvents()



if __name__ == "__main__":
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    main_win = main_window()
    sys.exit(app.exec_())

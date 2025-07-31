

# PyQt5 imports
from PyQt5.QtWidgets import QDialog, QApplication

# OpenGL imports
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL_2D_class_GLFW_old import gl2D
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle
from HersheyFont import HersheyFont
hf = HersheyFont()
# your generated QtDesigner UI
from Simple_GL_ui import Ui_Dialog


from Butte_solver_for_animation import System

hf = HersheyFont()


class main_window(QDialog):
    def __init__(self):
        super(main_window, self).__init__()
        # --- setup the Qt UI ---
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        #
        self.system = System()
        self.system.ReadNetworkData()
        self.system.createRadialNodeLocations()
        self.system.createAxialNodeLocations()
        self.system.createNodes()
        self.system.linkNodes()
        self.temps = self.system.runTransient(24, dt=3600.0)
        self.current_hour = 0



        self.setupGLWindows()
        self.assign_widgets()

        self.show()

    def assign_widgets(self):
        self.ui.pushButton_Exit.clicked.connect(self.ExitApp)
        self.ui.pushButton_start.clicked.connect(self.StartAnimation)
        self.ui.horizontalSlider_zoom.valueChanged.connect(self.glZoomSlider)

    def glZoomSlider(self):
        zoomval = float(self.ui.horizontalSlider_zoom.value() / 200 + 0.25)
        self.glwindow1.glZoom(zoomval)
        self.glwindow1.glUpdate()

    def ExitApp(self):
        QApplication.exit()

    def setupGLWindows(self):
        # send your OpenGLWidget and the draw‑callback
        self.glwindow1 = gl2D(self.ui.openGLWidget, self.DrawingCallback)




    def DrawingCallback(self):
        self.system.draw_selected_nodes(self)






    def StartAnimation(self):
        # one frame per saved hour (0–24)
        nframes = self.temps.shape[0]
        self.glwindow1.glStartAnimation(
            self.AnimationCallback,
            nframes,
            delaytime=0.5,
            reverse=False,
            repeat=False,
            reset=True
        )

    def AnimationCallback(self, frame, nframes):
        # update which hour to display
        self.current_hour = frame
        self.current_temps = self.temps[self.current_hour]

        # redraw
        self.glwindow1.glUpdate()

        # process Qt events so you can pause/stop
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    main_win = main_window()
    sys.exit(app.exec_())

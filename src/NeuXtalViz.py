import os
import sys
import traceback
import subprocess

os.environ["QT_API"] = "pyqt5"

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QAction,
    QStackedWidget,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
)

from qtpy.QtGui import QIcon

from NeuXtalViz._version import __version__

import pyvista

pyvista.set_plot_theme("document")

import qdarktheme

qdarktheme.enable_hi_dpi()

# import qdarkstyle
# from qdarkstyle.light.palette import LightPalette

from NeuXtalViz.views.crystal_structure_tools import CrystalStructureView
from NeuXtalViz.models.crystal_structure_tools import CrystalStructureModel
from NeuXtalViz.presenters.crystal_structure_tools import CrystalStructure

from NeuXtalViz.views.ub_tools import UBView
from NeuXtalViz.models.ub_tools import UBModel
from NeuXtalViz.presenters.ub_tools import UB

from NeuXtalViz.views.sample_tools import SampleView
from NeuXtalViz.models.sample_tools import SampleModel
from NeuXtalViz.presenters.sample_tools import Sample

from NeuXtalViz.views.volume_slicer import VolumeSlicerView
from NeuXtalViz.models.volume_slicer import VolumeSlicerModel
from NeuXtalViz.presenters.volume_slicer import VolumeSlicer

from NeuXtalViz.views.experiment_planner import ExperimentView
from NeuXtalViz.models.experiment_planner import ExperimentModel
from NeuXtalViz.presenters.experiment_planner import Experiment


class NeuXtalViz(QMainWindow):
    __instance = None

    def __new__(cls):
        if NeuXtalViz.__instance is None:
            NeuXtalViz.__instance = QMainWindow.__new__(cls)
        return NeuXtalViz.__instance

    def __init__(self, parent=None):
        super().__init__(parent)

        self._topaz_path = "/SNS/TOPAZ"

        icon = os.path.join(os.path.dirname(__file__), "icons/neuxtalviz.png")
        self.setWindowIcon(QIcon(icon))
        self.setWindowTitle("NeuXtalViz {}".format(__version__))
        # self.resize(1200, 900)

        main_window = QWidget(self)
        self.setCentralWidget(main_window)

        layout = QVBoxLayout(main_window)

        app_stack = QStackedWidget()

        app_menu = self.menuBar().addMenu("Applications")

        cs_action = QAction("Crystal Structure", self)
        cs_action.triggered.connect(lambda: app_stack.setCurrentIndex(0))
        app_menu.addAction(cs_action)

        s_action = QAction("Sample", self)
        s_action.triggered.connect(lambda: app_stack.setCurrentIndex(1))
        app_menu.addAction(s_action)

        vs_action = QAction("Volume Slicer", self)
        vs_action.triggered.connect(lambda: app_stack.setCurrentIndex(2))
        app_menu.addAction(vs_action)

        cs_view = CrystalStructureView(self)
        cs_model = CrystalStructureModel()
        self.cs = CrystalStructure(cs_view, cs_model)
        app_stack.addWidget(cs_view)

        s_view = SampleView(self)
        s_model = SampleModel()
        self.s = Sample(s_view, s_model)
        app_stack.addWidget(s_view)

        vs_view = VolumeSlicerView(self)
        vs_model = VolumeSlicerModel()
        self.vs = VolumeSlicer(vs_view, vs_model)
        app_stack.addWidget(vs_view)

        layout.addWidget(app_stack)

        ub_action = QAction("UB", self)
        ub_action.triggered.connect(lambda: app_stack.setCurrentIndex(3))
        app_menu.addAction(ub_action)

        ub_view = UBView(self)
        ub_model = UBModel()
        self.ub = UB(ub_view, ub_model)
        app_stack.addWidget(ub_view)

        ep_action = QAction("Planner", self)
        ep_action.triggered.connect(lambda: app_stack.setCurrentIndex(4))
        app_menu.addAction(ep_action)

        ep_view = ExperimentView(self)
        ep_model = ExperimentModel()
        self.ep = Experiment(ep_view, ep_model)
        app_stack.addWidget(ep_view)

        layout.addWidget(app_stack)

        app_menu = self.menuBar().addMenu("Reduction")

        topaz_action = QAction("TOPAZ", self)
        topaz_action.triggered.connect(self.topaz_reduction_GUI)
        app_menu.addAction(topaz_action)

        garnet_action = QAction("garnet", self)
        garnet_action.triggered.connect(self.garnet_reduction_GUI)
        app_menu.addAction(garnet_action)

        app_menu = self.menuBar().addMenu("Analysis")

        shelxle_action = QAction("ShelXle", self)
        shelxle_action.triggered.connect(self.shelxle_GUI)
        app_menu.addAction(shelxle_action)

        olex2_action = QAction("Olex2", self)
        olex2_action.triggered.connect(self.olex2_reduction_GUI)
        app_menu.addAction(olex2_action)

        fullprof_action = QAction("FullProf", self)
        fullprof_action.triggered.connect(self.fullprof_reduction_GUI)
        app_menu.addAction(fullprof_action)

        app_menu = self.menuBar().addMenu("Interface")

        structdiff_action = QAction("Structure/Diffuse", self)
        structdiff_action.triggered.connect(self.structdiff_GUI)
        app_menu.addAction(structdiff_action)

        # self.showMaximized()

    def topaz_reduction_GUI(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", self._topaz_path
        )

        if directory:
            self._topaz_path = directory
            main_py_path = os.path.join(directory, "main.py")
            if os.path.isfile(main_py_path):
                try:
                    subprocess.Popen(["mantidpython", main_py_path])
                except subprocess.CalledProcessError as e:
                    QMessageBox.critical(
                        self, "Error", f"Failed to execute main.py:\n{e}"
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "The selected directory does not contain main.py.",
                )

    def shelxle_GUI(self):
        try:
            subprocess.Popen(["shelxle"])
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to execute shelxle:\n{e}"
            )

    def garnet_reduction_GUI(self):
        try:
            subprocess.Popen(["/SNS/software/scd/garnet.sh"])
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to execute garnet:\n{e}"
            )

    def olex2_reduction_GUI(self):
        try:
            subprocess.Popen(["olex2"])
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to execute olex2:\n{e}"
            )

    def fullprof_reduction_GUI(self):
        try:
            subprocess.Popen(["fullprof"])
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to execute fullprof:\n{e}"
            )

    def structdiff_GUI(self):
        path = os.path.dirname(__file__)
        file = os.path.join(path, "NeuXtalViz/views/command_browser.py")
        try:
            subprocess.Popen(["python", file])
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(
                self, "Error", f"Failed to execute structdiff:\n{e}"
            )


def handle_exception(exc_type, exc_value, exc_traceback):
    error_message = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Application Error")
    msg_box.setText("An unexpected error occurred. Please see details below:")
    msg_box.setDetailedText(error_message)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.exec_()


def gui():
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("light")
    # app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))
    window = NeuXtalViz()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui()

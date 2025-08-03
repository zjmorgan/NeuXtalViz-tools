from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QTabWidget,
    QComboBox,
    QLineEdit,
    QSlider,
    QFileDialog,
    QCheckBox,
)

from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Qt
from PyQt5.QtCore import pyqtSignal

import numpy as np
import pyvista as pv

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D

from NeuXtalViz.views.base_view import NeuXtalVizWidget
from NeuXtalViz.config import colormap

colormap.add_modified()

cmaps = {
    "Sequential": "viridis",
    "Binary": "binary",
    "Diverging": "bwr",
    "Rainbow": "turbo",
    "Modified": "modified",
}

opacities = {
    "Linear": {"Low->High": "linear", "High->Low": "linear_r"},
    "Geometric": {"Low->High": "geom", "High->Low": "geom_r"},
    "Sigmoid": {"Low->High": "sigmoid", "High->Low": "sigmoid_r"},
}


class VolumeSlicerView(NeuXtalVizWidget):
    slice_ready = pyqtSignal()
    cut_ready = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setToolTip(
            "Switch between different volume slicing tools and views."
        )

        self.slicer_tab()

        self.layout().addWidget(self.tab_widget, stretch=1)

    def slicer_tab(self):
        slice_tab = QWidget()
        self.tab_widget.addTab(slice_tab, "Slicer")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.container = QWidget()
        self.container.setVisible(False)

        plots_layout = QVBoxLayout()
        slice_params_layout = QHBoxLayout()
        view_params_layout = QHBoxLayout()
        collabsible_layout = QVBoxLayout(self.container)
        cut_params_layout = QHBoxLayout()
        draw_layout = QHBoxLayout()

        self.vol_scale_combo = QComboBox(self)
        self.vol_scale_combo.addItem("Linear")
        self.vol_scale_combo.addItem("Log")
        self.vol_scale_combo.setCurrentIndex(0)
        self.vol_scale_combo.setToolTip(
            "Select the scaling for the volume data (Linear or Logarithmic)."
        )

        self.opacity_combo = QComboBox(self)
        self.opacity_combo.addItem("Linear")
        self.opacity_combo.addItem("Geometric")
        self.opacity_combo.addItem("Sigmoid")
        self.opacity_combo.setCurrentIndex(0)
        self.opacity_combo.setToolTip(
            "Choose the opacity mapping for the volume rendering."
        )

        self.range_combo = QComboBox(self)
        self.range_combo.addItem("Low->High")
        self.range_combo.addItem("High->Low")
        self.range_combo.setCurrentIndex(0)
        self.range_combo.setToolTip(
            "Set the direction of the opacity or color range."
        )

        self.clim_combo = QComboBox(self)
        self.clim_combo.addItem("Min/Max")
        self.clim_combo.addItem("μ±3×σ")
        self.clim_combo.addItem("Q₃/Q₁±1.5×IQR")
        self.clim_combo.setCurrentIndex(2)
        self.clim_combo.setToolTip(
            "Choose the method for setting color limits for the slice."
        )

        self.vlim_combo = QComboBox(self)
        self.vlim_combo.addItem("Min/Max")
        self.vlim_combo.addItem("μ±3×σ")
        self.vlim_combo.addItem("Q₃/Q₁±1.5×IQR")
        self.vlim_combo.setCurrentIndex(2)
        self.vlim_combo.setToolTip(
            "Choose the method for setting value limits for the cut."
        )

        self.cbar_combo = QComboBox(self)
        self.cbar_combo.addItem("Sequential")
        self.cbar_combo.addItem("Rainbow")
        self.cbar_combo.addItem("Binary")
        self.cbar_combo.addItem("Diverging")
        self.cbar_combo.addItem("Modified")
        self.cbar_combo.setToolTip(
            "Select the colormap for the slice visualization."
        )

        self.load_NXS_button = QPushButton("Load NXS", self)
        self.load_NXS_button.setToolTip(
            "Load a NeXus (NXS) file for volume slicing."
        )

        draw_layout.addWidget(self.vol_scale_combo)
        draw_layout.addWidget(self.opacity_combo)
        draw_layout.addWidget(self.range_combo)
        draw_layout.addWidget(self.clim_combo)
        draw_layout.addWidget(self.cbar_combo)
        draw_layout.addWidget(self.load_NXS_button)

        self.slice_combo = QComboBox(self)
        self.slice_combo.addItem("Axis 1/2")
        self.slice_combo.addItem("Axis 1/3")
        self.slice_combo.addItem("Axis 2/3")
        self.slice_combo.setCurrentIndex(0)
        self.slice_combo.setToolTip("Select the plane for slicing the volume.")

        self.cut_combo = QComboBox(self)
        self.cut_combo.addItem("Axis 1")
        self.cut_combo.addItem("Axis 2")
        self.cut_combo.setCurrentIndex(0)
        self.cut_combo.setToolTip(
            "Select the axis for cutting through the slice."
        )

        slice_label = QLabel("Slice:", self)
        cut_label = QLabel("Cut:", self)

        self.slice_line = QLineEdit("0.0")
        self.slice_line.setValidator(validator)
        self.slice_line.setToolTip(
            "Set the position of the slice along the selected plane."
        )

        self.cut_line = QLineEdit("0.0")
        self.cut_line.setValidator(validator)
        self.cut_line.setToolTip(
            "Set the position of the cut along the selected axis."
        )

        validator = QDoubleValidator(0.0001, 100, 5, notation=notation)

        slice_thickness_label = QLabel("Thickness:", self)
        cut_thickness_label = QLabel("Thickness:", self)

        self.slice_thickness_line = QLineEdit("0.1")
        self.slice_thickness_line.setValidator(validator)
        self.slice_thickness_line.setToolTip("Set the thickness of the slice.")
        self.cut_thickness_line = QLineEdit("0.5")
        self.cut_thickness_line.setValidator(validator)
        self.cut_thickness_line.setToolTip("Set the thickness of the cut.")

        self.slice_scale_combo = QComboBox(self)
        self.slice_scale_combo.addItem("Linear")
        self.slice_scale_combo.addItem("Log")
        self.slice_scale_combo.setToolTip(
            "Select the scale for the slice plot (Linear or Logarithmic)."
        )

        self.cut_scale_combo = QComboBox(self)
        self.cut_scale_combo.addItem("Linear")
        self.cut_scale_combo.addItem("Log")
        self.cut_scale_combo.setToolTip(
            "Select the scale for the cut plot (Linear or Logarithmic)."
        )

        slider_layout = QVBoxLayout()
        bar_layout = QHBoxLayout()

        self.min_slider = QSlider(Qt.Vertical)
        self.max_slider = QSlider(Qt.Vertical)
        self.min_slider.setRange(0, 100)
        self.max_slider.setRange(0, 100)
        self.min_slider.setValue(0)
        self.max_slider.setValue(100)
        self.min_slider.setTracking(False)
        self.max_slider.setTracking(False)
        self.min_slider.setToolTip(
            "Adjust the minimum value for the colorbar range."
        )
        self.max_slider.setToolTip(
            "Adjust the maximum value for the colorbar range."
        )

        self.vmin_line = QLineEdit("")
        self.vmax_line = QLineEdit("")

        self.xmin_line = QLineEdit("")
        self.xmax_line = QLineEdit("")

        self.ymin_line = QLineEdit("")
        self.ymax_line = QLineEdit("")

        validator = QDoubleValidator(-1e32, 1e32, 6, notation=notation)

        self.vmin_line.setValidator(validator)
        self.vmax_line.setValidator(validator)
        self.vmin_line.setToolTip("Set the minimum value for the colorbar.")
        self.vmax_line.setToolTip("Set the maximum value for the colorbar.")

        self.xmin_line.setValidator(validator)
        self.xmax_line.setValidator(validator)
        self.xmin_line.setToolTip("Set the minimum value for the X axis.")
        self.xmax_line.setToolTip("Set the maximum value for the X axis.")

        self.ymin_line.setValidator(validator)
        self.ymax_line.setValidator(validator)
        self.ymin_line.setToolTip("Set the minimum value for the Y axis.")
        self.ymax_line.setToolTip("Set the maximum value for the Y axis.")

        xmin_label = QLabel("X Min:", self)
        xmax_label = QLabel("X Max:", self)

        ymin_label = QLabel("Y Min:", self)
        ymax_label = QLabel("Y Max:", self)

        vmin_label = QLabel("Min:", self)
        vmax_label = QLabel("Max:", self)

        bar_layout.addWidget(self.min_slider)
        bar_layout.addWidget(self.max_slider)

        self.save_slice_button = QPushButton("Save Slice", self)
        self.save_slice_button.setToolTip(
            "Save the current slice as a CSV file."
        )
        self.save_cut_button = QPushButton("Save Cut", self)
        self.save_cut_button.setToolTip("Save the current cut as a CSV file.")

        self.toggle_line_box = QCheckBox("Show Line Cut")
        self.toggle_line_box.setChecked(False)
        self.toggle_line_box.setToolTip(
            "Show or hide the line cut overlay on the slice plot."
        )

        slider_layout.addLayout(bar_layout)

        slice_params_layout.addWidget(self.slice_combo)
        slice_params_layout.addWidget(slice_label)
        slice_params_layout.addWidget(self.slice_line)
        slice_params_layout.addWidget(slice_thickness_label)
        slice_params_layout.addWidget(self.slice_thickness_line)
        slice_params_layout.addWidget(self.toggle_line_box)
        slice_params_layout.addStretch(1)
        slice_params_layout.addWidget(self.save_slice_button)
        slice_params_layout.addWidget(self.slice_scale_combo)

        view_params_layout.addWidget(xmin_label)
        view_params_layout.addWidget(self.xmin_line)
        view_params_layout.addWidget(xmax_label)
        view_params_layout.addWidget(self.xmax_line)
        view_params_layout.addWidget(ymin_label)
        view_params_layout.addWidget(self.ymin_line)
        view_params_layout.addWidget(ymax_label)
        view_params_layout.addWidget(self.ymax_line)
        view_params_layout.addStretch(1)
        view_params_layout.addWidget(self.vlim_combo)
        view_params_layout.addWidget(vmin_label)
        view_params_layout.addWidget(self.vmin_line)
        view_params_layout.addWidget(vmax_label)
        view_params_layout.addWidget(self.vmax_line)

        cut_params_layout.addWidget(self.cut_combo)
        cut_params_layout.addWidget(cut_label)
        cut_params_layout.addWidget(self.cut_line)
        cut_params_layout.addWidget(cut_thickness_label)
        cut_params_layout.addWidget(self.cut_thickness_line)
        cut_params_layout.addStretch(1)
        cut_params_layout.addWidget(self.save_cut_button)
        cut_params_layout.addWidget(self.cut_scale_combo)

        plots_layout.addLayout(draw_layout)

        self.canvas_slice = FigureCanvas(Figure(constrained_layout=True))
        self.canvas_cut = FigureCanvas(
            Figure(constrained_layout=True, figsize=(6.4, 3.2))
        )

        image_layout = QHBoxLayout()
        line_layout = QHBoxLayout()

        fig_2d_layout = QVBoxLayout()
        fig_1d_layout = QVBoxLayout()

        fig_2d_layout.addWidget(NavigationToolbar2QT(self.canvas_slice, self))
        fig_2d_layout.addWidget(self.canvas_slice)

        fig_1d_layout.addWidget(NavigationToolbar2QT(self.canvas_cut, self))
        fig_1d_layout.addWidget(self.canvas_cut)

        image_layout.addLayout(fig_2d_layout)
        image_layout.addLayout(slider_layout)

        line_layout.addLayout(fig_1d_layout)

        plots_layout.addLayout(image_layout)
        plots_layout.addLayout(slice_params_layout)
        plots_layout.addLayout(view_params_layout)

        collabsible_layout.addLayout(line_layout)
        collabsible_layout.addLayout(cut_params_layout)

        plots_layout.addWidget(self.container)

        self.fig_slice = self.canvas_slice.figure
        self.fig_cut = self.canvas_cut.figure

        self.ax_slice = self.fig_slice.subplots(1, 1)
        self.ax_cut = self.fig_cut.subplots(1, 1)

        self.cb = None

        slice_tab.setLayout(plots_layout)

        self.toggle_line_box.toggled.connect(self.toggle_container)

    def toggle_container(self, state):
        self.container.setVisible(state)
        self.update_lines(state)

    def connect_save_slice(self, save_slice):
        self.save_slice_button.clicked.connect(save_slice)

    def connect_save_cut(self, save_cut):
        self.save_cut_button.clicked.connect(save_cut)

    def connect_vol_scale_combo(self, update_vol):
        self.vol_scale_combo.currentIndexChanged.connect(update_vol)

    def connect_opacity_combo(self, update_opacity):
        self.opacity_combo.currentIndexChanged.connect(update_opacity)

    def connect_range_combo(self, update_range):
        self.range_combo.currentIndexChanged.connect(update_range)

    def connect_clim_combo(self, update_clim):
        self.clim_combo.currentIndexChanged.connect(update_clim)

    def connect_vlim_combo(self, update_clim):
        self.vlim_combo.currentIndexChanged.connect(update_clim)

    def connect_cbar_combo(self, update_cbar):
        self.cbar_combo.currentIndexChanged.connect(update_cbar)

    def connect_slice_thickness_line(self, update_slice):
        self.slice_thickness_line.editingFinished.connect(update_slice)

    def connect_cut_thickness_line(self, update_cut):
        self.cut_thickness_line.editingFinished.connect(update_cut)

    def connect_slice_line(self, update_slice):
        self.slice_line.editingFinished.connect(update_slice)

    def connect_cut_line(self, update_cut):
        self.cut_line.editingFinished.connect(update_cut)

    def connect_slice_scale_combo(self, update_slice):
        self.slice_scale_combo.currentIndexChanged.connect(update_slice)

    def connect_cut_scale_combo(self, update_cut):
        self.cut_scale_combo.currentIndexChanged.connect(update_cut)

    def connect_slice_combo(self, update_slice):
        self.slice_combo.currentIndexChanged.connect(update_slice)

    def connect_cut_combo(self, update_cut):
        self.cut_combo.currentIndexChanged.connect(update_cut)

    def connect_min_slider(self, update_colorbar):
        self.min_slider.valueChanged.connect(update_colorbar)

    def connect_max_slider(self, update_colorbar):
        self.max_slider.valueChanged.connect(update_colorbar)

    def connect_vmin_line(self, update_vals):
        self.vmin_line.editingFinished.connect(update_vals)

    def connect_vmax_line(self, update_vals):
        self.vmax_line.editingFinished.connect(update_vals)

    def connect_xmin_line(self, update_vals):
        self.xmin_line.editingFinished.connect(update_vals)

    def connect_xmax_line(self, update_vals):
        self.xmax_line.editingFinished.connect(update_vals)

    def connect_ymin_line(self, update_vals):
        self.ymin_line.editingFinished.connect(update_vals)

    def connect_ymax_line(self, update_vals):
        self.ymax_line.editingFinished.connect(update_vals)

    def save_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self, "Save csv file", "", "CSV files (*.csv)", options=options
        )

        return filename

    def update_colorbar_min(self):
        min_val = self.min_slider.value()
        max_val = self.max_slider.value()

        if min_val >= max_val:
            self.min_slider.blockSignals(True)
            self.min_slider.setValue(max_val - 1)
            self.min_slider.blockSignals(False)

        self.update_slice_color()

    def update_colorbar_max(self):
        min_val = self.min_slider.value()
        max_val = self.max_slider.value()

        if min_val >= max_val:
            self.max_slider.blockSignals(True)
            self.max_slider.setValue(min_val + 1)
            self.max_slider.blockSignals(False)

        self.update_slice_color()

    def update_slice_color(self):
        if self.cb is not None:
            min_slider, max_slider = self.get_color_bar_values()

            vmin = self.vmin + (self.vmax - self.vmin) * min_slider / 100
            vmax = self.vmin + (self.vmax - self.vmin) * max_slider / 100

            self.update_colorbar_vlims(vmin, vmax)

    def update_colorbar_vlims(self, vmin, vmax):
        if self.cb is not None:
            self.set_vmin_value(vmin)
            self.set_vmax_value(vmax)

            self.im.set_clim(vmin=vmin, vmax=vmax)
            self.cb.update_normal(self.im)
            self.cb.minorticks_on()

            self.canvas_slice.draw_idle()
            self.canvas_slice.flush_events()

    def get_color_bar_values(self):
        return self.min_slider.value(), self.max_slider.value()

    def reset_slider(self):
        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)
        self.min_slider.setValue(0)
        self.max_slider.setValue(100)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)

    def connect_load_NXS(self, load_NXS):
        self.load_NXS_button.clicked.connect(load_NXS)

    def load_NXS_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self, "Load NXS file", "", "NXS files (*.nxs)", options=options
        )

        return filename

    def add_histo(self, histo_dict, normal, norm, value):
        opacity = opacities[self.get_opacity()][self.get_range()]

        log_scale = True if self.get_vol_scale() == "Log" else False

        cmap = cmaps[self.get_colormap()]

        self.clear_scene()

        self.norm = np.array(norm).copy()
        origin = norm
        origin[np.abs(origin).tolist().index(1)] = value

        signal = histo_dict["signal"]
        labels = histo_dict["labels"]

        min_lim = histo_dict["min_lim"]
        max_lim = histo_dict["max_lim"]
        spacing = histo_dict["spacing"]

        P = histo_dict["projection"]
        T = histo_dict["transform"]
        S = histo_dict["scales"]

        grid = pv.ImageData()

        grid.dimensions = np.array(signal.shape) + 1

        grid.origin = min_lim
        grid.spacing = spacing

        min_bnd = min_lim * S
        max_bnd = max_lim * S

        bounds = np.array([[min_bnd[i], max_bnd[i]] for i in [0, 1, 2]])
        limits = np.array([[min_lim[i], max_lim[i]] for i in [0, 1, 2]])

        a = pv._vtk.vtkMatrix3x3()
        b = pv._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                a.SetElement(i, j, T[i, j])
                b.SetElement(i, j, P[i, j])

        grid.cell_data["scalars"] = signal.flatten(order="F")

        normal /= np.linalg.norm(normal)

        origin = np.dot(P, origin)

        clim = [np.nanmin(signal), np.nanmax(signal)]

        if not np.all(np.isfinite(clim)):
            clim = [0.1, 10]

        self.clip = self.plotter.add_volume_clip_plane(
            grid,
            opacity=opacity,
            log_scale=log_scale,
            clim=clim,
            normal=normal,
            origin=origin,
            origin_translation=False,
            show_scalar_bar=False,
            normal_rotation=False,
            cmap=cmap,
            user_matrix=b,
        )

        prop = self.clip.GetOutlineProperty()
        prop.SetOpacity(0)

        prop = self.clip.GetEdgesProperty()
        prop.SetOpacity(0)

        actor = self.plotter.show_grid(
            xtitle=labels[0],
            ytitle=labels[1],
            ztitle=labels[2],
            font_size=8,
            minor_ticks=True,
        )

        actor.SetAxisBaseForX(*T[:, 0])
        actor.SetAxisBaseForY(*T[:, 1])
        actor.SetAxisBaseForZ(*T[:, 2])

        actor.bounds = bounds.ravel()
        actor.SetXAxisRange(limits[0])
        actor.SetYAxisRange(limits[1])
        actor.SetZAxisRange(limits[2])

        axis0_args = *limits[0], actor.n_xlabels, actor.x_label_format
        axis1_args = *limits[1], actor.n_ylabels, actor.y_label_format
        axis2_args = *limits[2], actor.n_zlabels, actor.z_label_format

        axis0_label = pv.plotting.cube_axes_actor.make_axis_labels(*axis0_args)
        axis1_label = pv.plotting.cube_axes_actor.make_axis_labels(*axis1_args)
        axis2_label = pv.plotting.cube_axes_actor.make_axis_labels(*axis2_args)

        actor.SetAxisLabels(0, axis0_label)
        actor.SetAxisLabels(1, axis1_label)
        actor.SetAxisLabels(2, axis2_label)

        self.reset_scene()

        self.clip.AddObserver("InteractionEvent", self.interaction_callback)

        self.P_inv = np.linalg.inv(P)

    def interaction_callback(self, caller, event):
        orig = caller.GetOrigin()
        # norm = caller.GetNormal()

        # norm /= np.linalg.norm(norm)
        # norm = self.norm

        ind = np.abs(self.norm).tolist().index(1)

        value = np.dot(self.P_inv, orig)[ind]

        self.slice_line.blockSignals(True)
        self.set_slice_value(value)
        self.slice_line.blockSignals(False)

        self.slice_ready.emit()

    def connect_slice_ready(self, reslice):
        self.slice_ready.connect(reslice)

    def __format_axis_coord(self, x, y):
        x, y, _ = np.dot(self.T_inv, [x, y, 1])
        return "x={:.3f}, y={:.3f}".format(x, y)

    def add_slice(self, slice_dict):
        self.max_slider.blockSignals(True)
        self.max_slider.setValue(100)
        self.max_slider.blockSignals(False)

        self.min_slider.blockSignals(True)
        self.min_slider.setValue(0)
        self.min_slider.blockSignals(False)

        cmap = cmaps[self.get_colormap()]

        x = slice_dict["x"]
        y = slice_dict["y"]

        labels = slice_dict["labels"]
        title = slice_dict["title"]
        signal = slice_dict["signal"]

        scale = self.get_slice_scale()

        vmin = np.nanmin(signal)
        vmax = np.nanmax(signal)

        if np.isclose(vmax, vmin) or not np.isfinite([vmin, vmax]).all():
            vmin, vmax = (0.1, 1) if scale == "log" else (0, 1)

        T = slice_dict["transform"]
        aspect = slice_dict["aspect"]

        self.T_inv = np.linalg.inv(T)

        self.ax_slice.format_coord = self.__format_axis_coord

        transform = Affine2D(T) + self.ax_slice.transData
        self.transform = transform

        self.xlim = np.array([x.min(), x.max()])
        self.ylim = np.array([y.min(), y.max()])

        if self.cb is not None:
            self.cb.remove()

        self.ax_slice.clear()

        im = self.ax_slice.pcolormesh(
            x,
            y,
            signal,
            norm=scale,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
            rasterized=True,
            transform=transform,
        )

        self.im = im
        self.vmin, self.vmax = self.im.norm.vmin, self.im.norm.vmax

        self.set_vmin_value(self.vmin)
        self.set_vmax_value(self.vmax)

        self.set_xmin_value(self.xlim[0])
        self.set_xmax_value(self.xlim[1])

        self.set_ymin_value(self.ylim[0])
        self.set_ymax_value(self.ylim[1])

        self.ax_slice.set_aspect(aspect)
        self.ax_slice.set_xlabel(labels[0])
        self.ax_slice.set_ylabel(labels[1])
        self.ax_slice.set_title(title)
        self.ax_slice.minorticks_on()

        self.ax_slice.xaxis.get_major_locator().set_params(integer=True)
        self.ax_slice.yaxis.get_major_locator().set_params(integer=True)

        self.cb = self.fig_slice.colorbar(self.im, ax=self.ax_slice)
        self.cb.minorticks_on()

        self.canvas_slice.draw_idle()
        self.canvas_slice.flush_events()

    def update_lines(self, alpha):
        lines = self.ax_slice.get_lines()
        for line in lines:
            line.set_alpha(alpha)
        self.canvas_slice.draw_idle()
        self.canvas_slice.flush_events()

    def add_cut(self, cut_dict):
        x = cut_dict["x"]
        y = cut_dict["y"]
        e = cut_dict["e"]

        val = cut_dict["value"]

        label = cut_dict["label"]
        title = cut_dict["title"]

        scale = self.get_cut_scale()

        line_cut = self.get_cut()

        lines = self.ax_slice.get_lines()
        for line in lines:
            line.remove()

        xlim = self.xlim
        ylim = self.ylim

        thick = self.get_cut_thickness()

        delta = 0 if thick is None else thick / 2

        if line_cut == "Axis 2":
            l0 = [val - delta, val - delta], ylim
            l1 = [val + delta, val + delta], ylim
            direction = "vertical"
        else:
            l0 = xlim, [val - delta, val - delta]
            l1 = xlim, [val + delta, val + delta]
            direction = "horizontal"

        l = self.toggle_line_box.isChecked()

        self.ax_slice.plot(*l0, "w--", lw=1, alpha=l, transform=self.transform)
        self.ax_slice.plot(*l1, "w--", lw=1, alpha=l, transform=self.transform)

        self.ax_cut.clear()

        self.ax_cut.errorbar(x, y, e)
        self.ax_cut.set_xlabel(label)
        self.ax_cut.set_yscale(scale)
        self.ax_cut.set_title(title)
        self.ax_cut.minorticks_on()

        self.ax_cut.xaxis.get_major_locator().set_params(integer=True)

        self.canvas_cut.draw_idle()
        self.canvas_cut.flush_events()

        self.canvas_slice.draw_idle()
        self.canvas_slice.flush_events()

        self.linecut = {
            "is_dragging": False,
            "line_cut": (xlim, ylim, delta, direction),
        }

        self.fig_slice.canvas.mpl_connect("button_press_event", self.on_press)

        self.fig_slice.canvas.mpl_connect(
            "button_release_event", self.on_release
        )

        self.fig_slice.canvas.mpl_connect(
            "motion_notify_event", self.on_motion
        )

    def on_press(self, event):
        if (
            event.inaxes == self.ax_slice
            and self.fig_slice.canvas.toolbar.mode == ""
            and self.toggle_line_box.isChecked()
        ):
            self.linecut["is_dragging"] = True

    def on_release(self, event):
        self.linecut["is_dragging"] = False

        self.cut_ready.emit()

    def connect_cut_ready(self, recut):
        self.cut_ready.connect(recut)

    def on_motion(self, event):
        if self.linecut["is_dragging"] and event.inaxes == self.ax_slice:
            lines = self.ax_slice.get_lines()
            for line in lines:
                line.remove()

            xlim, ylim, delta, direction = self.linecut["line_cut"]

            x, y, _ = np.dot(self.T_inv, [event.xdata, event.ydata, 1])

            self.cut_line.blockSignals(True)

            if direction == "vertical":
                l0 = [x - delta, x - delta], ylim
                l1 = [x + delta, x + delta], ylim
                self.set_cut_value(x)
            else:
                l0 = xlim, [y - delta, y - delta]
                l1 = xlim, [y + delta, y + delta]
                self.set_cut_value(y)

            self.cut_line.blockSignals(False)

            self.ax_slice.plot(
                *l0, "w--", linewidth=1, transform=self.transform
            )

            self.ax_slice.plot(
                *l1, "w--", linewidth=1, transform=self.transform
            )

            self.canvas_slice.draw_idle()
            self.canvas_slice.flush_events()

    def get_vol_scale(self):
        return self.vol_scale_combo.currentText()

    def get_opacity(self):
        return self.opacity_combo.currentText()

    def get_range(self):
        return self.range_combo.currentText()

    def get_colormap(self):
        return self.cbar_combo.currentText()

    def get_slice_value(self):
        if self.slice_line.hasAcceptableInput():
            return float(self.slice_line.text())

    def get_cut_value(self):
        if self.cut_line.hasAcceptableInput():
            return float(self.cut_line.text())

    def set_slice_value(self, val):
        self.slice_line.setText(str(round(val, 4)))

    def set_cut_value(self, val):
        self.cut_line.setText(str(round(val, 4)))

    def get_slice_thickness(self):
        if self.slice_thickness_line.hasAcceptableInput():
            return float(self.slice_thickness_line.text())

    def get_cut_thickness(self):
        if self.cut_thickness_line.hasAcceptableInput():
            return float(self.cut_thickness_line.text())

    def set_slice_thickness(self, val):
        self.slice_thickness_line.setText(str(val))

    def set_cut_thickness(self, val):
        self.cut_thickness_line.setText(str(val))

    def get_clim_clip_type(self):
        return self.clim_combo.currentText()

    def get_vlim_clip_type(self):
        return self.vlim_combo.currentText()

    def get_slice(self):
        return self.slice_combo.currentText()

    def get_cut(self):
        return self.cut_combo.currentText()

    def get_slice_scale(self):
        return self.slice_scale_combo.currentText().lower()

    def get_cut_scale(self):
        return self.cut_scale_combo.currentText().lower()

    def get_vmin_value(self):
        if self.vmin_line.hasAcceptableInput():
            return float(self.vmin_line.text())

    def get_vmax_value(self):
        if self.vmax_line.hasAcceptableInput():
            return float(self.vmax_line.text())

    def set_vmin_value(self, val):
        self.vmin_line.setText(str(round(val, 5)))

    def set_vmax_value(self, val):
        self.vmax_line.setText(str(round(val, 5)))

    def get_xmin_value(self):
        if self.xmin_line.hasAcceptableInput():
            return float(self.xmin_line.text())

    def get_xmax_value(self):
        if self.xmax_line.hasAcceptableInput():
            return float(self.xmax_line.text())

    def set_xmin_value(self, val):
        self.xmin_line.setText(str(round(val, 4)))

    def set_xmax_value(self, val):
        self.xmax_line.setText(str(round(val, 4)))

    def get_ymin_value(self):
        if self.ymin_line.hasAcceptableInput():
            return float(self.ymin_line.text())

    def get_ymax_value(self):
        if self.ymax_line.hasAcceptableInput():
            return float(self.ymax_line.text())

    def set_ymin_value(self, val):
        self.ymin_line.setText(str(round(val, 4)))

    def set_ymax_value(self, val):
        self.ymax_line.setText(str(round(val, 4)))

    def set_slice_lim(self, xlim, ylim):
        if self.cb is not None:
            xmin, xmax = xlim
            ymin, ymax = ylim
            T = np.linalg.inv(self.T_inv)
            xmin, ymin, _ = np.dot(T, [xmin, ymin, 1])
            xmax, ymax, _ = np.dot(T, [xmax, ymax, 1])
            self.ax_slice.set_xlim(xmin, xmax)
            self.ax_slice.set_ylim(ymin, ymax)
            self.canvas_slice.draw_idle()
            self.canvas_slice.flush_events()

    def set_cut_lim(self, lim):
        if self.cb is not None:
            self.ax_cut.set_xlim(*lim)
            self.canvas_cut.draw_idle()
            self.canvas_cut.flush_events()

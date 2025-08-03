"""
VolumeSlicerModel
----------------
Model for handling 3D volume slicing, cutting, and histogram operations using Mantid workspaces.

This class provides methods to load, process, and extract slices and cuts from multidimensional histogram workspaces.
It supports saving slices/cuts, checking workspace states, and computing transformations for visualization.

Attributes
----------
shape : tuple
    Shape of the loaded volume data.
min_lim : np.ndarray
    Minimum bounds for each dimension.
max_lim : np.ndarray
    Maximum bounds for each dimension.
labels : list
    Axis labels for the volume.
spacing : np.ndarray
    Bin widths for each dimension.
signals : list
    Downsampled signal arrays for each axis.
spacings : list
    Downsampled spacings for each axis.

Methods
-------
load_md_histo_workspace(filename)
    Load and preprocess a Mantid MD histogram workspace.
save_slice(filename)
    Save the current slice to an ASCII file.
save_cut(filename)
    Save the current cut to an ASCII file.
is_histo_loaded()
    Check if a histogram workspace is loaded.
is_sliced()
    Check if a slice workspace exists.
is_cut()
    Check if a cut workspace exists.
set_B()
    Set the B matrix from the workspace UB.
set_W()
    Set the W matrix from the workspace log.
get_histo_info(normal)
    Get histogram info for a given normal.
get_slice_info(normal, value, thickness)
    Get slice info for a given normal and value.
get_cut_info(axis, value, thickness)
    Get cut info for a given axis and value.
calculate_clim(trans, method)
    Calculate color limits for visualization.
orientation_matrix()
    Compute the orientation matrix.
get_transform(reciprocal)
    Get the transformation matrix.
get_transforms()
    Get projection, transform, and scale matrices.
get_normal_plane(ind)
    Get the normal vector for a plane.
"""

from mantid.simpleapi import (
    LoadMD,
    IntegrateMDHistoWorkspace,
    CompactMD,
    mtd,
)

import numpy as np
import scipy.linalg

import skimage.measure

from NeuXtalViz.models.base_model import NeuXtalVizModel
from NeuXtalViz.models.utilities import SaveMDToAscii


class VolumeSlicerModel(NeuXtalVizModel):
    def __init__(self):
        """
        Initialize the VolumeSlicerModel.
        """
        super(VolumeSlicerModel, self).__init__()

    def load_md_histo_workspace(self, filename):
        """
        Load and preprocess a Mantid MD histogram workspace from a file.

        Parameters
        ----------
        filename : str
            Path to the Mantid MD histogram file to load.
        """
        LoadMD(Filename=filename, OutputWorkspace="histo")

        signal = mtd["histo"].getSignalArray().copy()
        signal_var = mtd["histo"].getErrorSquaredArray().copy()

        mask = np.isfinite(signal) & np.isfinite(signal_var)

        signal[~mask] = 0
        signal_var[~mask] = 0

        mtd["histo"].setSignalArray(signal)
        mtd["histo"].setErrorSquaredArray(signal_var)

        CompactMD(InputWorkspace="histo", OutputWorkspace="volume")

        signal = mtd["volume"].getSignalArray()

        self.shape = signal.shape

        dims = [mtd["volume"].getDimension(i) for i in range(3)]

        self.min_lim = np.array(
            [dim.getMinimum() + dim.getBinWidth() * 0.5 for dim in dims]
        )

        self.max_lim = np.array(
            [dim.getMaximum() - dim.getBinWidth() * 0.5 for dim in dims]
        )

        self.labels = [
            "{} ({})".format(dim.name, dim.getUnits()) for dim in dims
        ]

        self.spacing = np.array([dim.getBinWidth() for dim in dims])

        scale = 0.125 / self.spacing
        scale[scale <= 1] = 1
        scale = scale.round().astype(int)

        blocks = [(scale[0], 1, 1), (1, scale[1], 1), (1, 1, scale[2])]

        self.signals = []
        self.spacings = []
        for block in blocks:
            self.spacings.append(self.spacing * np.array(block))
            self.signals.append(
                skimage.measure.block_reduce(
                    signal, block_size=block, func=np.nanmean, cval=np.nan
                )
            )

        self.set_B()
        self.set_W()

    def save_slice(self, filename):
        """
        Save the current slice workspace to an ASCII file.

        Parameters
        ----------
        filename : str
            Output filename for the slice.
        """
        SaveMDToAscii("slice", filename)

    def save_cut(self, filename):
        """
        Save the current cut workspace to an ASCII file.

        Parameters
        ----------
        filename : str
            Output filename for the cut.
        """
        SaveMDToAscii("cut", filename)

    def is_histo_loaded(self):
        """
        Check if a histogram workspace is loaded in Mantid.

        Returns
        -------
        bool
            True if 'histo' workspace exists, False otherwise.
        """
        return mtd.doesExist("histo")

    def is_sliced(self):
        """
        Check if a slice workspace exists in Mantid.

        Returns
        -------
        bool
            True if 'slice' workspace exists, False otherwise.
        """
        return mtd.doesExist("slice")

    def is_cut(self):
        """
        Check if a cut workspace exists in Mantid.

        Returns
        -------
        bool
            True if 'cut' workspace exists, False otherwise.
        """
        return mtd.doesExist("cut")

    def set_B(self):
        """
        Set the B matrix from the UB of the loaded histogram workspace.
        """
        if self.has_UB("histo"):
            ei = mtd["histo"].getExperimentInfo(0)

            B = ei.sample().getOrientedLattice().getB().copy()

            self.set_UB(B)

    def set_W(self):
        """
        Set the W matrix from the workspace log if available, otherwise identity.
        """
        ei = mtd["histo"].getExperimentInfo(0)

        self.W = np.eye(3)

        if ei.run().hasProperty("W_MATRIX"):
            self.W = ei.run().getLogData("W_MATRIX").value.reshape(3, 3)

    def get_histo_info(self, normal):
        """
        Get histogram information for a given normal direction.

        Parameters
        ----------
        normal : array-like
            Normal vector for the slicing direction.

        Returns
        -------
        dict
            Dictionary containing signal, limits, spacing, labels, and transforms.
        """
        ind = np.abs(normal).tolist().index(1)

        histo_dict = {}

        histo_dict["signal"] = self.signals[ind].copy()

        histo_dict["min_lim"] = self.min_lim
        histo_dict["max_lim"] = self.max_lim
        histo_dict["spacing"] = self.spacings[ind]
        histo_dict["labels"] = self.labels

        P, T, S = self.get_transforms()

        histo_dict["transform"] = T
        histo_dict["projection"] = P
        histo_dict["scales"] = S

        return histo_dict

    def get_slice_info(self, normal, value, thickness=0.01):
        """
        Get slice information for a given normal and value.

        Parameters
        ----------
        normal : array-like
            Normal vector for the slicing direction.
        value : float
            Position along the normal to slice.
        thickness : float, optional
            Thickness of the slice (default 0.01).

        Returns
        -------
        dict
            Dictionary containing x, y, labels, signal, transform, aspect, value, and title.
        """
        self.normal = normal

        slice_dict = {}

        integrate = [value - thickness, value + thickness]

        self.integrate = integrate

        pbin = [None if norm == 0 else integrate for norm in normal]

        IntegrateMDHistoWorkspace(
            InputWorkspace="volume",
            P1Bin=pbin[0],
            P2Bin=pbin[1],
            P3Bin=pbin[2],
            OutputWorkspace="slice",
        )

        i = np.abs(normal).tolist().index(1)

        form = "{} = ({:.2f},{:.2f})"

        title = form.format(mtd["slice"].getDimension(i).name, *integrate)

        dims = mtd["slice"].getNonIntegratedDimensions()

        x, y = [
            np.linspace(
                dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
            )
            for dim in dims
        ]

        labels = ["{} ({})".format(dim.name, dim.getUnits()) for dim in dims]

        slice_dict["x"] = x
        slice_dict["y"] = y
        slice_dict["labels"] = labels

        signal = mtd["slice"].getSignalArray().T.copy().squeeze()

        signal[signal <= 0] = np.nan
        signal[np.isinf(signal)] = np.nan

        slice_dict["signal"] = signal

        Bp = np.dot(self.UB, self.W)

        Q, R = scipy.linalg.qr(Bp)

        ind = np.abs(normal) != 1

        v = scipy.linalg.cholesky(np.dot(R.T, R)[ind][:, ind], lower=False)

        v /= v[0, 0]

        T = np.eye(3)
        T[:2, :2] = v

        s = np.diag(T).copy()
        T[1, 1] = 1

        T[0, 2] = -T[0, 1] * y.min()

        slice_dict["transform"] = T
        slice_dict["aspect"] = s[1]
        slice_dict["value"] = value
        slice_dict["title"] = title

        return slice_dict

    def get_cut_info(self, axis, value, thickness=0.01):
        """
        Get cut information for a given axis and value.

        Parameters
        ----------
        axis : array-like
            Axis along which to cut (e.g., [1,0,0]).
        value : float
            Position along the axis to cut.
        thickness : float, optional
            Thickness of the cut (default 0.01).

        Returns
        -------
        dict
            Dictionary containing x, y, e, label, value, and title.
        """
        cut_dict = {}

        integrate = [value - thickness, value + thickness]

        pbin = [None if ax == 0 else integrate for ax in axis]

        IntegrateMDHistoWorkspace(
            InputWorkspace="slice",
            P1Bin=pbin[0],
            P2Bin=pbin[1],
            P3Bin=pbin[2],
            OutputWorkspace="cut",
        )

        i = np.abs(self.normal).tolist().index(1)
        j = np.array(axis).tolist().index(1)

        form = "{} = ({:.2f},{:.2f})"

        title = form.format(mtd["slice"].getDimension(i).name, *self.integrate)
        title += " / "
        title += form.format(mtd["cut"].getDimension(j).name, *integrate)

        dim = mtd["cut"].getNonIntegratedDimensions()[0]

        x = np.linspace(
            dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
        )

        x = 0.5 * (x[1:] + x[:-1])

        label = "{} ({})".format(dim.name, dim.getUnits())

        cut_dict["x"] = x
        cut_dict["y"] = mtd["cut"].getSignalArray().squeeze()
        cut_dict["e"] = np.sqrt(mtd["cut"].getErrorSquaredArray().squeeze())
        cut_dict["label"] = label
        cut_dict["value"] = value
        cut_dict["title"] = title

        return cut_dict

    def calculate_clim(self, trans, method="normal"):
        """
        Calculate color limits for visualization based on a method.

        Parameters
        ----------
        trans : np.ndarray
            Array of values to calculate limits for.
        method : str, optional
            Method for calculation: 'normal', 'boxplot', or 'minmax'.

        Returns
        -------
        np.ndarray
            Array with values clipped to the calculated color limits.
        """
        trans[~np.isfinite(trans)] = np.nan

        vmin, vmax = np.nanmin(trans), np.nanmax(trans)

        if np.isclose(vmin, vmax) or not np.isfinite(vmin):
            vmin = vmax / 100
        elif not np.isfinite([vmin, vmax]).all():
            vmin, vmax = 1e-3, 1 + 3

        if method == "normal":
            mu, sigma = np.nanmean(trans), np.nanstd(trans)

            spread = 3 * sigma

            cmin, cmax = mu - spread, mu + spread

        elif method == "boxplot":
            Q1, Q3 = np.nanpercentile(trans, [25, 75])

            IQR = Q3 - Q1

            spread = 1.5 * IQR

            cmin, cmax = Q1 - spread, Q3 + spread

        else:
            cmin, cmax = vmin, vmax

        clim = [cmin if cmin > vmin else vmin, cmax if cmax < vmax else vmax]

        trans[trans < clim[0]] = clim[0]
        trans[trans > clim[1]] = clim[1]

        return trans

    def orientation_matrix(self):
        """
        Compute the orientation matrix for the current UB and W matrices.

        Returns
        -------
        np.ndarray
            Orientation matrix.
        """
        Bp = np.dot(self.UB, self.W)

        Q, R = scipy.linalg.qr(Bp)

        v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

        Q = np.dot(Bp, np.linalg.inv(v))

        return np.dot(Q.T, self.UB)

    def get_transform(self, reciprocal=True):
        """
        Get the transformation matrix for the current UB and W matrices.

        Parameters
        ----------
        reciprocal : bool, optional
            If True, return reciprocal transformation; else real space.

        Returns
        -------
        np.ndarray
            Transformation matrix.
        """
        if self.UB is not None:
            b = self.UB / np.linalg.norm(self.UB, axis=0)

            Bp = np.dot(self.UB, self.W)

            Q, R = scipy.linalg.qr(Bp)

            v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

            Q = np.dot(Bp, np.linalg.inv(v))

            T = np.dot(Q.T, b)

            if not reciprocal:
                T = np.column_stack(
                    [
                        np.cross(T[:, 1], T[:, 2]),
                        np.cross(T[:, 2], T[:, 0]),
                        np.cross(T[:, 0], T[:, 1]),
                    ]
                )

            return T

    def get_transforms(self):
        """
        Get projection, transform, and scale matrices for the current UB and W.

        Returns
        -------
        tuple
            (projection, transform, scale) matrices.
        """
        Bp = np.dot(self.UB, self.W)

        Q, R = scipy.linalg.qr(Bp)

        v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

        s = np.linalg.norm(v, axis=0)
        t = v / s
        p = v / v[0, 0]

        s = np.linalg.norm(p, axis=0)

        return p, t, s

    def get_normal_plane(self, ind):
        """
        Get the normal vector for a plane given an index vector.

        Parameters
        ----------
        ind : array-like
            Index vector for the plane.

        Returns
        -------
        np.ndarray
            Normal vector for the plane.
        """
        if self.UB is not None:
            Bp = np.dot(self.UB, self.W)

            Q, R = scipy.linalg.qr(Bp)

            v = scipy.linalg.cholesky(np.dot(R.T, R), lower=False)

            matrix = np.cross(
                np.dot(v, np.roll(np.eye(3), 2, 1)).T,
                np.dot(v, np.roll(np.eye(3), 1, 1)).T,
            ).T

            vec = np.dot(matrix, ind)

            return vec

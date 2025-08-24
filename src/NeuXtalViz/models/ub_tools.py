"""
UBModel and related utilities for crystallographic analysis in NeuXtalViz.

This module provides the UBModel class, which encapsulates methods for loading, manipulating, and analyzing crystallographic data, including peak finding, UB matrix determination, lattice refinement, and peak clustering. It interfaces with Mantid algorithms for data processing and supports both conventional and modulated structures.

Classes
-------
UBModel
    Main model for UB matrix and peak table operations, including loading/saving, indexing, integration, and clustering.

"""

import os
from collections import defaultdict

from mantid.simpleapi import (
    SelectCellWithForm,
    ShowPossibleCells,
    TransformHKL,
    CalculatePeaksHKL,
    IndexPeaks,
    FindUBUsingFFT,
    FindUBUsingLatticeParameters,
    FindUBUsingIndexedPeaks,
    OptimizeLatticeForCellType,
    CalculateUMatrix,
    HasUB,
    SetUB,
    LoadIsawUB,
    SaveIsawUB,
    FindPeaksMD,
    PredictPeaks,
    PredictSatellitePeaks,
    CentroidPeaksMD,
    IntegratePeaksMD,
    FilterPeaks,
    SortPeaksWorkspace,
    LoadIsawPeaks,
    DeleteWorkspace,
    DeleteTableRows,
    CombinePeaksWorkspaces,
    CreatePeaksWorkspace,
    ConvertQtoHKLMDHisto,
    CompactMD,
    CopySample,
    CreateSampleWorkspace,
    CloneWorkspace,
    SaveNexus,
    LoadNexus,
    HB3AAdjustSampleNorm,
    LoadWANDSCD,
    Load,
    Rebin,
    SetGoniometer,
    PreprocessDetectorsToMD,
    CompressEvents,
    GroupDetectors,
    GroupWorkspaces,
    UnGroupWorkspace,
    RenameWorkspace,
    ConvertToMD,
    ConvertHFIRSCDtoMDE,
    LoadMD,
    SaveMD,
    MergeMD,
    LoadNexus,
    LoadIsawDetCal,
    LoadParameterFile,
    LoadEmptyInstrument,
    ApplyCalibration,
    BinMD,
    ConvertUnits,
    CropWorkspace,
    mtd,
)

from mantid import config

config["Q.convention"] = "Crystallography"

from mantid.geometry import (
    CrystalStructure,
    ReflectionGenerator,
    ReflectionConditionFilter,
    PointGroupFactory,
    UnitCell,
)

from mantid.kernel import V3D, FloatTimeSeriesProperty

from sklearn.cluster import DBSCAN

import numpy as np
import scipy
import json

from NeuXtalViz.models.base_model import NeuXtalVizModel
from NeuXtalViz.config.instruments import beamlines

lattice_group = {
    "Triclinic": "-1",
    "Monoclinic": "2/m",
    "Orthorhombic": "mmm",
    "Tetragonal": "4/mmm",
    "Rhombohedral": "-3m",
    "Hexagonal": "6/mmm",
    "Cubic": "m-3m",
}

centering_reflection = {
    "P": "Primitive",
    "I": "Body centred",
    "F": "All-face centred",
    "R": "Primitive",  # rhomb axes
    "R(obv)": "Rhombohderally centred, obverse",  # hex axes
    "R(rev)": "Rhombohderally centred, reverse",  # hex axes
    "A": "A-face centred",
    "B": "B-face centred",
    "C": "C-face centred",
}

variable = {
    "I/σ": "Signal/Noise",
    "I": "Intensity",
    "d": "DSpacing",
    "λ": "Wavelength",
    "Q": "QMod",
    "h^2+k^2+l^2": "h^2+k^2+l^2",
    "m^2+n^2+p^2": "m^2+n^2+p^2",
    "Run #": "RunNumber",
}


class UBModel(NeuXtalVizModel):
    """
    Model for UB matrix and peak table operations in NeuXtalViz.

    Provides methods for loading, saving, and manipulating
    crystallographic data, including peak finding, UB matrix
    determination, lattice refinement, and clustering. Integrates with
    Mantid algorithms for data processing and supports both conventional
    and modulated structures.
    """

    def __init__(self):
        """
        Initialize the UBModel instance and set up internal state.
        """

        super(UBModel, self).__init__()

        self.Q = None
        self.table = "ub_peaks"
        self.cell = "ub_lattice"
        self.primitive_cell = "primitive_cell"

        self.peak_info = None

        CreateSampleWorkspace(OutputWorkspace="ub_lattice")

    def has_Q(self):
        """
        Check if the Q workspace exists in Mantid.

        Returns
        -------
        bool
            True if Q workspace exists, False otherwise.
        """

        if self.Q is None:
            return False
        elif mtd.doesExist(self.Q):
            return True
        else:
            return False

    def has_peaks(self):
        """
        Check if the peaks table exists in Mantid.

        Returns
        -------
        bool
            True if peaks table exists, False otherwise.
        """

        if mtd.doesExist(self.table):
            return True
        else:
            return False

    def has_UB(self):
        """
        Check if the UB matrix is defined on the sample.

        Returns
        -------
        bool
            True if UB matrix is present, False otherwise.
        """

        return HasUB(Workspace=self.cell)

    def get_UB(self):
        """
        Retrieve the UB matrix from the oriented lattice.

        Returns
        -------
        np.ndarray
            3x3 UB matrix if present, else None.
        """

        if self.has_UB():
            return mtd[self.cell].sample().getOrientedLattice().getUB().copy()

    def update_UB(self):
        """
        Update the UB matrix on the sample and synchronize with Mantid
        workspaces.
        """

        UB = self.get_UB()

        if UB is not None:
            self.set_UB(UB)

    def get_instrument_name(self, instrument):
        """
        Get the instrument name string for a given instrument identifier.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Instrument name.
        """

        return beamlines[instrument]["Name"]

    def get_goniometers(self, instrument):
        """
        Get goniometer settings for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        list
            List of goniometer settings.
        """

        return beamlines[instrument]["Goniometers"]

    def get_wavelength(self, instrument):
        """
        Get the wavelength for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        float
            Wavelength in angstroms.
        """

        return beamlines[instrument]["Wavelength"]

    def get_raw_file_path(self, instrument):
        """
        Get the raw data file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            File path to raw data.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
            "IPTS-{}",
            inst["RawFile"],
        )

    def get_shared_file_path(self, instrument, ipts):
        """
        Get the shared file path for a given instrument and IPTS.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        ipts : str
            IPTS number.

        Returns
        -------
        str
            Shared file path.
        """

        inst = beamlines[instrument]

        if ipts is not None:
            filepath = os.path.join(
                "/",
                inst["Facility"],
                inst["InstrumentName"],
                "IPTS-{}".format(ipts),
                "shared",
            )
            if os.path.exists(filepath):
                return filepath

        filepath = os.path.join("/", inst["Facility"], inst["InstrumentName"])

        return filepath

    def get_calibration_file_path(self, instrument):
        """
        Get the calibration file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Calibration file path.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/",
            inst["Facility"],
            inst["InstrumentName"],
            "shared",
            "calibration",
        )

    def get_vanadium_file_path(self, instrument):
        """
        Get the vanadium calibration file path for a given instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Vanadium calibration file path.
        """

        inst = beamlines[instrument]

        return os.path.join(
            "/", inst["Facility"], inst["InstrumentName"], "shared", "Vanadium"
        )

    def load_data(self, instrument, IPTS, runs, exp, time_stop):
        """
        Load experimental data for a given instrument and run parameters.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        IPTS : str
            IPTS number.
        runs : list
            List of run numbers.
        exp : str
            Experiment identifier.
        time_stop : float
            Time to stop loading data.
        """

        filepath = self.get_raw_file_path(instrument)

        inst = beamlines[instrument]

        grouping = inst["Grouping"]

        LoadEmptyInstrument(
            InstrumentName=inst["Name"],
            OutputWorkspace="goniometer",
        )

        if instrument == "DEMAND":
            filenames = [filepath.format(IPTS, exp, runs)]
            if np.all([os.path.exists(filename) for filename in filenames]):
                self.runs = runs
                HB3AAdjustSampleNorm(
                    Filename=filenames,
                    OutputType="Detector",
                    NormaliseBy="None",
                    Grouping=grouping,
                    OutputWorkspace="data",
                )
                group = mtd["data"].isGroup()
                if not group:
                    GroupWorkspaces(
                        InputWorkspaces="data", OutputWorkspace="data"
                    )
                return True
        elif instrument == "WAND²":
            filenames = [filepath.format(IPTS, run) for run in runs]
            self.runs = runs
            if np.all([os.path.exists(filename) for filename in filenames]):
                filenames = ",".join([filename for filename in filenames])
                LoadWANDSCD(
                    Filename=filenames,
                    Grouping=grouping,
                    OutputWorkspace="data",
                )
                group = mtd["data"].isGroup()
                if not group:
                    GroupWorkspaces(
                        InputWorkspaces="data", OutputWorkspace="data"
                    )
                return True
        else:
            filenames = [filepath.format(IPTS, run) for run in runs]
            if np.all([os.path.exists(filename) for filename in filenames]):
                self.runs = runs
                filenames = ",".join([filename for filename in filenames])
                Load(
                    Filename=filenames,
                    FilterByTimeStop=time_stop,
                    NumberOfBins=1,
                    OutputWorkspace="data",
                )
                group = mtd["data"].isGroup()
                if not group:
                    GroupWorkspaces(
                        InputWorkspaces="data", OutputWorkspace="data"
                    )
                input_ws = mtd["data"].getNames()[0]
                PreprocessDetectorsToMD(
                    InputWorkspace=input_ws, OutputWorkspace="detectors"
                )
                cols, rows = inst["BankPixels"]
                c, r = [int(val) for val in grouping.split("x")]
                shape = (-1, cols, rows)
                # det_id = np.array(mtd['detectors'].column(4)).reshape(*shape)
                det_map = np.array(mtd["detectors"].column(5)).reshape(*shape)
                shape = det_map.shape
                i, j, k = np.meshgrid(
                    np.arange(shape[0]),
                    np.arange(shape[1]),
                    np.arange(shape[2]),
                    indexing="ij",
                )
                keys = np.stack((i, j // c, k // r), axis=-1)
                keys_flat = keys.reshape(-1, keys.shape[-1])
                det_map_flat = det_map.ravel().astype(str)
                grouped_ids = defaultdict(list)
                for key, detector_id in zip(
                    map(tuple, keys_flat), det_map_flat
                ):
                    grouped_ids[key].append(detector_id)
                detector_list = ",".join(
                    "+".join(group) for group in grouped_ids.values()
                )
                GroupDetectors(
                    InputWorkspace="data",
                    OutputWorkspace="data",
                    GroupingPattern=detector_list,
                )
                return True

    def calibrate_data(self, instrument, det_cal, gon_cal, tube_cal):
        """
        Calibrate the loaded data using detector, goniometer, and tube
        calibration files.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        det_cal : str
            Detector calibration file.
        tube_cal : str
            Tube calibration file.
        gon_cal : str
            Goniometer calibration file.
        """

        filepath = self.get_raw_file_path(instrument)

        if mtd.doesExist("data"):
            goniometers = self.get_goniometers(instrument)
            while len(goniometers) < 6:
                goniometers.append(None)

            SetGoniometer(
                Workspace="data",
                Axis0=goniometers[0],
                Axis1=goniometers[1],
                Axis2=goniometers[2],
                Average=False if "HFIR" in filepath else True,
            )

            if tube_cal != "" and os.path.exists(tube_cal):
                LoadNexus(Filename=tube_cal, OutputWorkspace="tube_table")
                ApplyCalibration(
                    Workspace="data", CalibrationTable="tube_table"
                )

            if det_cal != "" and os.path.exists(det_cal):
                if os.path.splitext(det_cal)[1] == ".xml":
                    LoadParameterFile(Workspace="data", Filename=det_cal)
                else:
                    ws = mtd["data"]
                    group = ws.isGroup()
                    if group:
                        for input_ws in ws.getNames():
                            LoadIsawDetCal(
                                InputWorkspace=input_ws, Filename=det_cal
                            )
                    else:
                        LoadIsawDetCal(InputWorkspace="data", Filename=det_cal)

            if (
                gon_cal != ""
                and os.path.exists(gon_cal)
                and os.path.splitext(gon_cal)[1] == ".xml"
            ):

                LoadParameterFile(Workspace="goniometer", Filename=gon_cal)

                inst = mtd["goniometer"].getInstrument()

                workspaces = (
                    list(mtd["data"].getNames())
                    if mtd["data"].isGroup()
                    else ["data"]
                )

                for workspace in workspaces:
                    run = mtd[workspace].run()

                    params = ["omega-offset", "chi-offset"]

                    for i, param in enumerate(params):
                        if inst.hasParameter(param):
                            val = inst.getNumberParameter(param)[0]
                            name = goniometers[i].split(",")[0]
                            values = run.getProperty(name).value
                            times = run.getProperty(name).times
                            log = FloatTimeSeriesProperty(name)
                            for t, v in zip(times, values):
                                log.addValue(t, v + val)
                            run[name] = log

                    SetGoniometer(
                        Workspace=workspace,
                        Axis0=goniometers[0],
                        Axis1=goniometers[1],
                        Axis2=goniometers[2],
                        Average=False if "HFIR" in filepath else True,
                    )

                    run = mtd[workspace].run()

                    param = "goniometer-tilt"
                    if inst.hasParameter(param):
                        v = inst.getStringParameter(param)[0]
                        G = np.array(v.split(",")).astype(float).reshape(3, 3)
                        gon = run.getGoniometer()
                        gon.setR(G @ gon.getR())

    def get_number_workspaces(self):
        """
        Get the number of workspaces in the current Mantid session.

        Returns
        -------
        int
            Number of workspaces.
        """

        if mtd.doesExist("data"):
            input_ws_names = mtd["data"].getNames()
            return len(input_ws_names)

    def convert_data(self, instrument, wavelength, lorentz, min_d=None):
        """
        Convert loaded data to Q-space using Mantid algorithms.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        wavelength : float
            Wavelength in angstroms.
        lorentz : bool
            Whether to apply Lorentz correction.
        min_d : float, optional
            Minimum d-spacing. Default is None.
        """

        filepath = self.get_raw_file_path(instrument)

        if min_d is not None:
            Q_max = 2 * np.pi / min_d

        if mtd.doesExist("data"):
            input_ws_names = mtd["data"].getNames()
            input_ws = input_ws_names[0]

            Rs = []

            if "HFIR" in filepath:
                r = mtd[input_ws].getExperimentInfo(0).run()

                two_theta = r.getProperty("TwoTheta").value
                az_phi = r.getProperty("Azimuthal").value

                for ws in input_ws_names:
                    r = mtd[ws].getExperimentInfo(0).run()
                    Rs.append(
                        [
                            r.getGoniometer(i).getR()
                            for i in range(r.getNumGoniometers())
                        ]
                    )

                lamda = wavelength[0]

                counts = [
                    np.swapaxes(mtd[ws].getSignalArray().copy(), 0, 1)
                    for ws in input_ws_names
                ]

                counts = [c.reshape(-1, c.shape[2]) for c in counts]

                if min_d is None:
                    k = 2 * np.pi / wavelength[0]
                    Q_max = k * np.sin(0.5 * max(two_theta))

                ConvertHFIRSCDtoMDE(
                    InputWorkspace="data",
                    Wavelength=wavelength[0],
                    LorentzCorrection=lorentz,
                    MinValues=[-Q_max, -Q_max, -Q_max],
                    MaxValues=[+Q_max, +Q_max, +Q_max],
                    MaxRecursionDepth=5,
                    OutputWorkspace="md",
                )
            else:

                if min_d is None:
                    k = 2 * np.pi / min(wavelength)
                    Q_max = k * np.sin(0.5 * max(two_theta))

                ConvertUnits(
                    InputWorkspace="data",
                    Target="MomentumTransfer",
                    OutputWorkspace="data",
                )

                CropWorkspace(
                    InputWorkspace="data",
                    XMin=0,
                    XMax=Q_max,
                    OutputWorkspace="data",
                )

                ConvertUnits(
                    InputWorkspace="data",
                    Target="Wavelength",
                    OutputWorkspace="data",
                )

                CropWorkspace(
                    InputWorkspace="data",
                    XMin=wavelength[0],
                    XMax=wavelength[1],
                    OutputWorkspace="data",
                )

                CompressEvents(
                    InputWorkspace="data",
                    Tolerance=1e-4,
                    OutputWorkspace="data",
                )

                Rebin(
                    InputWorkspace="data",
                    OutputWorkspace="data",
                    Params=[wavelength[0], 0.01, wavelength[1]],
                )

                lamda = mtd[input_ws].extractX()[0]
                lamda = 0.5 * (lamda[1:] + lamda[:-1])

                PreprocessDetectorsToMD(
                    InputWorkspace=input_ws, OutputWorkspace="detectors"
                )

                two_theta = mtd["detectors"].column("TwoTheta")
                az_phi = mtd["detectors"].column("Azimuthal")

                for ws in input_ws_names:
                    Rs.append(mtd[ws].run().getGoniometer().getR())

                counts = [mtd[ws].extractY().copy() for ws in input_ws_names]

                ConvertToMD(
                    InputWorkspace="data",
                    QDimensions="Q3D",
                    dEAnalysisMode="Elastic",
                    Q3DFrames="Q_sample",
                    LorentzCorrection=lorentz,
                    MinValues=[-Q_max, -Q_max, -Q_max],
                    MaxValues=[+Q_max, +Q_max, +Q_max],
                    PreprocDetectorsWS="detectors",
                    OutputWorkspace="md",
                )

            input_ws_names = mtd["md"].getNames()
            input_ws = input_ws_names[0]

            if len(input_ws_names) > 1:
                MergeMD(InputWorkspaces="md", OutputWorkspace="md")

            else:
                UnGroupWorkspace(InputWorkspace="md")

                RenameWorkspace(InputWorkspace=input_ws, OutputWorkspace="md")

            self.Q_max_cut = Q_max

            self.Q = "md"

            BinMD(
                InputWorkspace=self.Q,
                AlignedDim0="Q_sample_x,{},{},256".format(-Q_max, Q_max),
                AlignedDim1="Q_sample_y,{},{},256".format(-Q_max, Q_max),
                AlignedDim2="Q_sample_z,{},{},256".format(-Q_max, Q_max),
                OutputWorkspace="Q3D",
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace=self.Q,
                NumberOfPeaks=0,
                OutputWorkspace=self.table,
            )

            CopySample(
                InputWorkspace=self.Q,
                OutputWorkspace=self.cell,
                CopyName=False,
                CopyMaterial=False,
                CopyEnvironment=False,
                CopyShape=False,
            )

            CompactMD(InputWorkspace="Q3D", OutputWorkspace="Q3D")

            signal = mtd["Q3D"].getSignalArray().copy()
            signal[np.isclose(signal, 0)] = np.nan

            threshold = np.nanpercentile(signal, 90)
            signal[signal <= threshold] = np.nan

            # threshold = np.nanpercentile(signal, 99)
            # signal[signal >= threshold] = threshold

            dims = [mtd["Q3D"].getDimension(i) for i in range(3)]

            x, y, z = [
                np.linspace(
                    dim.getMinimum() + dim.getBinWidth() / 2,
                    dim.getMaximum() - dim.getBinWidth() / 2,
                    dim.getNBins(),
                )
                for dim in dims
            ]

            Qx, Qy, Qz = np.meshgrid(x, y, z, indexing="ij")

            mask = (Qx**2 + Qy**2 + Qz**2) > self.Q_max_cut**2
            signal[mask] = np.nan

            self.spacing = tuple([dim.getBinWidth() for dim in dims])

            self.min_lim = x[0], y[0], z[0]
            self.max_lim = x[-1], y[-1], z[-1]

            signal = np.log10(signal)

            smin = np.nanmin(signal)
            smax = np.nanmax(signal)

            self.signal = np.round(
                255 * (signal - smin) / (smax - smin)
            ).astype(np.uint8)

            self.wavelength = wavelength
            self.counts = counts

            self.two_theta = np.array(two_theta)
            self.lamda = lamda
            self.Rs = Rs

            kf_x = np.sin(two_theta) * np.cos(az_phi)
            kf_y = np.sin(two_theta) * np.sin(az_phi)
            kf_z = np.cos(two_theta)

            self.nu = np.rad2deg(np.arcsin(kf_y))
            self.gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

    def add_peak(self, ind, val, horz, vert):
        """
        Add a peak to the peaks table.

        Parameters
        ----------
        ind : int
            Index of the run.
        val : float
            Peak value (e.g., d-spacing or angle).
        horz : float
            Horizontal coordinate (e.g., angle or hkl).
        vert : float
            Vertical coordinate (e.g., angle or hkl).
        """

        R = self.Rs[ind]

        if type(self.lamda) is float:
            wl = self.lamda
            x = np.rad2deg(np.arccos([0.5 * (np.trace(r) - 1) for r in R]))
            R = R[np.argmin(np.abs(x - val))]
        else:
            wl = self.lamda[np.argmin(np.abs(self.lamda - val))]

        k = 2 * np.pi / wl

        Qx = k * np.cos(np.deg2rad(vert)) * np.sin(np.deg2rad(horz))
        Qy = k * np.sin(np.deg2rad(vert))
        Qz = k * (np.cos(np.deg2rad(vert)) * np.cos(np.deg2rad(horz)) - 1)

        mtd["ub_peaks"].run().getGoniometer().setR(R)
        peak = mtd["ub_peaks"].createPeak([Qx, Qy, Qz])
        peak.setRunNumber(self.runs[ind])
        mtd["ub_peaks"].addPeak(peak)

    def calculate_hkl_position(self, ind, h, k, l):
        """
        Calculate the HKL position for a given peak index and Miller indices.

        Parameters
        ----------
        ind : int
            Index of the peak.
        h, k, l : int
            Miller indices.

        Returns
        -------
        tuple
            Calculated values (x, gamma, nu) for the given HKL.
        """

        if self.has_UB():
            UB = self.get_UB()
            Q = 2 * np.pi * UB @ np.array([h, k, l])

            R = self.Rs[ind]

            if type(self.lamda) is float:
                wl = self.lamda
                Q = np.einsum("kij,j->ki", R, Q)
                lamda = (
                    4 * np.pi * np.abs(Q[2]) / np.linalg.norm(Q, axis=0) ** 2
                )
                R = R[np.argmin(np.abs(lamda - wl))]
                Q = np.einsum("ij,j->i", R, Q)
                x = np.rad2deg(np.arccos(0.5 * (np.trace(R) - 1)))
            else:
                Q = np.einsum("ij,j->i", R, Q)
                wl = 4 * np.pi * np.abs(Q[2]) / np.linalg.norm(Q) ** 2
                x = wl

            az_phi = np.arctan2(Q[1], Q[0])
            two_theta = 2 * np.abs(np.arcsin(Q[2] / np.linalg.norm(Q)))

            kf_x = np.sin(two_theta) * np.cos(az_phi)
            kf_y = np.sin(two_theta) * np.sin(az_phi)
            kf_z = np.cos(two_theta)

            nu = np.rad2deg(np.arcsin(kf_y))
            gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

            return x, gamma, nu

    def roi_scan_to_hkl(self, ind, val, horz, vert):
        """
        Convert a ROI scan position to HKL coordinates.

        Parameters
        ----------
        ind : int
            Index of the run.
        val : float
            Peak value (e.g., d-spacing or angle).
        horz : float
            Horizontal coordinate (e.g., angle or hkl).
        vert : float
            Vertical coordinate (e.g., angle or hkl).

        Returns
        -------
        np.ndarray
            Calculated HKL coordinates.
        """

        if self.has_UB():
            R = self.Rs[ind]

            if type(self.lamda) is float:
                wl = self.lamda
                x = np.rad2deg(np.arccos([0.5 * (np.trace(r) - 1) for r in R]))
                R = R[np.argmin(np.abs(x - val))]
            else:
                wl = self.lamda[np.argmin(np.abs(self.lamda - val))]

            k = 2 * np.pi / wl

            Qx = k * np.cos(np.deg2rad(vert)) * np.sin(np.deg2rad(horz))
            Qy = k * np.sin(np.deg2rad(vert))
            Qz = k * (np.cos(np.deg2rad(vert)) * np.cos(np.deg2rad(horz)) - 1)

            UB = self.get_UB()

            hkl = np.dot(np.linalg.inv(2 * np.pi * (R @ UB)), [Qx, Qy, Qz])

            return hkl

    def calculate_instrument_view(self, ind, d_min, d_max):
        """
        Calculate the instrument view for a given peak index and d-spacing range.

        Parameters
        ----------
        ind : int
            Index of the peak.
        d_min, d_max : float
            Minimum and maximum d-spacing values.

        Returns
        -------
        dict
            Instrument view data including d, gamma, nu, and counts.
        """

        inst_view = {}

        R = self.Rs[ind]

        if type(self.lamda) is float:
            lamda = np.full(len(R), self.lamda)
        else:
            lamda = self.lamda

        if np.isclose(d_min, d_max) or d_max < d_min:
            d_min, d_max = 0, np.inf

        d = 0.5 * lamda / np.sin(0.5 * self.two_theta[:, np.newaxis])
        mask = (d > d_min) & (d < d_max)

        rows, cols = np.nonzero(mask)

        vals = self.counts[ind].copy()
        vals[~mask] = np.nan

        uni_rows = np.unique(rows)

        counts = np.nansum(vals[uni_rows], axis=1)

        sort = np.argsort(counts)

        inst_view["d"] = d
        inst_view["d_min"] = d_min
        inst_view["d_max"] = d_max
        inst_view["gamma"] = self.gamma[uni_rows][sort]
        inst_view["nu"] = self.nu[uni_rows][sort]
        inst_view["counts"] = counts[sort]
        inst_view["ind"] = ind

        self.inst_view = inst_view

    def extract_roi(self, horz, vert, horz_roi, vert_roi, val):
        """
        Extract a region of interest (ROI) from the instrument view.

        Parameters
        ----------
        horz, vert : float
            Horizontal and vertical coordinates for the ROI center.
        horz_roi, vert_roi : float
            Horizontal and vertical ROI sizes.
        val : float
            Value at the ROI center.

        Returns
        -------
        dict
            Extracted ROI data including label, x, and y values.
        """

        inst_view = self.inst_view

        roi_view = {}

        d = inst_view["d"]
        d_min = inst_view["d_min"]
        d_max = inst_view["d_max"]
        gamma = inst_view["gamma"]
        nu = inst_view["nu"]
        ind = inst_view["ind"]

        if horz_roi == 0:
            horz_roi = (gamma.max() - gamma.min()) / 2

        if vert_roi == 0:
            vert_roi = (nu.max() - nu.min()) / 2

        if horz < gamma.min() or val > gamma.max():
            val = (gamma.max() + gamma.min()) / 2

        if vert < nu.min() or val > nu.max():
            val = (nu.max() + nu.min()) / 2

        R = self.Rs[ind]

        if type(self.lamda) is float:
            x = np.rad2deg(np.arccos([0.5 * (np.trace(r) - 1) for r in R]))
            label = "angle"
        else:
            x = self.lamda
            label = "wavelength"

        mask = (
            (d > d_min)
            & (d < d_max)
            & (self.gamma[:, np.newaxis] > horz - horz_roi)
            & (self.gamma[:, np.newaxis] < horz + horz_roi)
            & (self.nu[:, np.newaxis] > vert - vert_roi)
            & (self.nu[:, np.newaxis] < vert + vert_roi)
        )

        rows, cols = np.nonzero(mask)

        vals = self.counts[ind]

        uni_cols, inv_ind = np.unique(cols, return_inverse=True)

        x = x[uni_cols]
        y = np.bincount(inv_ind, weights=vals[mask])

        if len(x) > 1:
            if val < x.min() or val > x.max():
                val = (x.max() + x.min()) / 2
        else:
            val = 0

        roi_view["horz"] = horz
        roi_view["vert"] = vert
        roi_view["horz_roi"] = horz_roi
        roi_view["vert_roi"] = vert_roi
        roi_view["val"] = val
        roi_view["label"] = label
        roi_view["x"] = x
        roi_view["y"] = y
        roi_view["label"] = label

        self.roi_view = roi_view

    def is_sliced(self):
        """
        Check if the data has been sliced.

        Returns
        -------
        bool
            True if data is sliced, False otherwise.
        """

        return mtd.doesExist("slice")

    def get_slice_info(self, U, V, W, normal, value, thickness, width):
        """
        Get the slicing information for a given normal and value.

        Parameters
        ----------
        U, V, W : list
            Normalized direction cosines for the slice.
        normal : list
            Normal vector for the slice.
        value : float
            Value at which to slice.
        thickness : float
            Thickness of the slice.
        width : float
            Width of the output histogram bins.

        Returns
        -------
        dict
            Dictionary with slice information including extents, bins, and transform.
        """

        UB = self.get_UB()

        if self.has_UB() and self.has_Q():
            Wt = np.column_stack([U, V, W])

            slice_dict = {}

            Bp = np.dot(UB, Wt)

            bp_inv = np.linalg.inv(2 * np.pi * Bp)

            Qx_min, Qy_min, Qz_min = self.min_lim
            Qx_max, Qy_max, Qz_max = self.max_lim

            corners = np.array(
                [
                    [Qx_min, Qy_min, Qz_min],
                    [Qx_max, Qy_min, Qz_min],
                    [Qx_min, Qy_max, Qz_min],
                    [Qx_max, Qy_max, Qz_min],
                    [Qx_min, Qy_min, Qz_max],
                    [Qx_max, Qy_min, Qz_max],
                    [Qx_min, Qy_max, Qz_max],
                    [Qx_max, Qy_max, Qz_max],
                ]
            )

            trans_corners = np.einsum("ij,kj->ki", bp_inv, corners)

            min_values = np.ceil(np.min(trans_corners, axis=0))
            max_values = np.floor(np.max(trans_corners, axis=0))

            bin_sizes = np.round((max_values - min_values) / width).astype(int)

            min_values = min_values.tolist()
            max_values = max_values.tolist()

            bin_sizes = bin_sizes.tolist()

            extents = []
            bins = []

            integrate = [value - thickness, value + thickness]

            for ind, i in enumerate(normal):
                if i == 0:
                    extents += [min_values[ind], max_values[ind]]
                    bins += [1 + bin_sizes[ind]]
                else:
                    extents += integrate
                    bins += [1]

            self.copy_UB_to_peaks()

            ConvertQtoHKLMDHisto(
                InputWorkspace=self.Q,
                PeaksWorkspace=self.table,
                UProj=U,
                VProj=V,
                WProj=W,
                Extents=extents,
                Bins=bins,
                OutputWorkspace="slice",
            )

            CompactMD(InputWorkspace="slice", OutputWorkspace="slice")

            i = np.array(normal).tolist().index(1)

            form = "{} = ({:.2f},{:.2f})"

            title = form.format(mtd["slice"].getDimension(i).name, *integrate)

            dims = mtd["slice"].getNonIntegratedDimensions()

            x, y = [
                np.linspace(
                    dim.getMinimum(), dim.getMaximum(), dim.getNBoundaries()
                )
                for dim in dims
            ]

            labels = [
                "{} ({})".format(dim.name, dim.getUnits()) for dim in dims
            ]

            slice_dict["x"] = x
            slice_dict["y"] = y
            slice_dict["labels"] = labels

            signal = mtd["slice"].getSignalArray().T.copy().squeeze()

            # signal[signal <= 0] = np.nan
            # signal[np.isinf(signal)] = np.nan

            slice_dict["signal"] = signal

            Q, R = scipy.linalg.qr(Bp)

            ind = np.array(normal) != 1

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

    def calculate_clim(self, data, method="normal"):
        """
        Calculate the color limits for the given data using the specified method.

        Parameters
        ----------
        data : np.ndarray
            Input data for which to calculate color limits.
        method : str, optional
            Method for calculating limits ('normal', 'boxplot', or 'manual'). Default is 'normal'.

        Returns
        -------
        tuple
            Lower and upper color limits.
        """

        trans = data.copy()
        trans[~np.isfinite(trans)] = np.nan
        trans[np.isclose(trans, 0)] = np.nan

        vmin, vmax = np.nanmin(trans), np.nanmax(trans)

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

        if np.isclose(cmin, cmax) or cmax < cmin:
            cmin, cmax = vmin, vmax

        clim = [cmin if cmin > vmin else vmin, cmax if cmax < vmax else vmax]

        trans[trans < clim[0]] = clim[0]
        trans[trans > clim[1]] = clim[1]

        return trans

    def get_has_Q_vol(self):
        """
        Check if the Q volume data exists.

        Returns
        -------
        bool
            True if Q volume data exists, False otherwise.
        """

        return mtd.doesExist("Q3D")

    def get_Q_info(self):
        """
        Extract Q-space information from the model.

        Returns
        -------
        dict
            Dictionary containing Q-space signal, min/max limits, spacing, and optionally x, y, z coordinates.
        """

        Q_dict = {}

        if self.get_has_Q_vol():
            Q_dict["signal"] = self.signal
            # Q_dict["opacity"] = self.opacity

            Q_dict["min_lim"] = self.min_lim
            Q_dict["max_lim"] = self.max_lim
            Q_dict["spacing"] = self.spacing

            # Q_dict["x"] = self.x
            # Q_dict["y"] = self.y
            # Q_dict["z"] = self.z

        if self.has_peaks():
            self.sort_peaks_by_hkl(self.table)

            self.sort_peaks_by_d(self.table)

            Qs, Is, inds, pk_nos, Ts, rows = [], [], [], [], [], []

            for j, peak in enumerate(mtd[self.table]):
                T = np.zeros((4, 4))

                I = peak.getIntensity()

                ind = (peak.getHKL().norm2() > 0) * 1.0

                shape = eval(peak.getPeakShape().toJSON())

                pk_no = j

                Q = peak.getQSampleFrame()

                if any(["radius" in key for key in shape.keys()]):
                    if shape.get("radius0") is not None:
                        r, v = [], []
                        for i in range(3):
                            r.append(shape["radius{}".format(i)])
                            v.append(shape["direction{}".format(i)].split(" "))
                        r = np.array(r)
                        v = np.array(v).T.astype(float)

                    else:
                        r = np.array([shape["radius"]] * 3)
                        v = np.eye(3)

                    P = np.dot(v, np.dot(np.diag(r), v.T))

                else:
                    P = np.eye(3) * 0.2

                T[:3, -1] = Q
                T[:3, :3] = P
                T[-1, -1] = 1

                Qs.append(Q)
                Is.append(I)
                inds.append(ind)
                pk_nos.append(pk_no)
                Ts.append(T)
                rows.append(j)

            Q_dict["coordinates"] = Qs
            Q_dict["intensities"] = Is
            Q_dict["indexings"] = inds
            Q_dict["numbers"] = pk_nos
            Q_dict["transforms"] = Ts
            Q_dict["rows"] = rows

        return Q_dict if len(Q_dict.keys()) > 0 else None

    def get_lattice_constants(self):
        """
        Get the lattice constants from the oriented lattice.

        Returns
        -------
        list
            List of lattice constants [a, b, c, alpha, beta, gamma].
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            params = ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

            return np.array(params).round(8).tolist()

    def get_lattice_constant_errors(self):
        """
        Get the errors in the lattice constants from the oriented lattice.

        Returns
        -------
        list
            List of errors in lattice constants [error_a, error_b, error_c, error_alpha, error_beta, error_gamma].
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            params = (
                ol.errora(),
                ol.errorb(),
                ol.errorc(),
                ol.erroralpha(),
                ol.errorbeta(),
                ol.errorgamma(),
            )

            return np.array(params).round(8).tolist()

    def simplify_vector(self, vec):
        """
        Simplify a vector to its primitive integer form.

        Parameters
        ----------
        vec : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Simplified integer vector.
        """

        vec = vec / np.linalg.norm(vec)

        vec *= 10

        vec = np.round(vec).astype(int)

        return vec // np.gcd.reduce(vec)

    def get_sample_directions(self):
        """
        Get the sample directions from the UB matrix.

        Returns
        -------
        list
            List of simplified sample direction vectors.
        """

        if self.has_UB():
            UB = mtd[self.cell].sample().getOrientedLattice().getUB()

            vecs = np.linalg.inv(UB).T

            return [self.simplify_vector(vec) for vec in vecs]

    def copy_UB_from_peaks(self):
        """
        Copy the UB matrix from the peaks workspace to the cell workspace.
        """

        CopySample(
            InputWorkspace=self.table,
            OutputWorkspace=self.cell,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def copy_UB_to_peaks(self):
        """
        Copy the UB matrix from the cell workspace to the peaks workspace.
        """

        CopySample(
            InputWorkspace=self.cell,
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def copy_UB_from_Q(self):
        """
        Copy the UB matrix from the Q workspace to the cell workspace.
        """

        CopySample(
            InputWorkspace=self.Q,
            OutputWorkspace=self.cell,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def copy_UB_to_Q(self):
        """
        Copy the UB matrix from the cell workspace to the Q workspace.
        """

        CopySample(
            InputWorkspace=self.cell,
            OutputWorkspace=self.Q,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def save_UB(self, filename):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        SaveIsawUB(InputWorkspace=self.cell, Filename=filename)

    def load_UB(self, filename):
        """
        Load UB from file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        LoadIsawUB(InputWorkspace=self.cell, Filename=filename)

    def determine_UB_with_niggli_cell(self, min_d, max_d, tol=0.1):
        """
        Determine UB with primitive lattice using min/max lattice constant.

        Parameters
        ----------
        min_d : float
            Minimum lattice parameter in ansgroms.
        max_d : float
            Maximum lattice parameter in angstroms.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingFFT(
            PeaksWorkspace=self.table, MinD=min_d, MaxD=max_d, Tolerance=tol
        )

        self.copy_UB_from_peaks()

        self.update_UB()

        CloneWorkspace(
            InputWorkspace=self.table, OutputWorkspace=self.primitive_cell
        )

    def determine_UB_with_lattice_parameters(
        self, a, b, c, alpha, beta, gamma, tol=0.1
    ):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingLatticeParameters(
            PeaksWorkspace=self.table,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Tolerance=tol,
            NumInitial=150,
            FixParameters=False,
            Iterations=1,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def refine_UB_without_constraints(self, tol=0.1, sat_tol=None):
        """
        Refine UB with unconstrained lattice parameters.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(
            PeaksWorkspace=self.table,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def refine_UB_with_constraints(self, cell, tol=0.1):
        """
        Refine UB with constraints corresponding to lattice system.

        +----------------+---------------+----------------------+
        | Lattice system | Lengths       | Angles               |
        +================+===============+======================+
        | Cubic          | :math:`a=b=c` | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Hexagonal      | :math:`a=b`   | :math:`α=β=90,γ=120` |
        +----------------+---------------+----------------------+
        | Rhombohedral   | :math:`a=b=c` | :math:`α=β=γ`        |
        +----------------+---------------+----------------------+
        | Tetragonal     | :math:`a=b`   | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Orthorhombic   | None          | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Monoclinic     | None          | :math:`α=γ=90`       |
        +----------------+---------------+----------------------+
        | Triclinic      | None          | None                 |
        +----------------+---------------+----------------------+

        Parameters
        ----------
        cell : float
            Lattice system.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        self.copy_UB_to_peaks()

        OptimizeLatticeForCellType(
            PeaksWorkspace=self.table, CellType=cell, Apply=True, Tolerance=tol
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def refine_U_only(self, a, b, c, alpha, beta, gamma):
        """
        Refine the U orientation only.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        self.copy_UB_to_peaks()

        CalculateUMatrix(
            PeaksWorkspace=self.table,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def select_cell(self, number, tol=0.1):
        """
        Transform to conventional cell using form number.

        Parameters
        ----------
        number : int
            Form number.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        CopySample(
            InputWorkspace=self.primitive_cell,
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        SelectCellWithForm(
            PeaksWorkspace=self.table,
            FormNumber=number,
            Apply=True,
            Tolerance=tol,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def possible_conventional_cells(self, max_error=0.2, permutations=True):
        """
        List possible conventional cells.

        Parameters
        ----------
        max_error : float, optional
            Max scalar error to report form numbers. The default is 0.2.
        permutations : bool, optional
            Allow permutations of the lattice. The default is True.

        Returns
        -------
        vals : list
            List of form results.

        """

        CopySample(
            InputWorkspace=self.primitive_cell,
            OutputWorkspace=self.table,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

        result = ShowPossibleCells(
            PeaksWorkspace=self.table,
            MaxScalarError=max_error,
            AllowPermutations=permutations,
            BestOnly=False,
        )

        vals = [json.loads(cell) for cell in result.Cells]

        cells = []
        for i, val in enumerate(vals):
            form = val["FormNumber"]
            error = val["Error"]
            cell = val["CellType"]
            centering = val["Centering"]
            bravais = cell, centering
            a = val["a"]
            b = val["b"]
            c = val["c"]
            alpha = val["alpha"]
            beta = val["beta"]
            gamma = val["gamma"]
            vol = val["volume"]
            params = a, b, c, alpha, beta, gamma, vol
            cell = form, error, bravais, params
            cells.append(cell)

        return cells

    def transform_lattice(self, transform, tol=0.1):
        """
        Apply a cell transformation to the lattice.

        Parameters
        ----------
        transform : 3x3 array-like
            Transform to apply to hkl values.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        hkl_trans = ",".join(9 * ["{}"]).format(*transform)

        TransformHKL(
            PeaksWorkspace=self.table,
            Tolerance=tol,
            HKLTransform=hkl_trans,
            FindError=mtd[self.table].getNumberPeaks() > 3,
        )

        self.copy_UB_from_peaks()

        self.update_UB()

    def generate_lattice_transforms(self, cell):
        """
        Obtain possible transforms compatabile with a unit cell lattice.

        Parameters
        ----------
        cell : str
            Latttice system.

        Returns
        -------
        transforms : dict
            Transform dictionary with symmetry operation as key.

        """

        symbol = lattice_group[cell]

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
                transform[name] = T

        return {key: transform[key] for key in sorted(transform.keys())}

    def index_peaks(
        self,
        tol=0.1,
        sat_tol=None,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
        round_hkl=True,
    ):
        """
        Index the peaks and calculate the lattice parameter uncertainties.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.
        round_hkl : bool, optional
            Round integers to integer. The default is True.

        Returns
        -------
        indexing : list
            Result of indexing including number indexed and errors.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        save = True if max_order > 0 else False

        indexing = IndexPeaks(
            PeaksWorkspace=self.table,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
            RoundHKLs=round_hkl,
            CommonUBForAll=True,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            SaveModulationInfo=save,
        )

        return indexing

    def calculate_hkl(self):
        """
        Calculate hkl values without rounding.

        """

        CalculatePeaksHKL(PeaksWorkspace=self.table, OverWrite=True)

    def find_peaks(self, min_dist, density=1000, max_peaks=50, edge_pixels=0):
        """
        Harvest strong peak locations from Q-sample into a peaks table.

        Parameters
        ----------
        min_dist : float
            Minimum distance enforcing lower limit of peak spacing.
        density : int, optional
            Threshold density. The default is 1000.
        max_peaks : int, optional
            Maximum number of peaks to find. The default is 50.
        edge_pixels: int, optional
            Nnumber of edge pixels to exclude. The default is 0.

        """

        FindPeaksMD(
            InputWorkspace=self.Q,
            PeakDistanceThreshold=min_dist,
            MaxPeaks=max_peaks,
            PeakFindingStrategy="VolumeNormalization",
            DensityThresholdFactor=density,
            EdgePixels=edge_pixels,
            OutputWorkspace=self.table,
        )

        self.integrate_peaks(min_dist, 1, 1, method="sphere", centroid=False)

        self.clear_intensity()

        self.copy_UB_to_peaks()

    def centroid_peaks(self, peak_radius):
        """
        Re-center peak locations using centroid within given radius

        Parameters
        ----------
        peak_radius : float
            Integration region radius.

        """

        CentroidPeaksMD(
            InputWorkspace=self.Q,
            PeakRadius=peak_radius,
            PeaksWorkspace=self.table,
            OutputWorkspace=self.table,
        )

    def integrate_peaks(
        self,
        peak_radius,
        background_inner_fact=1,
        background_outer_fact=1.5,
        method="sphere",
        centroid=True,
    ):
        """
        Integrate peaks using spherical or ellipsoidal regions.
        Ellipsoid integration adapts itself to the peak distribution.

        Parameters
        ----------
        peak_radius : float
            Integration region radius.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        method : str, optional
            Integration method. The default is 'sphere'.
        centroid : str, optional
            Shift peak position to centroid. The default is True.

        """

        background_inner_radius = peak_radius * background_inner_fact
        background_outer_radius = peak_radius * background_outer_fact

        if method == "sphere" and centroid:
            self.centroid_peaks(peak_radius)

        IntegratePeaksMD(
            InputWorkspace=self.Q,
            PeaksWorkspace=self.table,
            PeakRadius=peak_radius,
            BackgroundInnerRadius=background_inner_radius,
            BackgroundOuterRadius=background_outer_radius,
            Ellipsoid=True if method == "ellipsoid" else False,
            FixQAxis=False,
            FixMajorAxisLength=False,
            UseCentroid=True,
            MaxIterations=3,
            ReplaceIntensity=True,
            IntegrateIfOnEdge=True,
            AdaptiveQBackground=False,
            MaskEdgeTubes=False,
            OutputWorkspace=self.table,
        )

    def clear_intensity(self):
        """
        Clear the intensity values of all peaks in the peaks table.
        """

        for peak in mtd[self.table]:
            peak.setIntensity(0)
            peak.setSigmaIntensity(0)

    def get_max_d_spacing(self, ws):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Parameters
        ----------
        ws : str
            Workspace with UB defined on oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=ws):
            if hasattr(mtd[ws], "sample"):
                ol = mtd[ws].sample().getOrientedLattice()
            else:
                for i in range(mtd[ws].getNumExperimentInfo()):
                    sample = mtd[ws].getExperimentInfo(i).sample()
                    if sample.hasOrientedLattice():
                        ol = sample.getOrientedLattice()
                        SetUB(Workspace=ws, UB=ol.getUB())
                ol = mtd[ws].getExperimentInfo(i).sample().getOrientedLattice()

            return 1 / min([ol.astar(), ol.bstar(), ol.cstar()])

    def predict_peaks(
        self, centering, d_min, lamda_min, lamda_max, edge_pixels=0
    ):
        """
        Predict peak Q-sample locations with UB and lattice centering.

        +--------+-----------------------+
        | Symbol | Reflection condition  |
        +========+=======================+
        | P      | None                  |
        +--------+-----------------------+
        | I      | :math:`h+k+l=2n`      |
        +--------+-----------------------+
        | F      | :math:`h,k,l` unmixed |
        +--------+-----------------------+
        | R      | None                  |
        +--------+-----------------------+
        | R(obv) | :math:`-h+k+l=3n`     |
        +--------+-----------------------+
        | R(rev) | :math:`h-k+l=3n`      |
        +--------+-----------------------+
        | A      | :math:`k+l=2n`        |
        +--------+-----------------------+
        | B      | :math:`l+h=2n`        |
        +--------+-----------------------+
        | C      | :math:`h+k=2n`        |
        +--------+-----------------------+

        Parameters
        ----------
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        edge_pixels: int, optional
            Nnumber of edge pixels to exclude. The default is 0.

        """

        self.copy_UB_to_peaks()

        d_max = self.get_max_d_spacing(self.table)

        self.copy_UB_to_Q()

        PredictPeaks(
            InputWorkspace=self.Q,
            WavelengthMin=lamda_min,
            WavelengthMax=lamda_max,
            MinDSpacing=d_min,
            MaxDSpacing=d_max * 1.2,
            ReflectionCondition=centering_reflection[centering],
            EdgePixels=edge_pixels,
            OutputWorkspace=self.table,
        )

        self.integrate_peaks(0.1, 1, 1, method="sphere", centroid=False)

        self.clear_intensity()

    def predict_modulated_peaks(
        self,
        d_min,
        lamda_min,
        lamda_max,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Predict the modulated peak positions based on main peaks.

        Parameters
        ----------
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        self.copy_UB_to_peaks()

        d_max = self.get_max_d_spacing(self.table)

        sat_peaks = self.table + "_sat"

        PredictSatellitePeaks(
            Peaks=self.table,
            SatellitePeaks=sat_peaks,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            IncludeIntegerHKL=False,
            IncludeAllPeaksInRange=True,
            WavelengthMin=lamda_min,
            WavelengthMax=lamda_max,
            MinDSpacing=d_min,
            MaxDSpacing=d_max * 1.2,
        )

        CombinePeaksWorkspaces(
            LHSWorkspace=self.table,
            RHSWorkspace=sat_peaks,
            OutputWorkspace=self.table,
        )

        DeleteWorkspace(Workspace=sat_peaks)

    def predict_satellite_peaks(
        self,
        lamda_min,
        lamda_max,
        d_min,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Locate satellite peaks from goniometer angles.

        Parameters
        ----------
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min : float
            Minimum wavelength.
        lamda_max : float
            Maximum wavelength.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        Rs = self.get_all_goniometer_matrices(self.Q)

        for R in Rs:
            self.set_goniometer(self.table, R)

            self.predict_modulated_peaks(
                d_min,
                lamda_min,
                lamda_max,
                mod_vec_1,
                mod_vec_2,
                mod_vec_3,
                max_order,
                cross_terms,
            )

            self.remove_duplicate_peaks(self.table)

    def sort_peaks_by_hkl(self, peaks):
        """
        Sort peaks table by descending hkl values.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace=peaks,
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace=peaks,
            )

    def sort_peaks_by_d(self, peaks):
        """
        Sort peaks table by descending d-spacing.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=peaks,
        )

    def remove_duplicate_peaks(self, peaks):
        """
        Omit duplicate peaks from different based on indexing.
        Table will be sorted.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        self.sort_peaks_by_hkl(peaks)

        for no in range(mtd[peaks].getNumberPeaks() - 1, 0, -1):
            if (
                mtd[peaks].getPeak(no).getHKL()
                - mtd[peaks].getPeak(no - 1).getHKL()
            ).norm2() == 0:
                DeleteTableRows(TableWorkspace=peaks, Rows=no)

    def get_all_goniometer_matrices(self, ws):
        """
        Extract all goniometer matrices.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.

        Returns
        -------
        Rs: list
            Goniometer matrices.

        """

        Rs = []

        for ei in range(mtd[ws].getNumExperimentInfo()):
            run = mtd[ws].getExperimentInfo(ei).run()

            n_gon = run.getNumGoniometers()

            Rs += [run.getGoniometer(i).getR() for i in range(n_gon)]

        return np.array(Rs)

    def renumber_runs_by_index(self, ws, peaks):
        """
        Re-label the runs by index based on goniometer setting.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.
        peaks : str
            Name of peaks table.

        """

        Rs = self.get_all_goniometer_matrices(ws)

        for no in range(mtd[peaks].getNumberPeaks()):
            peak = mtd[peaks].getPeak(no)

            R = peak.getGoniometerMatrix()

            ind = np.isclose(Rs, R).all(axis=(1, 2))
            i = -1 if not np.any(ind) else ind.tolist().index(True)

            peak.setRunNumber(i + 1)

    def load_Q(self, filename):
        """
        Load Q file.

        Parameters
        ----------
        filename : str
            Name of Q file with extension .nxs.

        """

        LoadMD(Filename=filename, OutputWorkspace=self.Q)

    def save_Q(self, filename):
        """
        Save Q file.

        Parameters
        ----------
        filename : str
            Name of Q file with extension .nxs.

        """

        SaveMD(Filename=filename, InputWorkspace=self.Q)

    def load_peaks(self, filename):
        """
        Load peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file (can be .nxs or .integrate, .peaks, etc.).
        """

        if filename.endswith(".nxs"):
            LoadNexus(Filename=filename, OutputWorkspace=self.table)
        else:
            LoadIsawPeaks(Filename=filename, OutputWorkspace=self.table)

    def save_peaks(self, filename):
        """
        Save the current peaks table to a file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        """

        SaveNexus(Filename=filename, InputWorkspace=self.table)

    def delete_peaks(self, peaks):
        """
        Remove peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.

        """

        if mtd.doesExist(peaks):
            DeleteWorkspace(Workspace=peaks)

    def filter_peaks(self, name, operator, value):
        """
        Filter out peaks based on value and operator.

        Parameters
        ----------
        name : str
            Filter name.
        operator : float
            Filter operator.
        value : float
            The filter value.

        """

        FilterPeaks(
            InputWorkspace=self.table,
            OutputWorkspace=self.table,
            FilterVariable=variable[name],
            FilterValue=value,
            Operator=operator,
            Criterion="!=",
            BankName="None",
        )

    def get_d_min(self):
        d_min = 0.7
        if self.has_peaks():
            for peak in mtd[self.table]:
                d_spacing = peak.getDSpacing()
                if d_spacing < d_min:
                    d_min = d_spacing
        return d_min

    def avoid_aluminum_contamination(self, d_min, d_max, delta=0.1):
        aluminum = CrystalStructure(
            "4.05 4.05 4.05", "F m -3 m", "Al 0 0 0 1.0 0.005"
        )

        generator = ReflectionGenerator(aluminum)

        hkls = generator.getUniqueHKLsUsingFilter(
            d_min, d_max, ReflectionConditionFilter.StructureFactor
        )

        ds = list(generator.getDValues(hkls))

        if self.has_peaks():
            for peak in mtd[self.table]:
                d_spacing = peak.getDSpacing()
                Q_mod = 2 * np.pi / d_spacing
                for d in ds:
                    Q = 2 * np.pi / d
                    if Q - delta < Q_mod < Q + delta or d_spacing > d_max:
                        peak.setRunNumber(-1)

            FilterPeaks(
                InputWorkspace=self.table,
                OutputWorkspace=self.table,
                FilterVariable="RunNumber",
                FilterValue="-1",
                Operator="!=",
                Criterion="!=",
                BankName="None",
            )

    def get_modulation_info(self):
        if self.has_peaks() and self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            return [ol.getModVec(i) for i in range(3)]

    def get_peak_info(self):
        """
        Extract detailed information for each peak in the current table.

        Returns
        -------
        list of dict
            List of dictionaries with peak properties (hkl, d-spacing, intensity, etc.).
        """

        if self.has_peaks():
            banks = mtd[self.table].column("BankName")

            peak_info = []
            for i, peak in enumerate(mtd[self.table]):
                peak_data = {
                    "hkl": list(peak.getHKL()),
                    "d_spacing": peak.getDSpacing(),
                    "wavelength": peak.getWavelength(),
                    "intensity": peak.getIntensity(),
                    "signal_to_noise": peak.getIntensityOverSigma(),
                    "sigma": peak.getSigmaIntensity(),
                    "int_hkl": list(peak.getIntHKL()),
                    "int_mnp": list(peak.getIntMNP()),
                    "run_number": peak.getRunNumber(),
                    "bank": banks[i],
                    "row": peak.getRow(),
                    "col": peak.getCol(),
                    "ind": peak.getHKL().norm2() > 0,
                    "Q": list(peak.getQSampleFrame()),
                    "peak_no": i,
                }
                peak_info.append(peak_data)
                peak.setPeakNumber(i)

            self.peak_info = peak_info

            return peak_info

    def get_peak(self, i):
        """
        Get a specific peak's information by index.

        Parameters
        ----------
        i : int
            Index of the peak.

        Returns
        -------
        dict or None
            Peak information dictionary or None if index is out of range.
        """

        if self.peak_info is not None:
            if i >= 0 and i < len(self.peak_info):
                return self.peak_info[i]

    def calculate_fractional(
        self, mod_vec_1, mod_vec_2, mod_vec_3, int_hkl, int_mnp
    ):
        """
        Calculate the fractional coordinates from the given modulation vectors and integer coordinates.

        Parameters
        ----------
        mod_vec_1, mod_vec_2, mod_vec_3 : list
            Modulation vectors.
        int_hkl : list
            Integer HKL coordinates.
        int_mnp : list
            Integer MNP coordinates.

        Returns
        -------
        np.ndarray
            Calculated fractional coordinates.
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            ol.setModVec1(V3D(*mod_vec_1))
            ol.setModVec2(V3D(*mod_vec_2))
            ol.setModVec3(V3D(*mod_vec_3))

        delta_hkl = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        return np.array(int_hkl) + np.dot(delta_hkl, int_mnp)

    def calculate_integer(self, mod_vec_1, mod_vec_2, mod_vec_3, hkl):
        """
        Calculate the integer coordinates from the given modulation vectors and HKL coordinates.

        Parameters
        ----------
        mod_vec_1, mod_vec_2, mod_vec_3 : list
            Modulation vectors.
        hkl : list
            HKL coordinates.

        Returns
        -------
        tuple
            Best integer coordinates (int_hkl, int_mnp) that minimize the error.
        """

        if self.has_UB():
            ol = mtd[self.cell].sample().getOrientedLattice()

            ol.setModVec1(V3D(*mod_vec_1))
            ol.setModVec2(V3D(*mod_vec_2))
            ol.setModVec3(V3D(*mod_vec_3))

        delta_hkl = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        bounds_m = range(-3, 4) if np.linalg.norm(mod_vec_1) > 0 else [0]
        bounds_n = range(-3, 4) if np.linalg.norm(mod_vec_2) > 0 else [0]
        bounds_p = range(-3, 4) if np.linalg.norm(mod_vec_3) > 0 else [0]

        min_error = np.inf

        for m in bounds_m:
            for n in bounds_n:
                for p in bounds_p:
                    int_mnp = np.array([m, n, p])
                    residual = np.array(hkl) - np.dot(delta_hkl, int_mnp)
                    int_hkl = np.round(residual).astype(int)
                    model = int_hkl + np.dot(delta_hkl, int_mnp)
                    error = np.linalg.norm(hkl - model)
                    if error < min_error:
                        min_error = error
                        best_solution = (int_hkl, int_mnp)

        return best_solution

    def set_peak(self, i, hkl, int_hkl, int_mnp):
        """
        Set the HKL, integer HKL, and integer MNP values for a specific peak.

        Parameters
        ----------
        i : int
            Index of the peak.
        hkl : list
            HKL coordinates.
        int_hkl : list
            Integer HKL coordinates.
        int_mnp : list
            Integer MNP coordinates.
        """

        peak = mtd[self.table].getPeak(i)

        peak.setHKL(*hkl)
        peak.setIntHKL(V3D(*np.array(int_hkl).astype(float).tolist()))
        peak.setIntMNP(V3D(*np.array(int_mnp).astype(float).tolist()))

    def calculate_peaks(self, hkl_1, hkl_2, a, b, c, alpha, beta, gamma):
        """
        Calculate d-spacing and angle between two HKL planes.

        Parameters
        ----------
        hkl_1, hkl_2 : list
            Miller indices for the two planes.
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        Returns
        -------
        tuple
            d-spacing for the two planes and the angle between them.
        """

        uc = UnitCell(a, b, c, alpha, beta, gamma)

        d_1 = d_2 = phi_12 = None
        if hkl_1 is not None:
            d_1 = uc.d(*hkl_1)
        if hkl_2 is not None:
            d_2 = uc.d(*hkl_2)
        if hkl_1 is not None and hkl_2 is not None:
            phi_12 = uc.recAngle(*hkl_1, *hkl_2)

        return d_1, d_2, phi_12

    def cluster_peaks(self, peak_info, eps=0.025, min_samples=15):
        """
        Cluster peaks using DBSCAN algorithm.

        Parameters
        ----------
        peak_info : dict
            Dictionary containing peak information (coordinates, inverse transform, etc.).
        eps : float, optional
            The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default is 0.025.
        min_samples : int, optional
            The number of samples in a neighborhood for a point to be considered as a core point. Default is 15.

        Returns
        -------
        bool
            True if clustering is successful, False otherwise.
        """

        T_inv = peak_info["inverse"]

        points = np.array(peak_info["coordinates"])

        clustering = DBSCAN(eps=eps, min_samples=min_samples)

        labels = clustering.fit_predict(points)

        uni_labels, inverse = np.unique(labels, return_inverse=True)

        centroids = []
        for label in uni_labels:
            if label >= 0:
                center = points[labels == label].mean(axis=0)
                centroids.append(np.dot(T_inv, center))
        centroids = np.array(centroids)

        success = False

        if centroids.shape[0] >= 0 and len(centroids.shape) == 2:
            null = np.argmin(np.linalg.norm(centroids, axis=1))

            mask = np.ones_like(centroids[:, 0], dtype=bool)
            mask[null] = False

            peaks = np.arange(mask.size)[mask]

            satellites = centroids[mask]
            nuclear = centroids[null]

            dist = scipy.spatial.distance_matrix(satellites, -satellites)

            n = dist.shape[0]

            if n > 2:
                success = True

                indices = np.column_stack(
                    [np.arange(n), np.argmin(dist, axis=0)]
                )
                indices = np.sort(indices, axis=1)
                indices = np.unique(indices, axis=0)

                clusters = labels.copy()
                clusters[labels == null] = 0

                mod = 1
                satellites = []
                for inds in indices:
                    i, j = peaks[inds[0]], peaks[inds[1]]
                    clusters[labels == i] = mod
                    clusters[labels == j] = mod
                    satellites.append(centroids[i])
                    mod += 1
                satellites = np.array(satellites)

                peak_info["clusters"] = clusters
                peak_info["nuclear"] = nuclear
                peak_info["satellites"] = satellites

        return success

    def get_cluster_info(self):
        """
        Get cluster information for peaks based on UB and peak table.

        Returns
        -------
        dict
            Dictionary with cluster coordinates, points, numbers, translation vectors, and transforms.
        """

        if self.has_UB() and self.has_peaks():
            UB = self.get_UB()
            peak_dict = {}

            Qs, HKLs, pk_nos = [], [], []

            for j, peak in enumerate(mtd[self.table]):
                pk_no = j + 1

                diff_HKL = peak.getHKL() - np.round(peak.getHKL())

                Q = 2 * np.pi * np.dot(UB, diff_HKL)

                Qs.append(Q)
                HKLs.append(diff_HKL)
                pk_nos.append(pk_no)

                diff_HKL = peak.getHKL() - np.round(peak.getHKL())

                Q = 2 * np.pi * np.dot(UB, -diff_HKL)

                Qs.append(Q)
                HKLs.append(diff_HKL)
                pk_nos.append(-pk_no)

            peak_dict["coordinates"] = Qs
            peak_dict["points"] = HKLs
            peak_dict["numbers"] = pk_nos

            translation = (
                2 * np.pi * UB[:, 0],
                2 * np.pi * UB[:, 1],
                2 * np.pi * UB[:, 2],
            )

            peak_dict["translation"] = translation

            T = np.column_stack(translation)

            peak_dict["transform"] = T
            peak_dict["inverse"] = np.linalg.inv(T)

            return peak_dict

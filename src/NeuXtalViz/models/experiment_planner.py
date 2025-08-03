"""
experiment_planner.py
---------------------

This module provides the ExperimentModel and CrystalPlan classes for planning and simulating crystallographic experiments in NeuXtalViz.

ExperimentModel manages instrument setup, calibration, sample and plan creation, UB matrix handling, and experiment simulation, interfacing with Mantid for data processing. CrystalPlan implements a genetic algorithm for optimizing experiment plans based on completeness and redundancy of predicted reflections.

Classes
-------
ExperimentModel
    Main model for experiment planning, instrument setup, and simulation.
CrystalPlan
    Genetic algorithm-based optimizer for experiment plans.
"""

import os

import csv
import itertools

from mantid.simpleapi import (
    CreatePeaksWorkspace,
    ConvertPeaksWorkspace,
    PredictPeaks,
    FilterPeaks,
    CombinePeaksWorkspaces,
    SortPeaksWorkspace,
    AddPeakHKL,
    CountReflections,
    HFIRCalculateGoniometer,
    SetUB,
    SetGoniometer,
    LoadNexus,
    SaveNexus,
    LoadIsawUB,
    LoadEmptyInstrument,
    LoadInstrument,
    LoadMask,
    LoadIsawDetCal,
    LoadParameterFile,
    MaskDetectors,
    ExtractMonitors,
    PreprocessDetectorsToMD,
    GroupDetectors,
    MaskBTP,
    AddSampleLog,
    CreateSampleWorkspace,
    CreateEmptyTableWorkspace,
    DeleteTableRows,
    CloneWorkspace,
    DeleteWorkspace,
    RenameWorkspaces,
    HasUB,
    mtd,
)

from mantid.kernel import V3D
from mantid.geometry import PointGroupFactory

from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from NeuXtalViz.models.base_model import NeuXtalVizModel
from NeuXtalViz.config.instruments import beamlines

point_group_centering = {
    "1": ["P"],
    "-1": ["P"],
    "2": ["P", "C"],
    "m": ["P", "C"],
    "2/m": ["P", "C"],
    "112": ["P", "C"],
    "11m": ["P", "C"],
    "112/m": ["P", "C"],
    "222": ["P", "I", "F", "C", "A", "B"],
    "mm2": ["P", "I", "F", "C", "A", "B"],
    "mmm": ["P", "I", "F", "C", "A", "B"],
    "4": ["P", "I"],
    "-4": ["P", "I"],
    "4/m": ["P", "I"],
    "422": ["P", "I"],
    "4mm": ["P", "I"],
    "-42m": ["P", "I"],
    "-4m2": ["P", "I"],
    "4/mmm": ["P", "I"],
    "3 r": ["P"],
    "-3 r": ["P"],
    "32 r": ["P"],
    "3m r": ["P"],
    "-3m r": ["P"],
    "3": ["Robv", "Rrev"],
    "-3": ["Robv", "Rrev"],
    "312": ["Robv", "Rrev"],
    "31m": ["Robv", "Rrev"],
    "32": ["Robv", "Rrev"],
    "321": ["Robv", "Rrev"],
    "3m": ["Robv", "Rrev"],
    "-31m": ["Robv", "Rrev"],
    "-3m": ["Robv", "Rrev"],
    "-3m1": ["Robv", "Rrev"],
    "6": ["P"],
    "-6": ["P"],
    "6/m": ["P"],
    "622": ["P"],
    "6mm": ["P"],
    "-62m": ["P"],
    "-6m2": ["P"],
    "6/mmm": ["P"],
    "23": ["P", "I", "F"],
    "m-3": ["P", "I", "F"],
    "432": ["P", "I", "F"],
    "-43m": ["P", "I", "F"],
    "m-3m": ["P", "I", "F"],
}

crystal_system_point_groups = {
    "Triclinic": ["1", "-1"],
    "Monoclinic": ["2", "m", "2/m", "112", "11m", "112/m"],
    "Orthorhombic": ["222", "mm2", "mmm"],
    "Tetragonal": ["4", "-4", "4/m", "422", "4mm", "-42m", "-4m2", "4/mmm"],
    "Trigonal/Rhombohedral": ["3 r", "-3 r", "32 r", "3m r", "-3m r"],
    "Trigonal/Hexagonal": [
        "3",
        "-3",
        "312",
        "31m",
        "32",
        "321",
        "3m",
        "-31m",
        "-3m",
        "-3m1",
    ],
    "Hexagonal": ["6", "-6", "6/m", "622", "6mm", "-62m", "-6m2", "6/mmm"],
    "Cubic": ["23", "m-3", "432", "-43m", "m-3m"],
}

centering_conditions = {
    "P": lambda h, k, l: True,
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: (h % 2 == k % 2 == l % 2),
    "C": lambda h, k, l: k % 2 == 0 and l % 2 == 0,
    "A": lambda h, k, l: h % 2 == 0 and l % 2 == 0,
    "B": lambda h, k, l: h % 2 == 0 and k % 2 == 0,
    "R": lambda h, k, l: True,
    "Robv": lambda h, k, l: (-h + k + l) % 3 == 0,
    "Rrev": lambda h, k, l: (h - k + l) % 3 == 0,
}


class ExperimentModel(NeuXtalVizModel):
    """
    Model for experiment planning, instrument setup, and simulation in NeuXtalViz.

    Handles instrument initialization, calibration, sample and plan creation, UB matrix management, and experiment simulation. Interfaces with Mantid for data processing and supports experiment plan optimization.
    """

    def __init__(self):
        """
        Initialize the ExperimentModel and create the initial coverage workspace.
        """
        super(ExperimentModel, self).__init__()

        CreatePeaksWorkspace(
            NumberOfPeaks=0,
            OutputType="LeanElasticPeak",
            OutputWorkspace="coverage",
        )

    def initialize_instrument(self, instrument, logs, cal, mask):
        """
        Set up the instrument workspace, apply calibration and masking, and prepare detector grouping.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        logs : dict
            Sample log values.
        cal : str
            Path to calibration file.
        mask : str
            Path to mask file.
        """

        inst = self.get_instrument_name(instrument)

        if not mtd.doesExist("instrument"):
            LoadEmptyInstrument(
                InstrumentName=inst, OutputWorkspace="instrument"
            )

            for key in logs.keys():
                AddSampleLog(
                    Workspace="instrument",
                    LogName=key,
                    LogText=str(logs[key]),
                    LogType="Number Series",
                    NumberType="Double",
                )

            if len(logs.keys()) > 0:
                LoadInstrument(
                    Workspace="instrument",
                    RewriteSpectraMap=False,
                    InstrumentName=inst,
                )

            if cal != "" and os.path.exists(cal):
                if os.path.splitext(cal)[1] == ".xml":
                    LoadParameterFile(Workspace="instrument", Filename=cal)
                else:
                    LoadIsawDetCal(InputWorkspace="instrument", Filename=cal)

            if mask != "" and os.path.exists(mask):
                if not mtd.doesExist("mask"):
                    LoadMask(
                        Instrument=inst, InputFile=mask, OutputWOrkspace="mask"
                    )
                MaskDetectors(Workspace="instrument", MaskedWorkspace="mask")

            ExtractMonitors(
                InputWorkspace="instrument",
                MonitorWorkspace="monitors",
                DetectorWorkspace="instrument",
            )

            cols, rows = beamlines[instrument]["BankPixels"]
            mask_cols, mask_rows = beamlines[instrument]["MaskEdges"]

            MaskBTP(
                Workspace="instrument",
                Instrument=inst,
                Tube="0-{},{}-{}".format(mask_cols, cols - mask_cols, cols),
            )

            MaskBTP(
                Workspace="instrument",
                Instrument=inst,
                Pixel="0-{},{}-{}".format(mask_rows, rows - mask_rows, rows),
            )

            banks = beamlines[instrument]["MaskBanks"]

            for bank in banks:
                MaskBTP(
                    Workspace="instrument",
                    Instrument=inst,
                    Bank=bank,
                )

            PreprocessDetectorsToMD(
                InputWorkspace="instrument", OutputWorkspace="detectors"
            )

            grouping = beamlines[instrument]["Grouping"]

            c, r = [int(val) for val in grouping.split("x")]
            shape = (-1, cols, rows)

            det_map = np.array(mtd["detectors"].column(5)).reshape(*shape)

            shape = det_map.shape
            i, j, k = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
            mask = np.array(mtd["detectors"].column(7)) == 0

            keys = np.stack((i, j // c, k // r), axis=-1)
            keys_flat = keys.reshape(-1, keys.shape[-1])
            det_map_flat = det_map.ravel()[mask].astype(str)
            grouped_ids = defaultdict(list)

            for key, detector_id in zip(map(tuple, keys_flat), det_map_flat):
                grouped_ids[key].append(detector_id)

            detector_list = ",".join(
                "+".join(group) for group in grouped_ids.values()
            )

            GroupDetectors(
                InputWorkspace="instrument",
                OutputWorkspace="instrument",
                GroupingPattern=detector_list,
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace="instrument",
                NumberOfPeaks=0,
                OutputType="LeanElasticPeak",
                OutputWorkspace="peak",
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace="instrument",
                NumberOfPeaks=0,
                OutputType="Peak",
                OutputWorkspace="peaks",
            )

            CreatePeaksWorkspace(
                InstrumentWorkspace="instrument",
                NumberOfPeaks=0,
                OutputType="Peak",
                OutputWorkspace="combined",
            )

            L2 = np.array(mtd["detectors"].column(1))[mask]
            tt = np.array(mtd["detectors"].column(2))[mask]
            az = np.array(mtd["detectors"].column(3))[mask]
            det_ID = np.array(mtd["detectors"].column(4))[mask]

            x = L2 * np.sin(tt) * np.cos(az)
            y = L2 * np.sin(tt) * np.sin(az)
            z = L2 * np.cos(tt)

            self.det_ID = det_ID.copy()
            self.nu = np.rad2deg(np.arcsin(y / L2))
            self.gamma = np.rad2deg(np.arctan2(x, z))

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

    def remove_instrument(self):
        """
        Remove instrument and related workspaces from Mantid.
        """

        if mtd.doesExist("instrument"):
            DeleteWorkspace(Workspace="instrument")

        if mtd.doesExist("cobmined"):
            DeleteWorkspace(Workspace="cobmined")

        if mtd.doesExist("filtered"):
            DeleteWorkspace(Workspace="filtered")

    def get_crystal_system_point_groups(self, crystal_system):
        """
        Get point groups for a given crystal system.

        Parameters
        ----------
        crystal_system : str
            Crystal system name.

        Returns
        -------
        list
            List of point group names.
        """

        return crystal_system_point_groups[crystal_system]

    def get_point_group_centering(self, point_group):
        """
        Get centering types for a given point group.

        Parameters
        ----------
        point_group : str
            Point group name.

        Returns
        -------
        list
            List of centering types.
        """

        return point_group_centering[point_group]

    def get_symmetry(self, point_group, centering):
        """
        Get symmetry tuple for Mantid algorithms.

        Parameters
        ----------
        point_group : str
            Point group name.
        centering : str
            Centering type.

        Returns
        -------
        tuple
            (point_group, centering) as strings.
        """

        return str(point_group), str(centering)

    def create_plan(self, table):
        """
        Create a Mantid table workspace for the experiment plan.

        Parameters
        ----------
        table : tuple
            Plan table data (pv, names, titles, settings, comments, counts, values, use).
        """

        pv, names, titles, settings, comments, counts, values, use = table

        CreateEmptyTableWorkspace(OutputWorkspace="plan")

        mtd["plan"].addColumn("str", pv)

        for name in names:
            mtd["plan"].addColumn("float", name)

        mtd["plan"].addColumn("str", "Comment")
        mtd["plan"].addColumn("str", "Wait For")
        mtd["plan"].addColumn("float", "Value")
        mtd["plan"].addColumn("bool", "Use")

        for title, setting, comment, count, value, active in zip(
            titles, settings, comments, counts, values, use
        ):
            row = {}
            row[pv] = title
            for angle, name in zip(setting, names):
                row[name] = np.round(angle, 2)
            row["Comment"] = comment
            row["Wait For"] = count
            row["Value"] = value
            row["Use"] = active
            mtd["plan"].addRow(row)

    def create_sample(self, instrument, mode, UB, wavelength, d_min):
        """
        Create a sample workspace and set sample logs for experiment simulation.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Experiment mode.
        UB : array-like
            UB matrix.
        wavelength : tuple
            (min, max) wavelength.
        d_min : float
            Minimum d-spacing.
        """

        CreateSampleWorkspace(OutputWorkspace="sample")

        SetUB(Workspace="sample", UB=UB)

        AddSampleLog(
            Workspace="sample",
            LogName="instrument",
            LogText=instrument,
            LogType="String",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="mode",
            LogText=mode,
            LogType="String",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="lamda_min",
            LogText=str(wavelength[0]),
            LogType="Number",
            NumberType="Double",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="lamda_max",
            LogText=str(wavelength[1]),
            LogType="Number",
            NumberType="Double",
        )

        AddSampleLog(
            Workspace="sample",
            LogName="d_min",
            LogText=str(d_min),
            LogType="Number",
            NumberType="Double",
        )

    def update_sample(self, crytsal_system, point_group, lattice_centering):
        """
        Update sample logs for crystal system, point group, and centering.
        """

        if mtd.doesExist("sample"):
            AddSampleLog(
                Workspace="sample",
                LogName="crystal_system",
                LogText=crytsal_system,
                LogType="String",
            )

            AddSampleLog(
                Workspace="sample",
                LogName="point_group",
                LogText=point_group,
                LogType="String",
            )

            AddSampleLog(
                Workspace="sample",
                LogName="lattice_centering",
                LogText=lattice_centering,
                LogType="String",
            )

    def update_goniometer_motors(self, limits, motors, cal, mask):
        """
        Update sample logs for goniometer motor limits and calibration/mask files.
        """

        if mtd.doesExist("sample"):
            mtd["sample"].run()["limits"] = np.array(limits).flatten().tolist()

            values = []
            for key in motors.keys():
                values.append(motors[key])
            if len(values) > 0:
                mtd["sample"].run()["motors"] = values

            mtd["sample"].run()["cal"] = cal
            mtd["sample"].run()["mask"] = mask

    def load_UB(self, filename):
        """
        Load a UB matrix from file and apply to the coverage workspace.

        Parameters
        ----------
        filename : str
            Path to UB file.
        """

        LoadIsawUB(InputWorkspace="coverage", Filename=filename)

        self.copy_UB()

    def get_UB(self):
        """
        Get the UB matrix from the coverage workspace.

        Returns
        -------
        np.ndarray or None
            UB matrix if present, else None.
        """

        if self.has_UB():
            return mtd["coverage"].sample().getOrientedLattice().getUB().copy()

    def copy_UB(self):
        """
        Copy the UB matrix from coverage to other relevant workspaces.
        """

        UB = self.get_UB()
        if UB is not None:
            self.set_UB(UB)

    def has_UB(self):
        """
        Check if the coverage workspace has a UB matrix.

        Returns
        -------
        bool
            True if UB is present, False otherwise.
        """

        if HasUB(Workspace="coverage"):
            return True
        else:
            return False

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

    def get_modes(self, instrument):
        """
        Get available goniometer modes for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        list
            List of goniometer modes.
        """

        return list(beamlines[instrument]["Goniometer"].keys())

    def get_counting_options(self, instrument):
        """
        Get available counting options for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        list
            List of counting options.
        """

        return beamlines[instrument]["Counting"]

    def get_scan_log(self, instrument):
        """
        Get scan log title for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        str
            Scan log title.
        """

        return beamlines[instrument]["Title"]

    def get_axes_polarities(self, instrument, mode):
        """
        Get goniometer axes and polarities for a given mode.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer mode.

        Returns
        -------
        tuple
            (axes, polarities) for the mode.
        """

        goniometers = beamlines[instrument]["Goniometer"][mode]

        axes = [goniometers[name][:-3] for name in goniometers.keys()]

        polarities = [goniometers[name][3] for name in goniometers.keys()]

        return axes, polarities

    def get_goniometer_axes(self, instrument, mode):
        """
        Get formatted goniometer axes for a given mode.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer mode.

        Returns
        -------
        list
            List of formatted axes strings.
        """

        goniometers = beamlines[instrument]["Goniometer"][mode]

        axes = [
            "{}," + ",".join(np.array(goniometers[name][:-2]).astype(str))
            for name in goniometers.keys()
        ]

        return axes

    def get_goniometers(self, instrument, mode):
        """
        Get goniometer names and settings for a given mode.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        mode : str
            Goniometer mode.

        Returns
        -------
        list
            List of goniometer name and settings tuples.
        """

        goniometers = beamlines[instrument]["Goniometer"][mode]

        return [(name, *goniometers[name][-2:]) for name in goniometers.keys()]

    def get_motors(self, instrument):
        """
        Get available motors for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        list
            List of motor name and setting tuples.
        """

        motors = beamlines[instrument].get("Motor")

        if motors is not None:
            return [(name, motors[name]) for name in motors.keys()]
        else:
            return []

    def get_wavelength(self, instrument):
        """
        Get wavelength range for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument identifier.

        Returns
        -------
        tuple
            (min, max) wavelength.
        """

        return beamlines[instrument]["Wavelength"]

    def save_plan(self, filename):
        """
        Save the experiment plan table to a CSV file.

        Parameters
        ----------
        filename : str
            Path to output CSV file.
        """

        plan_dict = mtd["plan"].toDict().copy()
        use_angle = plan_dict["Use"]

        for key in plan_dict.keys():
            items = plan_dict[key]
            items = [item for item, use in zip(items, use_angle) if use]
            plan_dict[key] = items

        plan_dict.pop("Use")

        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=plan_dict.keys())
            writer.writeheader()
            for row in zip(*plan_dict.values()):
                writer.writerow(dict(zip(plan_dict.keys(), row)))

    def save_experiment(self, filename):
        """
        Save the experiment plan and sample to a Nexus file.

        Parameters
        ----------
        filename : str
            Path to output Nexus file.
        """

        if mtd.doesExist("plan"):
            SaveNexus(InputWorkspace="plan", Filename=filename)
            if mtd.doesExist("sample"):
                SaveNexus(
                    InputWorkspace="sample", Filename=filename, Append=True
                )

    def load_experiment(self, filename):
        """
        Load an experiment plan and sample from a Nexus file.

        Parameters
        ----------
        filename : str
            Path to input Nexus file.

        Returns
        -------
        tuple
            (plan, config, symm) loaded from file.
        """

        LoadNexus(Filename=filename, OutputWorkspace="experiment")

        plan, sample = mtd["experiment"].getNames()

        UB = mtd[sample].sample().getOrientedLattice().getUB().copy()
        SetUB(Workspace="coverage", UB=UB)

        self.set_UB(UB)

        instrument = mtd[sample].run().getProperty("instrument").value
        mode = mtd[sample].run().getProperty("mode").value
        wl_min = mtd[sample].run().getProperty("lamda_min").value
        wl_max = mtd[sample].run().getProperty("lamda_max").value
        d_min = mtd[sample].run().getProperty("d_min").value
        cs = mtd[sample].run().getProperty("crystal_system").value
        pg = mtd[sample].run().getProperty("point_group").value
        lc = mtd[sample].run().getProperty("lattice_centering").value
        lims = mtd[sample].run().getProperty("limits").value
        mask = mtd[sample].run().getProperty("mask").value
        cal = mtd[sample].run().getProperty("cal").value
        lims = np.array(lims).reshape(-1, 2).tolist()
        vals = []
        if mtd[sample].run().hasProperty("motors"):
            vals = mtd[sample].run().getProperty("motors").value

        if np.isclose(wl_min, wl_max):
            wl = wl_min
        else:
            wl = [wl_min, wl_max]

        cols = mtd[plan].columnCount() - 5
        rows = mtd[plan].rowCount()

        titles = mtd[plan].column(0)
        comments = mtd[plan].column(cols + 1)
        counts = mtd[plan].column(cols + 2)
        values = mtd[plan].column(cols + 3)
        use = mtd[plan].column(cols + 4)

        settings = []
        for row in range(rows):
            angles = []
            for col in range(cols):
                angle = mtd[plan].cell(row, col + 1)
                angles.append(angle)
            settings.append(angles)

        plan = (titles, settings, comments, counts, values, use)
        config = (instrument, mode, wl, d_min, lims, vals, cal, mask)
        symm = (cs, pg, lc)

        return plan, config, symm

    def generate_axes(self, axes, polarities):
        """
        Generate formatted axes strings for goniometer settings.

        Parameters
        ----------
        axes : list
            List of axis settings.
        polarities : list
            List of axis polarities.
        """

        self.axes = [None] * 6

        for i, (axis, polarity) in enumerate(zip(axes, polarities)):
            self.axes[i] = "{},"
            self.axes[i] += ",".join(np.array([*axis, polarity]).astype(str))

    def get_setting(self, free_angles, limits):
        """
        Get a full setting vector from free angles and limits.

        Parameters
        ----------
        free_angles : list
            List of free angle values.
        limits : list
            List of (min, max) angle limits.

        Returns
        -------
        list
            Full setting vector.
        """

        setting = []
        col = 0
        for limit in limits:
            if np.isclose(limit[0], limit[1]):
                setting.append(limit[0])
            else:
                setting.append(free_angles[col])
                col += 1
        return setting

    def _calculate_matrices(self, axes, polarities, limits, step):
        """
        Calculate rotation matrices for all angle settings.

        Parameters
        ----------
        axes : list
            List of axis settings.
        polarities : list
            List of axis polarities.
        limits : list
            List of (min, max) angle limits.
        step : float
            Step size for angle scanning.

        Returns
        -------
        tuple
            (Rs, angles) where Rs is a list of rotation matrices and angles is an array of angle settings.
        """

        self.generate_axes(axes, polarities)

        free = 0
        for limit in limits:
            free += 1 - np.isclose(limit[0], limit[1])
        step *= free

        angular_coverage = []
        for limit in limits:
            angular_coverage.append(np.arange(limit[0], limit[1] + step, step))

        axes = np.array(axes)
        polarities = np.array(polarities)

        angle_settings = np.meshgrid(*angular_coverage, indexing="ij")
        angle_settings = np.reshape(angle_settings, (len(polarities), -1)).T

        angles = angle_settings.copy()

        angle_settings = angle_settings * polarities
        angle_settings = np.deg2rad(angle_settings)

        rotation_vectors = angle_settings[..., None] * axes
        rotation_vectors = rotation_vectors.reshape(-1, 3)

        all_rotations = Rotation.from_rotvec(rotation_vectors).as_matrix()
        all_rotations = all_rotations.reshape(*angle_settings.shape, 3, 3)

        Rs = []
        for i in range(all_rotations.shape[0]):
            R = np.eye(3)
            for j in range(all_rotations.shape[1]):
                R = R @ all_rotations[i, j, :, :]
            Rs.append(R)

        return Rs, angles

    def individual_peak(
        self, hkl, wavelength, axes, polarities, limits, equiv, pg, step=1
    ):
        """
        Calculate settings and angles for a single peak (and equivalents).

        Parameters
        ----------
        hkl : tuple
            Miller indices (h, k, l) of the reflection.
        wavelength : tuple
            (min, max) wavelength.
        axes : list
            List of axis settings.
        polarities : list
            List of axis polarities.
        limits : list
            List of (min, max) angle limits.
        equiv : bool
            Whether to calculate for equivalent peaks.
        pg : str
            Point group.
        step : float, optional
            Step size for angle scanning (default is 1).

        Returns
        -------
        tuple
            (gamma, nu, lamda) angle values.
        """

        pg = PointGroupFactory.createPointGroup(pg)

        hkls = pg.getEquivalents(hkl) if equiv else [hkl]

        angles, angles_gamma, angles_nu, angles_lamda = [], [], [], []

        for hkl in hkls:
            settings, values = self.calculate_individual_peak(
                hkl, wavelength, axes, polarities, limits, step
            )

            gamma, nu, lamda = values

            angles.append(settings)
            angles_gamma.append(gamma)
            angles_nu.append(nu)
            angles_lamda.append(lamda)

        angles = np.row_stack(angles)
        gamma = np.concatenate(angles_gamma)
        nu = np.concatenate(angles_nu)
        lamda = np.concatenate(angles_lamda)

        self.angles = angles
        self.angles_gamma = gamma
        self.angles_nu = nu
        self.angles_lamda = lamda

        self.angles_gamma_alt = None
        self.angles_nu_alt = None
        self.angles_lamda_alt = None

        return gamma, nu, lamda

    def calculate_individual_peak(
        self, hkl, wavelength, axes, polarities, limits, step=1
    ):
        """
        Calculate settings and angles for a single hkl reflection.

        Parameters
        ----------
        hkl : tuple
            Miller indices (h, k, l) of the reflection.
        wavelength : tuple
            (min, max) wavelength.
        axes : list
            List of axis settings.
        polarities : list
            List of axis polarities.
        limits : list
            List of (min, max) angle limits.
        step : float, optional
            Step size for angle scanning (default is 1).

        Returns
        -------
        tuple
            (settings, (gamma, nu, lamda)) where settings are the angle settings and (gamma, nu, lamda) are the angle values.
        """

        self.comment = "(" + " ".join(np.array(hkl).astype(str)) + ")"

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        FilterPeaks(
            InputWorkspace="peak",
            OutputWorkspace="peak",
            FilterVariable="RunNumber",
            FilterValue=-1,
            Operator="=",
        )

        UB = mtd["coverage"].sample().getOrientedLattice().getUB().copy()

        SetUB(Workspace="peak", UB=UB)

        AddPeakHKL(Workspace="peak", HKL=hkl)

        Q_sample = mtd["peak"].getPeak(0).getQSampleFrame()

        Q = np.sqrt(np.dot(Q_sample, Q_sample))

        Rs, angles = self._calculate_matrices(axes, polarities, limits, step)

        mtd["peak"].run().getGoniometer().setR(np.eye(3))
        mtd["peaks"].run().getGoniometer().setR(np.eye(3))

        Q_lab = np.einsum("kij,j->ki", Rs, Q_sample)

        lamda = -4 * np.pi * Q_lab[:, 2] / Q**2
        mask = (lamda > wavelength[0]) & (lamda < wavelength[1])

        k = 2 * np.pi / lamda

        ki = k[:, np.newaxis] * np.array([0, 0, 1])
        kf = Q_lab + ki

        gamma = np.rad2deg(np.arctan2(kf[:, 0], kf[:, 2]))[mask]
        nu = np.rad2deg(np.arcsin(kf[:, 1] / k))[mask]
        lamda = lamda[mask]

        settings = angles[mask]

        if len(lamda) > 0:
            k = 2 * np.pi / lamda
            Qx = k * np.cos(np.deg2rad(nu)) * np.sin(np.deg2rad(gamma))
            Qy = k * np.sin(np.deg2rad(nu))
            Qz = k * (np.cos(np.deg2rad(nu)) * np.cos(np.deg2rad(gamma)) - 1)

            mask = []
            for i in range(len(k)):
                peak = mtd["peaks"].createPeak(V3D(Qx[i], Qy[i], Qz[i]))
                mask.append(peak.getDetectorID() in self.det_ID)

            mask = np.array(mask)

            gamma = gamma[mask]
            nu = nu[mask]
            lamda = lamda[mask]

            settings = settings[mask]

        return settings, (gamma, nu, lamda)

    def simultaneous_peaks(
        self,
        hkl_1,
        hkl_2,
        wavelength,
        axes,
        polarities,
        limits,
        equiv,
        pg,
        step=1,
    ):
        """
        Calculate settings and angles for two simultaneous peaks (and equivalents).

        Parameters
        ----------
        hkl_1 : tuple
            Miller indices (h, k, l) of the first reflection.
        hkl_2 : tuple
            Miller indices (h, k, l) of the second reflection.
        wavelength : tuple
            (min, max) wavelength.
        axes : list
            List of axis settings.
        polarities : list
            List of axis polarities.
        limits : list
            List of (min, max) angle limits.
        equiv : bool
            Whether to calculate for equivalent peaks.
        pg : str
            Point group.
        step : float, optional
            Step size for angle scanning (default is 1).

        Returns
        -------
        tuple
            ((gamma, nu, lamda), (gamma_alt, nu_alt, lamda_alt)) where the first tuple is for the primary peaks and the second is for the alternative peaks.
        """

        pg = PointGroupFactory.createPointGroup(pg)

        hkls_1 = pg.getEquivalents(hkl_1) if equiv else [hkl_1]
        hkls_2 = pg.getEquivalents(hkl_2) if equiv else [hkl_2]

        pairs = list(itertools.product(hkls_1, hkls_2))

        angles, angles_gamma, angles_nu, angles_lamda = [], [], [], []

        angles_gamma_alt, angles_nu_alt, angles_lamda_alt = [], [], []

        for hkl_1, hkl_2 in pairs:
            settings, values0, values1 = self.simultaneous_peaks_hkl(
                hkl_1, hkl_2, wavelength, axes, polarities, limits, step
            )

            gamma0, nu0, lamda0 = values0
            gamma1, nu1, lamda1 = values1

            angles.append(settings)
            angles_gamma.append(gamma0)
            angles_nu.append(nu0)
            angles_lamda.append(lamda0)

            angles_gamma_alt.append(gamma1)
            angles_nu_alt.append(nu1)
            angles_lamda_alt.append(lamda1)

        angles = np.row_stack(angles)
        gamma = np.concatenate(angles_gamma)
        nu = np.concatenate(angles_nu)
        lamda = np.concatenate(angles_lamda)

        gamma_alt = np.concatenate(angles_gamma_alt)
        nu_alt = np.concatenate(angles_nu_alt)
        lamda_alt = np.concatenate(angles_lamda_alt)

        self.angles = angles
        self.angles_gamma = gamma
        self.angles_nu = nu
        self.angles_lamda = lamda

        self.angles_gamma_alt = gamma_alt
        self.angles_nu_alt = nu_alt
        self.angles_lamda_alt = lamda_alt

        return (gamma, nu, lamda), (gamma_alt, nu_alt, lamda_alt)

    def simultaneous_peaks_hkl(
        self, hkl_1, hkl_2, wavelength, axes, polarities, limits, step=1
    ):
        """
        Calculate settings and angles for a specific pair of hkl reflections.

        Parameters
        ----------
        hkl_1 : tuple
            Miller indices (h, k, l) of the first reflection.
        hkl_2 : tuple
            Miller indices (h, k, l) of the second reflection.
        wavelength : tuple
            (min, max) wavelength.
        axes : list
            List of axis settings.
        polarities : list
            List of axis polarities.
        limits : list
            List of (min, max) angle limits.
        step : float, optional
            Step size for angle scanning (default is 1).

        Returns
        -------
        tuple
            (settings, (gamma0, nu0, lamda0), (gamma1, nu1, lamda1)) where settings are the angle settings and (gamma, nu, lamda) are the angle values for each reflection.
        """

        self.comment = "(" + " ".join(np.array(hkl_1).astype(str)) + ")"

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        FilterPeaks(
            InputWorkspace="peak",
            OutputWorkspace="peak",
            FilterVariable="RunNumber",
            FilterValue=-1,
            Operator="=",
        )

        UB = mtd["coverage"].sample().getOrientedLattice().getUB().copy()

        SetUB(Workspace="peak", UB=UB)

        AddPeakHKL(Workspace="peak", HKL=hkl_1)
        AddPeakHKL(Workspace="peak", HKL=hkl_2)

        Q0_sample = mtd["peak"].getPeak(0).getQSampleFrame()
        Q1_sample = mtd["peak"].getPeak(1).getQSampleFrame()

        Q0 = np.sqrt(np.dot(Q0_sample, Q0_sample))
        Q1 = np.sqrt(np.dot(Q1_sample, Q1_sample))

        Rs, angles = self._calculate_matrices(axes, polarities, limits, step)

        Q0_lab = np.einsum("kij,j->ki", Rs, Q0_sample)
        Q1_lab = np.einsum("kij,j->ki", Rs, Q1_sample)

        lamda0 = -4 * np.pi * Q0_lab[:, 2] / Q0**2
        lamda1 = -4 * np.pi * Q1_lab[:, 2] / Q1**2

        mask = (
            (lamda0 > wavelength[0])
            & (lamda0 < wavelength[1])
            & (lamda1 > wavelength[0])
            & (lamda1 < wavelength[1])
        )

        k0 = 2 * np.pi / lamda0
        k1 = 2 * np.pi / lamda1

        k0i = k0[:, np.newaxis] * np.array([0, 0, 1])
        k1i = k1[:, np.newaxis] * np.array([0, 0, 1])

        k0f = Q0_lab + k0i
        k1f = Q1_lab + k1i

        gamma0 = np.rad2deg(np.arctan2(k0f[:, 0], k0f[:, 2]))[mask]
        gamma1 = np.rad2deg(np.arctan2(k1f[:, 0], k1f[:, 2]))[mask]

        nu0 = np.rad2deg(np.arcsin(k0f[:, 1] / k0))[mask]
        nu1 = np.rad2deg(np.arcsin(k1f[:, 1] / k1))[mask]

        lamda0 = lamda0[mask]
        lamda1 = lamda1[mask]

        angles = angles[mask]

        if len(lamda0) > 0:
            k0 = 2 * np.pi / lamda0
            k1 = 2 * np.pi / lamda1

            Q0x = k0 * np.cos(np.deg2rad(nu0)) * np.sin(np.deg2rad(gamma0))
            Q1x = k1 * np.cos(np.deg2rad(nu1)) * np.sin(np.deg2rad(gamma1))

            Q0y = k0 * np.sin(np.deg2rad(nu0))
            Q1y = k1 * np.sin(np.deg2rad(nu1))

            Q0z = k0 * (
                np.cos(np.deg2rad(nu0)) * np.cos(np.deg2rad(gamma0)) - 1
            )
            Q1z = k1 * (
                np.cos(np.deg2rad(nu1)) * np.cos(np.deg2rad(gamma1)) - 1
            )

            mask = []
            for i in range(len(k1)):
                peak0 = mtd["peaks"].createPeak(V3D(Q0x[i], Q0y[i], Q0z[i]))
                peak1 = mtd["peaks"].createPeak(V3D(Q1x[i], Q1y[i], Q1z[i]))
                mask.append(
                    (peak0.getDetectorID() in self.det_ID)
                    & (peak1.getDetectorID() in self.det_ID)
                )

            mask = np.array(mask)

            gamma0 = gamma0[mask]
            gamma1 = gamma1[mask]

            nu0 = nu0[mask]
            nu1 = nu1[mask]

            lamda0 = lamda0[mask]
            lamda1 = lamda1[mask]

            angles = angles[mask]

        return angles, (gamma0, nu0, lamda0), (gamma1, nu1, lamda1)

    def get_angles(self, gamma, nu):
        """
        Find the closest calculated angles to a given gamma and nu.

        Parameters
        ----------
        gamma : float
            Gamma angle.
        nu : float
            Nu angle.

        Returns
        -------
        tuple
            (angles, gamma, nu, lamda, gamma_alt, nu_alt, lamda_alt) where angles are the setting angles and the rest are the angle values.
        """

        if len(self.angles_gamma) > 0:
            d2 = (self.angles_gamma - gamma) ** 2 + (self.angles_nu - nu) ** 2

            i = np.argmin(d2)

            angles = self.angles[i]

            gamma = self.angles_gamma[i]
            nu = self.angles_nu[i]
            lamda = self.angles_lamda[i]

            gamma_alt = nu_alt = lamda_alt = None

            if self.angles_lamda_alt is not None:
                gamma_alt = self.angles_gamma_alt[i]
                nu_alt = self.angles_nu_alt[i]
                lamda_alt = self.angles_lamda_alt[i]

            return angles, gamma, nu, lamda, gamma_alt, nu_alt, lamda_alt

    def add_mesh(
        self, mesh_angles, wavelength, d_min, rows, free_angles, all_angles
    ):
        """
        Add a mesh of orientations to the experiment plan.

        Parameters
        ----------
        mesh_angles : tuple
            (limits, numbers) for the mesh grid.
        wavelength : tuple
            (min, max) wavelength.
        d_min : float
            Minimum d-spacing.
        rows : int
            Starting row index for the mesh.
        free_angles : list
            List of free angle settings.
        all_angles : list
            List of all angle settings.

        Returns
        -------
        list
            List of angle values at the mesh points.
        """

        limits, ns = mesh_angles

        mins, maxs = zip(*limits)

        axes = [
            np.linspace(lo, hi, n + 1)[:-1]
            for lo, hi, n in zip(mins, maxs, ns)
        ]

        grids = np.meshgrid(*axes, indexing="ij")
        points = np.stack(grids, axis=-1).reshape(-1, len(limits))

        indices = [all_angles.index(free) for free in free_angles]

        values = []
        for i, angles in enumerate(points):
            self.add_orientation(angles, wavelength, d_min, rows + i)
            values.append(angles[indices])

        return values

    def add_orientation(self, angles, wavelength, d_min, rows):
        """
        Add a single orientation to the experiment plan.

        Parameters
        ----------
        angles : list
            List of angle values.
        wavelength : tuple
            (min, max) wavelength.
        d_min : float
            Minimum d-spacing.
        rows : int
            Row index for the orientation.
        """

        if np.isclose(wavelength[0], wavelength[1]):
            wavelength = [0.975 * wavelength[0], 1.025 * wavelength[1]]

        axes = np.array(self.axes).copy().tolist()
        print(axes, angles)

        for i, angle in enumerate(angles):
            axes[i] = axes[i].format(angle)

        ol = mtd["coverage"].sample().getOrientedLattice()
        UB = ol.getUB().copy()

        SetUB(Workspace="instrument", UB=UB)

        SetGoniometer(
            Workspace="instrument",
            Axis0=axes[0],
            Axis1=axes[1],
            Axis2=axes[2],
            Axis3=axes[3],
            Axis4=axes[4],
            Axis5=axes[5],
        )

        d_max = 1.1 * np.max([ol.d(1, 0, 0), ol.d(0, 1, 0), ol.d(0, 0, 1)])

        ws = "peaks_orientation_{}".format(rows)

        PredictPeaks(
            InputWorkspace="instrument",
            MinDSpacing=d_min,
            MaxDSpacing=d_max,
            WavelengthMin=wavelength[0],
            WavelengthMax=wavelength[1],
            ReflectionCondition="Primitive",
            OutputWorkspace=ws,
        )

        SortPeaksWorkspace(
            InputWorkspace=ws,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=ws,
        )

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace=ws,
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace=ws,
            )

        for no in range(mtd[ws].getNumberPeaks() - 1, 0, -1):
            if (
                mtd[ws].getPeak(no).getHKL() - mtd[ws].getPeak(no - 1).getHKL()
            ).norm2() == 0:
                DeleteTableRows(TableWorkspace=ws, Rows=no)

        for no in range(mtd[ws].getNumberPeaks() - 1, 0, -1):
            if mtd[ws].getPeak(no).getDetectorID() not in self.det_ID:
                DeleteTableRows(TableWorkspace=ws, Rows=no)

        SortPeaksWorkspace(
            InputWorkspace=ws,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace=ws,
        )

        for peak in mtd[ws]:
            peak.setRunNumber(rows)

        mtd["instrument"].run().getGoniometer().setR(np.eye(3))

        SetUB(Workspace="combined", UB=UB)
        CombinePeaksWorkspaces(
            LHSWorkspace="combined",
            RHSWorkspace=ws,
            OutputWorkspace="combined",
        )

    def generate_table(self, row):
        """
        Generate a table of peaks for a given row in the plan.

        Parameters
        ----------
        row : int
            Row index in the plan.

        Returns
        -------
        list
            List of [h, k, l, d, lamda] for the peaks in the row.
        """

        if row == -1 and mtd.doesExist("missing"):
            CloneWorkspace(InputWorkspace="missing", OutputWorkspace="table")
        else:
            FilterPeaks(
                InputWorkspace="combined",
                FilterVariable="RunNumber",
                FilterValue=str(row),
                Operator="=",
                OutputWorkspace="table",
            )

        SortPeaksWorkspace(
            InputWorkspace="table",
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace="table",
        )

        columns = ["l", "k", "h"]

        for col in columns:
            SortPeaksWorkspace(
                InputWorkspace="table",
                ColumnNameToSortBy=col,
                SortAscending=False,
                OutputWorkspace="table",
            )

        SortPeaksWorkspace(
            InputWorkspace="table",
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
            OutputWorkspace="table",
        )

        h = mtd["table"].column("h")
        k = mtd["table"].column("k")
        l = mtd["table"].column("l")

        d = mtd["table"].column("DSpacing")
        lamda = mtd["table"].column("Wavelength")

        return np.array([h, k, l, d, lamda]).T.tolist()

    def calculate_statistics(self, point_group, lattice_centering, use, d_min):
        """
        Calculate completeness and redundancy statistics for the plan.

        Parameters
        ----------
        point_group : str
            Point group name.
        lattice_centering : str
            Lattice centering type.
        use : list
            List of booleans indicating which rows to use.
        d_min : float
            Minimum d-spacing.

        Returns
        -------
   
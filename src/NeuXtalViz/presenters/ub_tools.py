import numpy as np

from NeuXtalViz.presenters.base_presenter import NeuXtalVizPresenter


class UB(NeuXtalVizPresenter):
    def __init__(self, view, model):
        super(UB, self).__init__(view, model)

        self.view.connect_load_Q(self.load_Q)
        self.view.connect_save_Q(self.save_Q)
        self.view.connect_load_peaks(self.load_peaks)
        self.view.connect_save_peaks(self.save_peaks)
        self.view.connect_load_UB(self.load_UB)
        self.view.connect_save_UB(self.save_UB)
        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_wavelength(self.update_wavelength)

        self.view.connect_browse_calibration(self.load_detector_calibration)
        self.view.connect_browse_tube(self.load_tube_calibration)

        self.view.connect_convert_Q(self.convert_Q)
        self.view.connect_find_peaks(self.find_peaks)
        self.view.connect_find_spacing(self.update_find_spacing)
        self.view.connect_find_distance(self.update_find_distance)
        self.view.connect_index_peaks(self.index_peaks)
        self.view.connect_predict_peaks(self.predict_peaks)
        self.view.connect_integrate_peaks(self.integrate_peaks)
        self.view.connect_filter_peaks(self.filter_peaks)
        self.view.connect_find_conventional(self.find_conventional)
        self.view.connect_lattice_transform(self.lattice_transform)
        self.view.connect_symmetry_transform(self.symmetry_transform)
        self.view.connect_transform_UB(self.transform_UB)
        self.view.connect_optimize_UB(self.refine_UB)
        self.view.connect_find_niggli(self.find_niggli)
        self.view.connect_calculate_peaks(self.calculate_peaks)
        self.view.connect_cell_row_highligter(self.highlight_cell)
        self.view.connect_peak_row_highligter(self.highlight_peak)
        self.view.connect_select_cell(self.select_cell)

        self.switch_instrument()
        self.lattice_transform()

        self.view.connect_convert_to_hkl(self.convert_to_hkl)

        self.view.connect_data_combo(self.update_instrument_view)
        self.view.connect_diffraction(self.update_roi)
        self.view.connect_d_min(self.update_instrument_view)
        self.view.connect_d_max(self.update_instrument_view)
        self.view.connect_horizontal(self.update_roi)
        self.view.connect_vertical(self.update_roi)
        self.view.connect_horizontal_roi(self.update_roi)
        self.view.connect_vertical_roi(self.update_roi)

        self.view.connect_add_peak(self.add_peak)
        self.view.connect_check_hkl(self.calculate_hkl)

        self.view.connect_roi_ready(self.update_scan)
        self.view.connect_scan_ready(self.update_check_hkl)

        self.view.connect_h_index(self.hand_index_fractional)
        self.view.connect_k_index(self.hand_index_fractional)
        self.view.connect_l_index(self.hand_index_fractional)

        self.view.connect_integer_h_index(self.hand_index_integer)
        self.view.connect_integer_k_index(self.hand_index_integer)
        self.view.connect_integer_l_index(self.hand_index_integer)

        self.view.connect_integer_m_index(self.hand_index_integer)
        self.view.connect_integer_n_index(self.hand_index_integer)
        self.view.connect_integer_p_index(self.hand_index_integer)

        self.view.connect_min_slider(self.view.update_colorbar_min)
        self.view.connect_max_slider(self.view.update_colorbar_max)

        self.view.connect_slice_combo(self.reslice)
        self.view.connect_slice_thickness_line(self.reslice)
        self.view.connect_slice_width_line(self.reslice)

        self.view.connect_clim_combo(self.reslice)
        self.view.connect_cbar_combo(self.reslice)

        self.view.connect_slice_scale_combo(self.reslice)
        self.view.connect_slice_line(self.reslice)

        self.slice_idle = True
        self.volume_idle = True

        self.view.connect_cluster(self.cluster)

    def update_find_spacing(self):
        d = self.view.get_find_peaks_spacing()
        Q = 2 * np.pi / d
        self.view.set_find_peaks_distance(Q)

    def update_find_distance(self):
        Q = self.view.get_find_peaks_distance()
        d = 2 * np.pi / Q
        self.view.set_find_peaks_spacing(d)

    def hand_index_fractional(self):
        mod_info = self.get_modulation_info()
        hkl_info = self.view.get_indices()
        index_row = self.view.get_peak()

        if (
            mod_info is not None
            and hkl_info is not None
            and index_row is not None
        ):
            mod_vec_1, mod_vec_2, mod_vec_3, *_ = mod_info
            hkl, int_hkl, int_mnp = hkl_info

            int_hkl, int_mnp = self.model.calculate_integer(
                mod_vec_1, mod_vec_2, mod_vec_3, hkl
            )

            self.model.set_peak(index_row, hkl, int_hkl, int_mnp)

            self.view.update_table_index(index_row, hkl)

            self.view.set_indices(hkl, int_hkl, int_mnp)

    def hand_index_integer(self):
        mod_info = self.get_modulation_info()
        hkl_info = self.view.get_indices()
        index_row = self.view.get_peak()

        if (
            mod_info is not None
            and hkl_info is not None
            and index_row is not None
        ):
            mod_vec_1, mod_vec_2, mod_vec_3, *_ = mod_info
            hkl, int_hkl, int_mnp = hkl_info

            hkl = self.model.calculate_fractional(
                mod_vec_1, mod_vec_2, mod_vec_3, int_hkl, int_mnp
            )

            self.model.set_peak(index_row, hkl, int_hkl, int_mnp)

            self.view.update_table_index(index_row, hkl)

            self.view.set_indices(hkl, int_hkl, int_mnp)

    def convert_Q(self):
        worker = self.view.worker(self.convert_Q_process)
        worker.connect_result(self.convert_Q_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def convert_Q_complete(self, result):
        if result is not None:
            self.view.update_diffraction_label(result)

            self.update_instrument_view()

    def convert_Q_process(self, progress):
        instrument = self.view.get_instrument()
        wavelength = self.view.get_wavelength()
        tube_cal = self.view.get_tube_calibration()
        det_cal = self.view.get_detector_calibration()

        IPTS = self.view.get_IPTS()
        runs = self.view.get_runs()
        exp = self.view.get_experiment()
        lorentz = self.view.get_lorentz()
        time_stop = self.view.get_time_stop()
        d_min = self.view.get_convert_min_d()

        validate = [IPTS, runs, wavelength]

        if instrument == "DEMAND":
            validate.append(exp)

        if all(elem is not None for elem in validate):
            mono = np.isclose(wavelength[0], wavelength[1])

            progress("Processing...", 1)

            progress("Data loading...", 10)

            data_load = self.model.load_data(
                instrument,
                IPTS,
                runs,
                exp,
                time_stop,
            )

            if data_load is None:
                progress("Files do not exist.", 0)

            self.view.set_data_list(self.model.get_number_workspaces())

            progress("Data loaded...", 40)

            progress("Data calibrating...", 50)

            self.model.calibrate_data(instrument, det_cal, tube_cal)

            progress("Data calibrated...", 60)

            progress("Data converting...", 70)

            self.model.convert_data(instrument, wavelength, lorentz, d_min)

            progress("Data converted...", 99)

            progress("Data converted!", 0)

            return mono

        else:
            progress("Invalid parameters.", 0)

    def add_peak(self):
        if self.model.has_Q():
            ind = self.view.get_data_list()
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            val = self.view.get_diffraction()

            validate = [horz, vert, val]

            if all(elem is not None for elem in validate):
                self.model.add_peak(ind, val, horz, vert)
                self.visualize()

    def calculate_hkl(self):
        if self.model.has_Q():
            ind = self.view.get_data_list()
            hkl = self.view.get_check_hkl()

            validate = [ind, hkl]

            if all(elem is not None for elem in validate):
                vals = self.model.calculate_hkl_position(ind, *hkl)

                if vals is not None:
                    x, horz, vert = vals
                    self.view.set_diffraction(x)
                    self.view.set_horizontal(horz)
                    self.view.set_vertical(vert)
                    self.update_instrument_view()

    def update_instrument_view(self):
        worker = self.view.worker(self.update_instrument_view_process)
        worker.connect_result(self.update_instrument_view_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def update_instrument_view_complete(self, result):
        if result is not None:
            self.view.update_instrument_view(result[0])
            self.view.update_roi_view(result[1])
            self.view.update_scan_view(result[1])

            self.update_check_hkl()

    def update_instrument_view_process(self, progress):
        if self.model.has_Q():
            ind = self.view.get_data_list()
            d_min = self.view.get_d_min()
            d_max = self.view.get_d_max()
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            horz_roi = self.view.get_horizontal_roi()
            vert_roi = self.view.get_vertical_roi()
            val = self.view.get_diffraction()

            validate = [d_min, d_max, horz, vert, horz_roi, vert_roi, val]

            if all(elem is not None for elem in validate):
                progress("Processing...", 1)

                progress("Detector viewing...", 10)

                self.model.calculate_instrument_view(ind, d_min, d_max)

                progress("Detector viewed...", 50)

                self.model.extract_roi(horz, vert, horz_roi, vert_roi, val)

                progress("ROI viewed...", 70)

                progress("Data/ROI viewed!", 0)

                return self.model.inst_view, self.model.roi_view

        else:
            progress("Invalid parameters.", 0)

    def update_roi(self):
        if self.model.has_Q():
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            horz_roi = self.view.get_horizontal_roi()
            vert_roi = self.view.get_vertical_roi()
            val = self.view.get_diffraction()

            validate = [horz, vert, horz_roi, vert_roi, val]

            if all(elem is not None for elem in validate):
                self.model.extract_roi(horz, vert, horz_roi, vert_roi, val)

                self.view.update_roi_view(self.model.roi_view)

                self.update_check_hkl()

    def update_scan(self):
        if self.model.has_Q():
            horz = self.view.get_horizontal()
            vert = self.view.get_vertical()
            horz_roi = self.view.get_horizontal_roi()
            vert_roi = self.view.get_vertical_roi()
            val = self.view.get_diffraction()

            validate = [horz, vert, horz_roi, vert_roi, val]

            if all(elem is not None for elem in validate):
                self.model.extract_roi(horz, vert, horz_roi, vert_roi, val)

                self.view.update_scan_view(self.model.roi_view)

                self.update_check_hkl()

    def update_check_hkl(self):
        ind = self.view.get_data_list()
        horz = self.view.get_horizontal()
        vert = self.view.get_vertical()
        val = self.view.get_diffraction()

        validate = [horz, vert, val]

        if all(elem is not None for elem in validate):
            ind = self.view.get_data_list()
            hkl = self.model.roi_scan_to_hkl(ind, val, horz, vert)
            if hkl is not None:
                self.view.set_check_hkl(*hkl)

    def visualize(self):
        Q_hist = self.model.get_Q_info()

        if Q_hist is not None and self.volume_idle:
            self.volume_idle = False

            self.update_processing()

            self.update_processing("Updating view...", 50)

            self.view.add_Q_viz(Q_hist)

            if self.model.has_UB():
                self.model.update_UB()

                self.update_oriented_lattice()

                self.view.set_transform(self.model.get_transform())

                self.update_lattice_info()

            if self.model.has_peaks():
                peaks = self.model.get_peak_info()

                self.view.update_peaks_table(peaks)

            self.update_complete("Data visualized!")

            self.volume_idle = True

    def update_lattice_info(self):
        params = self.model.get_lattice_constants()
        errors = self.model.get_lattice_constant_errors()

        if params is not None:
            self.view.set_lattice_constants(params, errors)

        params = self.model.get_sample_directions()

        if params is not None:
            self.view.set_sample_directions(params)

    def find_peaks(self):
        worker = self.view.worker(self.find_peaks_process)
        worker.connect_result(self.find_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def find_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def find_peaks_process(self, progress):
        if self.model.has_Q():
            Q_min = self.view.get_find_peaks_distance()
            d_max = self.view.get_find_peaks_spacing()
            params = self.view.get_find_peaks_parameters()
            edge = self.view.get_find_peaks_edge()
            no_al = self.view.get_avoid_aluminum()

            if Q_min is not None and params is not None:
                progress("Processing...", 1)

                progress("Finding peaks...", 10)

                self.model.find_peaks(Q_min, *params, edge)
                d_min = self.model.get_d_min()

                if no_al and d_min < d_max:
                    self.model.avoid_aluminum_contamination(d_min, d_max)

                progress("Peaks found...", 90)

                progress("Peaks found!", 100)

            else:
                progress("Invalid parameters.", 0)

    def find_conventional(self):
        worker = self.view.worker(self.find_conventional_process)
        worker.connect_result(self.find_conventional_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def find_conventional_complete(self, result):
        pass

    def find_conventional_process(self, progress):
        if self.model.has_peaks():
            params = self.view.get_lattice_constants()
            tol = self.view.get_calculate_UB_tol()

            if params is not None and tol is not None:
                progress("Processing...", 1)

                progress("Finding UB...", 10)

                self.model.determine_UB_with_lattice_parameters(*params, tol)

                progress("UB found...", 90)

                progress("UB found!", 100)

            else:
                progress("Invalid parameters.", 0)

    def find_niggli(self):
        worker = self.view.worker(self.find_niggli_process)
        worker.connect_result(self.find_niggli_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def find_niggli_complete(self, result):
        self.show_cells()

    def find_niggli_process(self, progress):
        if self.model.has_peaks():
            params = self.view.get_min_max_constants()
            tol = self.view.get_calculate_UB_tol()

            if params is not None and tol is not None:
                progress("Processing...", 1)

                progress("Finding UB...", 10)

                self.model.determine_UB_with_niggli_cell(*params, tol)

                progress("UB found...", 90)

                progress("UB found!", 100)

            else:
                progress("Invalid parameters.", 0)

    def show_cells(self):
        worker = self.view.worker(self.show_cells_process)
        worker.connect_result(self.show_cells_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def show_cells_complete(self, result):
        if result is not None:
            self.view.update_cell_table(result)

    def show_cells_process(self, progress):
        if self.model.has_peaks() and self.model.has_UB():
            scalar = self.view.get_max_scalar_error()

            if scalar is not None:
                progress("Processing...", 1)

                progress("Finding possible cells...", 50)

                cells = self.model.possible_conventional_cells(scalar)

                progress("Possible cells found!", 100)

                return cells

            else:
                progress("Invalid parameters.", 0)

    def select_cell(self):
        worker = self.view.worker(self.select_cell_process)
        worker.connect_result(self.select_cell_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def select_cell_complete(self, result):
        pass

    def select_cell_process(self, progress):
        if self.model.has_peaks() and self.model.has_UB():
            form = self.view.get_form_number()
            tol = self.view.get_calculate_UB_tol()

            if form is not None and tol is not None:
                progress("Processing...", 1)

                progress("Selecting cell...", 50)

                self.model.select_cell(form, tol)

                progress("Cell selected...", 99)

                progress("Cell selected!", 100)

            else:
                progress("Invalid parameters.", 0)

    def highlight_cell(self):
        form = self.view.get_form()
        self.view.set_cell_form(form)

    def highlight_peak(self):
        no = self.view.get_peak()
        if no is not None:
            peak = self.model.get_peak(no)
            if peak is not None:
                self.view.set_peak_info(peak)
                self.view.highlight_peak(no + 1)
                self.view.set_position(peak["Q"])

    def lattice_transform(self):
        cell = self.view.get_lattice_transform()

        Ts = self.model.generate_lattice_transforms(cell)

        self.view.update_symmetry_symbols(list(Ts.keys()))

        self.symmetry_transform()

    def symmetry_transform(self):
        cell = self.view.get_lattice_transform()

        Ts = self.model.generate_lattice_transforms(cell)

        symbol = self.view.get_symmetry_symbol()

        if symbol in Ts.keys():
            T = Ts[symbol]

            self.view.set_transform_matrix(T)

    def transform_UB(self):
        worker = self.view.worker(self.transform_UB_process)
        worker.connect_result(self.transform_UB_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def transform_UB_complete(self, result):
        self.model.copy_UB_from_peaks()

    def transform_UB_process(self, progress):
        if self.model.has_peaks() and self.model.has_UB():
            params = self.view.get_transform_matrix()
            tol = self.view.get_transform_UB_tol()

            if params is not None and tol is not None:
                progress("Processing...", 1)

                progress("Transforming UB...", 50)

                self.model.transform_lattice(params, tol)

                progress("UB transformed...", 99)

                progress("UB transformed!", 100)

            else:
                progress("Invalid parameters.", 0)

    def refine_UB(self):
        worker = self.view.worker(self.refine_UB_process)
        worker.connect_result(self.refine_UB_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def refine_UB_complete(self, result):
        self.model.copy_UB_from_peaks()

    def refine_UB_process(self, progress):
        if self.model.has_peaks():
            params = self.view.get_lattice_constants()
            tol = self.view.get_refine_UB_tol()
            option = self.view.get_refine_UB_option()

            if option == "Constrained" and params is not None:
                progress("Processing...", 1)

                progress("Refining orientation...", 50)

                self.model.refine_U_only(*params)

                progress("Orientation refined...", 99)

                progress("Orientation refined!", 100)

            elif tol is not None:
                progress("Processing...", 1)

                progress("Refining UB...", 50)

                if option == "Unconstrained":
                    self.model.refine_UB_without_constraints(tol)
                else:
                    self.model.refine_UB_with_constraints(option, tol)

                progress("UB refined...", 99)

                progress("UB refined!", 100)

            else:
                progress("Invalid parameters.", 0)

    def get_modulation_info(self):
        mod_info = self.view.get_max_order_cross_terms()
        if mod_info is not None:
            max_order, cross_terms = mod_info
        else:
            max_order, cross_terms = 0, False

        mod_vec = self.view.get_modulatation_offsets()
        if mod_vec is not None:
            mod_vec_1 = mod_vec[0:3]
            mod_vec_2 = mod_vec[3:6]
            mod_vec_3 = mod_vec[6:9]

        return mod_vec_1, mod_vec_2, mod_vec_3, max_order, cross_terms

    def index_peaks(self):
        worker = self.view.worker(self.index_peaks_process)
        worker.connect_result(self.index_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def index_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def index_peaks_process(self, progress):
        mod_info = self.get_modulation_info()

        mod_vec_1, mod_vec_2, mod_vec_3, max_order, cross_terms = mod_info

        if self.model.has_peaks() and self.model.has_UB():
            params = self.view.get_index_peaks_parameters()
            sat = self.view.get_index_satellite_peaks()
            round_hkl = self.view.get_index_peaks_round()

            if params is not None:
                tol, sat_tol = params

                if sat == False:
                    max_order = 0

                progress("Processing...", 1)

                progress("Indexing peaks...", 50)

                self.model.index_peaks(
                    tol,
                    sat_tol,
                    mod_vec_1,
                    mod_vec_2,
                    mod_vec_3,
                    max_order,
                    cross_terms,
                    round_hkl=round_hkl,
                )

                progress("Peaks indexed...", 99)

                progress("Peaks indexed!", 100)

            else:
                progress("Invalid parameters.", 0)

    def predict_peaks(self):
        worker = self.view.worker(self.predict_peaks_process)
        worker.connect_result(self.predict_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def predict_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def predict_peaks_process(self, progress):
        mod_info = self.get_modulation_info()

        mod_vec_1, mod_vec_2, mod_vec_3, max_order, cross_terms = mod_info

        centering = self.view.get_predict_peaks_centering()

        wavelength = self.view.get_wavelength()

        params = self.view.get_predict_peaks_parameters()

        # sat = self.view.get_predict_satellite_peaks()

        edge = self.view.get_predict_peaks_edge()

        if self.model.has_peaks() and self.model.has_UB():
            if wavelength is not None and params is not None:
                d_min, sat_d_min = params

                if sat_d_min < d_min:
                    sat_d_min = d_min

                lamda_min, lamda_max = wavelength

                if np.isclose(lamda_min, lamda_max):
                    lamda_min, lamda_max = 0.97 * lamda_min, 1.03 * lamda_max

                progress("Processing...", 1)

                progress("Predicting peaks...", 50)

                self.model.predict_peaks(
                    centering, d_min, lamda_min, lamda_max, edge
                )

                if self.view.get_predict_satellite_peaks():
                    progress("Predicting modulated...", 75)

                    self.model.predict_modulated_peaks(
                        sat_d_min,
                        lamda_min,
                        lamda_max,
                        mod_vec_1,
                        mod_vec_2,
                        mod_vec_3,
                        max_order,
                        cross_terms,
                    )

                progress("Peaks predicted...", 99)

                progress("Peaks predicted!", 100)

            else:
                progress("Invalid parameters.", 0)

    def integrate_peaks(self):
        worker = self.view.worker(self.integrate_peaks_process)
        worker.connect_result(self.integrate_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def integrate_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def integrate_peaks_process(self, progress):
        params = self.view.get_integrate_peaks_parameters()

        ellipsoid = self.view.get_ellipsoid()

        centroid = self.view.get_centroid()

        if self.model.has_peaks() and self.model.has_Q():
            if params is not None:
                method = "ellipsoid" if ellipsoid else "sphere"

                rad, inner_factor, outer_factor = params

                if inner_factor < 1:
                    inner_factor = 1
                if outer_factor < inner_factor:
                    outer_factor = inner_factor

                progress("Processing...", 1)

                progress("Integrating peaks...", 50)

                self.model.integrate_peaks(
                    rad,
                    inner_factor,
                    outer_factor,
                    method=method,
                    centroid=centroid,
                )

                progress("Peaks integrated...", 99)

                progress("Peaks integrated!", 100)

        else:
            progress("Invalid parameters.", 0)

    def filter_peaks(self):
        worker = self.view.worker(self.filter_peaks_process)
        worker.connect_result(self.filter_peaks_complete)
        worker.connect_finished(self.visualize)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def filter_peaks_complete(self, result):
        self.model.copy_UB_from_peaks()

    def filter_peaks_process(self, progress):
        name = self.view.get_filter_variable()
        operator = self.view.get_filter_comparison()
        value = self.view.get_filter_value()

        if self.model.has_peaks() and value is not None:
            progress("Processing...", 1)

            progress("Filtering peaks...", 50)

            self.model.filter_peaks(name, operator, value)

            progress("Peaks filtered...", 99)

            progress("Peaks filtered!", 100)

        else:
            progress("Invalid parameters.", 0)

    def load_detector_calibration(self):
        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_detector_cal_dialog(path)

        if filename:
            self.view.set_detector_calibration(filename)

    def load_tube_calibration(self):
        inst = self.view.get_instrument()

        path = self.model.get_calibration_file_path(inst)

        filename = self.view.load_tube_cal_dialog(path)

        if filename:
            self.view.set_tube_calibration(filename)

    def load_Q(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_Q_file_dialog(path)

        if filename:
            self.model.load_Q(filename)

    def save_Q(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_Q_file_dialog(path)

        if filename:
            self.model.save_Q(filename)

    def load_peaks(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_peaks_file_dialog(path)

        if filename:
            self.model.load_peaks(filename)

    def save_peaks(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_peaks_file_dialog(path)

        if filename:
            self.model.save_peaks(filename)

    def load_UB(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.load_UB_file_dialog(path)

        if filename:
            self.model.load_UB(filename)

            self.view.set_transform(self.model.get_transform())

    def save_UB(self):
        inst = self.view.get_instrument()
        ipts = self.view.get_IPTS()

        path = self.model.get_shared_file_path(inst, ipts)

        filename = self.view.save_UB_file_dialog(path)

        if filename:
            self.model.save_UB(filename)

    def switch_instrument(self):
        instrument = self.view.get_instrument()

        wavelength = self.model.get_wavelength(instrument)
        self.view.set_wavelength(wavelength)

        filepath = self.model.get_raw_file_path(instrument)
        self.view.clear_run_info(filepath)

    def update_wavelength(self):
        wl_min, wl_max = self.view.get_wavelength()
        self.view.update_wavelength(wl_min)

    def calculate_peaks(self):
        hkl_1, hkl_2 = self.view.get_input_hkls()
        constants = self.view.get_lattice_constants()
        if constants is not None:
            d_phi = self.model.calculate_peaks(hkl_1, hkl_2, *constants)
            self.view.set_d_phi(*d_phi)

    def get_normal(self):
        slice_plane = self.view.get_slice()

        if slice_plane == "Axis 1/2":
            norm = [0, 0, 1]
        elif slice_plane == "Axis 1/3":
            norm = [0, 1, 0]
        else:
            norm = [1, 0, 0]

        return norm

    def get_clim_method(self):
        ctype = self.view.get_clim_clip_type()

        if ctype == "μ±3×σ":
            method = "normal"
        elif ctype == "Q₃/Q₁±1.5×IQR":
            method = "boxplot"
        else:
            method = None

        return method

    def reslice(self):
        if self.model.is_sliced():
            self.convert_to_hkl()

    def convert_to_hkl(self):
        if self.slice_idle:
            self.slice_idle = False

            worker = self.view.worker(self.convert_to_hkl_process)
            worker.connect_result(self.convert_to_hkl_complete)
            worker.connect_finished(self.update_complete)
            worker.connect_progress(self.update_processing)

            self.view.start_worker_pool(worker)

    def convert_to_hkl_complete(self, result):
        if result is not None:
            self.view.reset_slider()
            self.view.update_slice(result)
        self.slice_idle = True

    def convert_to_hkl_process(self, progress):
        proj = self.view.get_projection_matrix()

        value = self.view.get_slice_value()

        thickness = self.view.get_slice_thickness()

        width = self.view.get_slice_width()

        validate = [proj, value, thickness, width]

        if all(elem is not None for elem in validate):
            proj = np.array(proj).reshape(3, 3)

            if not np.isclose(np.linalg.det(proj), 0):
                U, V, W = proj

                norm = self.get_normal()

                progress("Processing...", 1)

                slice_histo = self.model.get_slice_info(
                    U, V, W, norm, value, thickness, width
                )

                progress("Updating slice...", 50)

                if slice_histo is not None:
                    signal = slice_histo["signal"]

                    clip = self.model.calculate_clim(
                        signal, self.get_clim_method()
                    )

                    slice_histo["clip"] = clip

                    progress("Slice drawn!", 100)

                    return slice_histo

                else:
                    progress("Invalid parameters.", 0)

        else:
            progress("Invalid parameters.", 0)

    def cluster(self):
        worker = self.view.worker(self.cluster_process)
        worker.connect_result(self.cluster_complete)
        worker.connect_progress(self.update_processing)

        self.view.start_worker_pool(worker)

    def cluster_complete(self, result):
        if result is not None:
            self.update_processing("Adding peaks.", 30)
            self.view.add_cluster_peaks(result)
            self.view.update_cluster_table(result)
            self.update_processing("Peaks added!", 0)

    def cluster_process(self, progress):
        params = self.view.get_cluster_parameters()

        if params is not None:
            progress("Invalid parameters.", 0)

            peak_info = self.model.get_cluster_info()
            if peak_info is not None:
                progress("Clustering peaks.", 25)

                success = self.model.cluster_peaks(peak_info, *params)

                if success:
                    progress("Peaks clustered!", 100)

                    return peak_info

                else:
                    progress("Invalid cluster.", 0)

        else:
            progress("Invalid parameters.", 0)

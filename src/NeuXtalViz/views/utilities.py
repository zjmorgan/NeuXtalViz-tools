import sys
import traceback
import contextlib
import io
import logging
import threading

from qtpy.QtCore import QRunnable, QThreadPool, Signal, QObject, Slot


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    progress = Signal(str, int)
    result = Signal(object)
    output = Signal(str)


class EmittingStream(io.StringIO):
    def __init__(self, emit_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emit_func = emit_func

    def write(self, s):
        super().write(s)
        self.emit_func(s)

    def flush(self):
        pass


class SignalLogHandler(logging.Handler):
    def __init__(self, emit_func):
        super().__init__()
        self.emit_func = emit_func

    def emit(self, record):
        msg = self.format(record)
        self.emit_func(msg)


class Worker(QRunnable):
    def __init__(self, task, *args, **kwargs):
        super().__init__()

        self.signals = WorkerSignals()
        self.task = task
        self.args = args
        self.kwargs = kwargs

        self.stop_event = threading.Event()
        self.kwargs["progress"] = self.emit_progress

    @Slot()
    def run(self):
        def emit_to_signal(s):
            if s:
                self.signals.output.emit(s)

        out_stream = EmittingStream(emit_to_signal)
        err_stream = EmittingStream(emit_to_signal)
        log_handler = SignalLogHandler(emit_to_signal)
        log_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(log_handler)
        try:
            with (
                contextlib.redirect_stdout(out_stream),
                contextlib.redirect_stderr(err_stream),
            ):
                try:
                    result = self.task(*self.args, **self.kwargs)
                except:
                    traceback.print_exc()
                    exctype, value = sys.exc_info()[:2]
                    self.signals.error.emit(
                        (exctype, value, traceback.format_exc())
                    )
                else:
                    self.signals.result.emit(result)
                finally:
                    self.signals.finished.emit()
        finally:
            root_logger.removeHandler(log_handler)

    def emit_progress(self, message, progress):
        self.signals.progress.emit(message, progress)

    def connect_result(self, process):
        self.signals.result.connect(process)

    def connect_finished(self, process):
        self.signals.finished.connect(process)

    def connect_progress(self, process):
        self.signals.progress.connect(process)


class ThreadPool(QThreadPool):
    def __init__(self):
        super().__init__()

    def start_worker_pool(self, worker):
        self.start(worker)

import atexit
import datetime as dt
import faulthandler
import logging
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path

_AUTO_ENABLED = False
_FAULT_FILE_HANDLE = None
_SESSION_LOG_PATH = None


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _get_log_root(log_root=None):
    if log_root is None:
        log_root = os.environ.get("HRL_AUTO_DIAG_DIR", "run_diagnostics/auto_global")
    return Path(log_root)


def get_auto_diagnostics_dir(log_root=None):
    return str(_get_log_root(log_root).resolve())


def get_session_log_path():
    return _SESSION_LOG_PATH


def cleanup_auto_diagnostics(remove_all_run_diagnostics=False):
    root = _get_log_root()
    if root.exists():
        shutil.rmtree(root)

    if remove_all_run_diagnostics:
        run_root = Path("run_diagnostics")
        if run_root.exists():
            shutil.rmtree(run_root)


def _setup_python_logger(log_root):
    log_root.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    session_log = log_root / f"session_{timestamp}_pid{pid}.log"

    logger = logging.getLogger("hrl.auto")
    logger.setLevel(logging.INFO)

    # Keep idempotent behavior when imported multiple times.
    if not logger.handlers:
        file_handler = logging.FileHandler(session_log, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, session_log


def _install_exception_hooks(logger):
    old_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        old_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    if hasattr(threading, "excepthook"):
        old_thread_hook = threading.excepthook

        def _thread_hook(args):
            logger.critical(
                "Unhandled thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            old_thread_hook(args)

        threading.excepthook = _thread_hook


def _setup_fault_handler(log_root, logger):
    global _FAULT_FILE_HANDLE

    fault_path = log_root / "python_fault_auto.log"
    _FAULT_FILE_HANDLE = open(fault_path, "a", buffering=1, encoding="utf-8")
    _FAULT_FILE_HANDLE.write(f"\n[{dt.datetime.now().isoformat()}] auto fault handler enabled\n")

    faulthandler.enable(file=_FAULT_FILE_HANDLE, all_threads=True)

    try:
        import signal

        if hasattr(signal, "SIGBREAK") and hasattr(faulthandler, "register"):
            faulthandler.register(signal.SIGBREAK, file=_FAULT_FILE_HANDLE, all_threads=True)
    except Exception as exc:
        logger.warning("Could not register SIGBREAK handler: %s", exc)

    return fault_path


def enable_auto_diagnostics(log_root=None):
    global _AUTO_ENABLED, _SESSION_LOG_PATH

    if _AUTO_ENABLED:
        return {
            "enabled": True,
            "already_enabled": True,
            "session_log": _SESSION_LOG_PATH,
            "log_dir": get_auto_diagnostics_dir(log_root),
        }

    if _truthy(os.environ.get("HRL_DISABLE_AUTO_DIAGNOSTICS", "0")):
        return {
            "enabled": False,
            "disabled_by_env": True,
            "session_log": None,
            "log_dir": get_auto_diagnostics_dir(log_root),
        }

    root = _get_log_root(log_root)
    logger, session_log = _setup_python_logger(root)
    _SESSION_LOG_PATH = str(session_log)

    fault_path = _setup_fault_handler(root, logger)
    _install_exception_hooks(logger)

    logger.info("Auto diagnostics enabled")
    logger.info("python=%s", sys.executable)
    logger.info("version=%s", sys.version)
    logger.info("cwd=%s", os.getcwd())
    logger.info("argv=%s", sys.argv)
    logger.info("fault_log=%s", fault_path)

    def _on_exit():
        logger.info("Process exiting normally")

    atexit.register(_on_exit)

    _AUTO_ENABLED = True
    return {
        "enabled": True,
        "already_enabled": False,
        "session_log": _SESSION_LOG_PATH,
        "fault_log": str(fault_path),
        "log_dir": str(root.resolve()),
    }


def _auto_start():
    try:
        enable_auto_diagnostics()
    except Exception:
        # Last-resort fallback to stderr so diagnostics never crashes user code.
        traceback.print_exc()


_auto_start()

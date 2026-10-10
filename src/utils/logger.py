import logging
import multiprocessing
import sys
from logging.handlers import QueueHandler, QueueListener

def setup_main_logger(log_file=None, level=logging.INFO):
    """
    Creates the main logger with a QueueListener for thread + process safety.
    Returns (logger, queue, listener).
    """
    log_queue = multiprocessing.Manager().Queue()
    logger = logging.getLogger("main_logger")
    logger.setLevel(level)
    logger.propagate = False

    # Console + file handlers
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [PID:%(process)d] [TID:%(thread)d] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(formatter)

    # Queue listener writes logs from all threads/processes
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    return logger, log_queue, listener

def get_worker_logger(queue, name="worker", level=logging.INFO):
    """
    Returns a process/thread-safe logger that sends logs into the shared queue.
    """
    qh = QueueHandler(queue)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(qh)
    logger.propagate = False
    return logger

import logging

from pytorch_lightning.utilities.rank_zero import rank_zero_only

def get_pylogger(name=__name__) -> logging.Logger:
    """Initialize multi GPU friendly python command line logger"""
    logger = logging.getLogger(name)

    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return 

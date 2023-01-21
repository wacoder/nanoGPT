import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)

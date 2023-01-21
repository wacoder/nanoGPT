from functools import wraps
from importlib.util import find_spec
from pydoc import resolve
from typing import Callable, Sequence
import warnings
from pathlib import Path
from click import style

import rich
import rich.tree
import rich.syntax
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.get_pylogger(__name__) 

def task_wrapper(task_func: Callable) -> Callable: 
    """decorator that wraps the task function in extra utilities

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_logger` after the task is finished or failed
    - Logging if the exception occurs
    - Logging the output dir
    """
    @wraps(task_func)
    def wrapper(cfg: DictConfig):
        try: 
            extras(cfg)
            metric_dict, object_dict = task_func(cfg)
        except Exception as e: 
            log.exception("")
            raise e
        finally: 
            log.info(f"Output dir: {cfg.paths.output_dir}")
            close_loggers()
        return metric_dict, object_dict
    return wrapper



def extras(cfg: DictConfig) -> None: 
    """Apply optional utilities before the task is started

    Utitlis: 
    - Ignore python warning
    - Setting tags from command line
    - Rich config printing
    """
    if not cfg.get("extras"):
        log.warning("Extra config not found! <cfg.extras=null>")

    if cfg.extras.get("ignore_warning"):
        log.info("Disable python warning! <cfg.extras.ignore_warning=True>")
        warnings.filterwarnings("ignore")
    
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tasg! <cfg.extras.enforce_tags=True>")
        enforce_tag(cfg, save_to_file=True)
    
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def enforce_tag(cfg: DictConfig, save_to_file: bool = False) -> None: 
    """Prompt user to input tag from command line if no tag is provided in the config

    Args: 
        cfg (DictConfig): configuration composed by Hydra
        save_to_file (bool, optional): whether to export config to the hydra output folder
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")
        
        log.warning("No tags provided in config, Prompting user to input tags ...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]
        
        with open_dict(cfg):
            cfg.tags = tags
        
        log.info(f"Tags: {cfg.tags}")
        
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig, 
    print_order: Sequence[str] = (
        "data", 
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure

    Args:
        cfg (DictConfig): configuration composed by Hydra
        print_order (Sequence[str], optional): determines in what order config components are printed
        resolve (bool, optional): whether to resolve reference fields of DictConfig
        save_to_file (bool, optional): whether to export the config to the hydra output folder
    """
    style - "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(f"{field} not found in config, Skipping '{field}' config printing ...")
    
    # add all the other fields to queue (not specified in the print_order)
    for field in cfg: 
        if field not in queue: 
            queue.append(field)

    # generate config tree from queue
    for field in queue: 
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    
    # print config tree
    rich.print(tree)

def close_loggers() -> None: 
    """Make sure all loggers closed properly"""
    
    log.info("Closing loggers ... ")
    
    if find_spec("wandb"):
        import wandb
        if wandb.run: 
            log.info("Close wandb!")
            wandb.finish()
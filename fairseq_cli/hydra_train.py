#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import subprocess

from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq_cli.train import main as pre_main
from fairseq import distributed_utils
from fairseq.dataclass.configs import FairseqConfig

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, open_dict


logger = logging.getLogger("fairseq_cli.hydra_train")


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> None:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        with open_dict(cfg):
            # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
            cfg.job_logging_cfg = OmegaConf.to_container(HydraConfig.get().job_logging, resolve=True)
    
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.set_struct(cfg, True)

    if cfg.common.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, pre_main)
    else:
        distributed_utils.call_main(cfg, pre_main)
    
    if cfg.common.shutdown:
        print('SHUTTING DOWN BOX', flush=True)
        if (os.getenv('AZ_VM_NAME') is not None) and (os.getenv('AZ_RES_GROUP') is not None):
            #os.system("az vm deallocate -n " + os.getenv('AZ_VM_NAME') + " -g " + os.getenv('AZ_RES_GROUP'))
            subprocess.Popen(['az', 'vm', 'deallocate', '-n', os.getenv('AZ_VM_NAME'), '-g', os.getenv('AZ_RES_GROUP')])
        else:
            os.system('sudo shutdown +5')


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()

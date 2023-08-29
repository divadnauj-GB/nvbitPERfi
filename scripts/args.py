import os
from datetime import datetime
from omegaconf import OmegaConf

# Retrieve the configs path
conf_path = os.path.dirname(__file__)

# Retrieve the default config
args = OmegaConf.load(os.path.join(conf_path, "fault_models_config.yaml"))

# Read the cli args
cli_args = OmegaConf.from_cli()

# read a specific config file
if "config" in cli_args and cli_args.config:
    conf_args = OmegaConf.load(cli_args.config)
    args = OmegaConf.merge(args, conf_args)
# else:
# raise NotImplementedError('You should introduce the config')

# Merge cli args into config ones
args = OmegaConf.merge(args, cli_args)

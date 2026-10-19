"""Helpers shared by the command-line scripts in ``scripts/``."""

from pathlib import Path

import torch
from hydra.conf import HelpConf, HydraConf, JobConf
from hydra.utils import to_absolute_path
from hydra_zen import ZenStore

from smf_net.model import SMFNet, load_model

# Hydra's help without its (empty here) list of configuration groups.
HELP_TEMPLATE = """${hydra.help.header}

== Config ==
Override anything in the config (foo.bar=value)

$CONFIG
${hydra.help.footer}"""

COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    """The torch dtype a script's ``dtype`` option names."""

    return COMPUTE_DTYPES[name]


def resolve_model_source(source: str) -> str:
    """Make a local model path absolute, leaving a Hub repository ID unchanged.

    Hydra runs scripts from their output directory, so a relative path has to
    be resolved against the directory the script was started from.
    """

    local = to_absolute_path(source)
    return local if Path(local).exists() else source


def load_model_for_inference(source: str, device: str) -> SMFNet:
    """Load a model onto ``device`` with float32 weights, compiled on CUDA.

    Reduced precision is applied per operation through each sampler's
    ``compute_dtype`` instead of by storing the weights at lower precision.
    """

    model = load_model(source).to(device=device, dtype=torch.float32)
    if torch.device(device).type == "cuda":
        model.model.compile()
    return model


def store_help(store: ZenStore, name: str, description: str) -> None:
    """Name a script's Hydra app and describe it at the top of ``--help``.

    Without this, Hydra names the app after hydra-zen's launcher module, which
    shows up in ``--help`` and as the run's log file name.
    """

    store(
        HydraConf(
            job=JobConf(name=name),
            help=HelpConf(app_name=name, header=description, template=HELP_TEMPLATE),
        ),
        name="config",
        group="hydra",
    )

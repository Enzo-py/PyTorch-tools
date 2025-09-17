import torch

from enum import Enum
from typing import Union

class MetricRegistry:
    """
        Metrics names are availables via the enum MetricName
    """

    def __init__(self):
        self._registry: dict[str, dict[str, Union[str, callable]]] = {}
        """{name: {'desc': str, 'fn': callable}, ...}"""

    def register(self, name, desc=None):
        def decorator(fn):
            if name in self._registry: raise KeyError(f"The metric <{name}> already exist.")

            self._registry[name] = {"desc": desc or name, "fn": fn}
            return fn
        return decorator

    def get(self, name):
        return self._registry[name]["fn"]

    def all(self) -> dict[str, dict[str, Union[str, callable]]]:
        """
            Return
            {name: {'desc': str, 'fn': callable}, ...}
        """
        return self._registry

METRICS = MetricRegistry()

class MetricName(str, Enum):
    ACC_001_2T = "Acc<0.01 2T"
    ACC_1 = "Acc<1.0"
    ACC_001 = "Acc<0.01"
    KL = "KL loss"
    RECST = "Recst. loss"
    SIMCLR = "SimCLR loss"
    DWST = "Downstream task loss"
    MU_MEAN = "μ.mean"
    MU_STD = "μ.std"
    LOGVAR_MEAN = "logσ².mean"
    LOGVAR_STD = "logσ².std"


# --- metrics code
@METRICS.register(MetricName.ACC_001_2T.value, "|x - recon(x)| < 0.01 (avg over towers)")
def acc_001_twotower(out, target, others):
    def acc_per_pair(x, x_recon):
        return 

    x1, x2, _ = target
    x1 = x1[..., (0, 2, 3)]
    x1_recon = others["model"]["x1"]["recon"]
    x2_recon = others["model"]["x2"]["recon"]

    acc1 = (torch.abs(x1 - x1_recon) < 0.01).float().mean()
    acc2 = (torch.abs(x2 - x2_recon) < 0.01).float().mean()
    return ((acc1 + acc2) / 2).item()


@METRICS.register(MetricName.ACC_1.value, "|x - y| < 1.0")
def acc_1(out, target, others):
    return (torch.abs(out - target) < 1.0).float().mean().item()

@METRICS.register(MetricName.ACC_001.value, "|x - y| < 0.01")
def acc_001(out, target, others):
    if isinstance(target, (list, tuple)): target = target[-1]
    return (torch.abs(out - target) < 0.01).float().mean().item()

@METRICS.register(MetricName.KL.value, "KL loss for VAE")
def kl(out, target, others):
    if 'KL' in others['loss']:
        return others['loss']['KL']
    else: raise KeyError(f"Metric <{MetricName.KL.value}> need a loss that return the KL sub loss.")

@METRICS.register(MetricName.RECST.value, "Reconstruction loss for VAE")
def rec(out, target, others):
    if 'REC' in others['loss']:
        return others['loss']['REC']
    else: raise KeyError(f"Metric <{MetricName.RECST.value}> need a loss that return the REC sub loss.")

@METRICS.register(MetricName.SIMCLR.value, "SIMCLR loss for VAE")
def simclr(out, target, others):
    if 'SIMCLR' in others['loss']:
        return others['loss']['SIMCLR']
    else: raise KeyError(f"Metric <{MetricName.SIMCLR.value}> need a loss that return the SIMCLR sub loss.")

@METRICS.register(MetricName.DWST.value, "Downstream task loss for VAE")
def simclr(out, target, others):
    if 'DWST' in others['loss']:
        return others['loss']['DWST']
    else: raise KeyError(f"Metric <{MetricName.DWST.value}> need a loss that return the DWST sub loss.")

@METRICS.register(MetricName.MU_MEAN.value, "Mean of μ")
def mu_mean(out, target, others):
    mu = others['model'].get("mu")
    if mu is None:
        raise KeyError(f"Metric <{MetricName.MU_MEAN.value}> needs 'mu' in others.")
    return mu.mean().detach().cpu().item()

@METRICS.register(MetricName.MU_STD.value, "Std of μ")
def mu_std(out, target, others):
    mu = others['model'].get("mu")
    if mu is None:
        raise KeyError(f"Metric <{MetricName.MU_STD.value}> needs 'mu' in others.")
    return mu.std().detach().cpu().item()

@METRICS.register(MetricName.LOGVAR_MEAN.value, "Mean of logσ²")
def logvar_mean(out, target, others):
    logvar = others['model'].get("logvar")
    if logvar is None:
        raise KeyError(f"Metric <{MetricName.LOGVAR_MEAN.value}> needs 'logvar' in others.")
    return logvar.mean().detach().cpu().item()

@METRICS.register(MetricName.LOGVAR_STD.value, "Std of logσ²")
def logvar_std(out, target, others):
    logvar = others['model'].get("logvar")
    if logvar is None:
        raise KeyError(f"Metric <{MetricName.LOGVAR_STD.value}> needs 'logvar' in others.")
    return logvar.std().detach().cpu().item()


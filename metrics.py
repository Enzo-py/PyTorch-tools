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
    ACC_001 = "Acc<0.01"
    ACC_1 = "Acc<1.0"
    KL = "KL loss"
    RECST = "Recst. loss"
    MU_MEAN = "μ.mean"
    MU_STD = "μ.std"
    LOGVAR_MEAN = "logσ².mean"
    LOGVAR_STD = "logσ².std"


# --- metrics code

@METRICS.register(MetricName.ACC_001.value, "|x - y| < 0.01")
def acc_001(out, target, others):
    return (torch.abs(out - target) < 0.01).float().mean().item()

@METRICS.register(MetricName.ACC_1.value, "|x - y| < 1.0")
def acc_1(out, target, others):
    return (torch.abs(out - target) < 1.0).float().mean().item()

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


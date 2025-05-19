import os
import importlib.util
import sys
import json
import torch

import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_sched

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Any, Union
from utils.models.metrics import METRICS
from utils.models.tracker import ModuleTracker


def save_json(data, path: str):
    """
    Sauvegarde dans un fichier JSON.

    Args:
        data (jsonable): Le dictionnaire à sauvegarder.
        path (str): Le chemin complet du fichier de sortie (ex: runs/mon_run/data.json).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

class Trainer:
    """
    1. use ModuleTracker.track to be able to track any model or loss
    2. use trainer
    """

    def __init__(
            self, 
            model_cls: type, model_config: dict[str, Any],
            loss_cls: type, loss_config: dict[str, Any],
            metrics: list[str],
            optimizer_cls:type=None, lr=0.001,
            scheduler_cls:type=None, scheduler_params=None,

            base_dir="runs", loading=False, run_name:str = None
        ):

        #init model
        self.model_cls = model_cls
        self.model: torch.nn.Module = model_cls(**model_config)

        #init loss
        self.loss_cls = loss_cls
        self.loss: torch.nn.Module = loss_cls(**loss_config)

        self.metrics = {name: METRICS.get(name) for name in metrics}

        if optimizer_cls is None: optimizer_cls = torch.optim.Adam
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        if scheduler_params is None and scheduler_cls is None: scheduler_params = {'mode': 'min', 'factor': 0.3, 'patience': 3, 'min_lr': 1e-6}
        elif scheduler_params is None: scheduler_params = {}

        if scheduler_cls is None:  scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)

        # save config
        self.config = {
            "model_config": model_config,
            "model_cls": model_cls.__name__,
            "loss_config": loss_config,
            "loss_module": None,
            "loss_cls": loss_cls.__name__,
            "metrics": metrics,
            "optimizer":{
                "class": optimizer_cls.__name__,
                "lr": lr
            },
            "scheduler": {
                "class": scheduler_cls.__name__,
                "params": scheduler_params
            }
        }

        self.base_dir = base_dir

        # generate run dir
        if loading:
            self.run_name = run_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model_cls.__name__
            self.run_name = f"{model_name}_{timestamp}"
        
        self.save_dir = Path(base_dir) / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        loss_module = getattr(loss_cls, '__module__', '')
        if loss_module.startswith('torch'): self.config['loss_module'] = loss_module
        elif loss_module.startswith('builtins'): self.config['loss_module'] = loss_module[len('builtins'):]
        else: self._save_module_code(self.loss, "loss.py")

        self._save_module_code(self.model, "model.py")
        save_json(self.config, self.save_dir / 'config.json')

        self.start_epoch = 0
        self.best_metric = None
        self.logs:dict[str, dict[str, Union[list[float], dict[str, list[float]]]]] = {
            'loss': {'train': [], 'test': []}, 
            'metrics': {name: {'train': [], 'test': []} for name in self.metrics}
        }
        """
        {
            'loss': {'train': [...], 'test': [...]}, 
            'metrics': {metric: {'train': [...], 'test': [...]}, ...}
        }
        """

        if loading:
            with open(os.path.join(self.save_dir, "log.json")) as f:
                self.logs = json.load(f)

    def _save_module_code(self, instance, filename):
        sources:list
        ref, sources = ModuleTracker.get_snapshot(instance)

        if sources is None:
            print("[warn] no tracked model (You may have forget the ModuleTracker.track decorator ?)")
            return

        sources.append(('imports', ModuleTracker.IMPORT_BLOCK))
        sources.reverse()
        with open(os.path.join(self.save_dir, filename), "w", encoding="utf-8") as f:
            for name, src in sources:
                f.write(f"# --- {name} ---\n{src}\n\n")

    def _resume_from_checkpoint(self, device):
        ckpt_path = self.save_dir / "checkpoint.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

            self.logs = ckpt["logs"]
            self.start_epoch = ckpt["epoch"]
            print(f"✅ Resuming from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
    
    def _log_epoch(self, train_loss, test_loss, train_metrics, test_metrics):
        self.logs["loss"]["train"].append(train_loss)
        self.logs["loss"]["test"].append(test_loss)
        for k in self.metrics:
            self.logs["metrics"][k]["train"].append(train_metrics[k])
            self.logs["metrics"][k]["test"].append(test_metrics[k])

    def _step(self, loader, device, train=False):
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics}
        num_batches = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                inputs = batch.to(device)

                outputs = self.model(inputs)
                others_outs = {}
                if isinstance(outputs, tuple): # differents values returned: (prediction, others_outs:dict)
                    outputs, others_outs = outputs

                loss = self.loss(outputs, inputs, others_outs)
                sub_losses = {}
                if isinstance(loss, tuple): # differents values returned: (total loss, sub_losses:dict)
                    loss, sub_losses = loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                for name, fn in self.metrics.items():
                    metric_totals[name] += fn(outputs, inputs, {'model': others_outs, 'loss': sub_losses})

                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_metrics = {k: float(v / num_batches) for k, v in metric_totals.items()}

        return avg_loss, avg_metrics

    def _run_epoch(self, epoch, train_loader, test_loader, device, verbose):
        train_loss, train_metrics = self._step(train_loader, device, train=True)
        test_loss, test_metrics = self._step(test_loader, device, train=False)

        self._log_epoch(train_loss, test_loss, train_metrics, test_metrics)

        main_metric = list(test_metrics.keys())[0] # TODO: unfix
        if self.best_metric is None or main_metric > self.best_metric:
            self.best_metric = main_metric
            self._save_best_sample(test_loader, device)

        self._save_checkpoint(epoch + 1)

        if verbose:
            self._print_epoch(epoch, train_loss, test_loss, train_metrics, test_metrics)

    def _print_epoch(self, epoch, train_loss, test_loss,
                    train_metrics: dict[str, float],
                    test_metrics : dict[str, float]) -> None:
        """
        Affiche un résumé lisible :
        • largeur des colonnes auto-ajustée au plus long nom de métrique
        • format numérique compact :  
            < 1 000      → 4 à 6 décimales  
            1 000-1e6    → 0 décimale  
        • flèche ↑ ↓ → pour la tendance de la dernière valeur test
        """
        def trend(key, mode="test"):
            hist = self.logs[key][mode] if key == "loss" else self.logs["metrics"][key][mode]
            if len(hist) < 2: return " "
            return "↑" if hist[-1] > hist[-2] else "↓" if hist[-1] < hist[-2] else "→"
        
        def fmt(v: float) -> str:
            """Formate les nombres avec séparation des milliers (pas de notation scientifique)."""
            abs_v = abs(v)
            if abs_v >= 1e6:
                return f"{v:,.0f}".replace(",", " ")
            elif abs_v >= 1e3:
                return f"{v:,.2f}".replace(",", " ")
            elif abs_v >= 1:
                return f"{v:,.4f}".replace(",", " ")
            else:
                return f"{v:,.6f}".replace(",", " ")

        # Largeur dynamique
        name_w = max(12, *(len(k) for k in self.metrics))
        V = [fmt(train_loss), fmt(test_loss)]
        for k in self.metrics:
            V += [fmt(train_metrics[k]), fmt(test_metrics[k])]
        val_w = max(10, *(len(v) for v in V))

        # En-tête
        hdr = f"\nEpoch {epoch + 1} Summary"
        sep = "-" * (name_w + 3 + 2 * (val_w + 8))

        # Ligne de loss
        loss_line = (f"{'Loss':<{name_w}} | "
                    f"Train: {fmt(train_loss):<{val_w}} | "
                    f"Test:  {fmt(test_loss):<{val_w}} {trend('loss')}")

        # Lignes de métriques
        metric_lines = [
            f"{k:<{name_w}} | "
            f"Train: {fmt(train_metrics[k]):<{val_w}} | "
            f"Test:  {fmt(test_metrics[k]):<{val_w}} {trend(k)}"
            for k in self.metrics
        ]

        print("\n".join([hdr, sep, loss_line] + metric_lines) + "\n")

    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "logs": self.logs,
            "best_metric": self.best_metric
        }

        path = self.save_dir / "checkpoint.pth"
        torch.save(checkpoint, path)

    def _save_best_sample(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            inputs = next(iter(test_loader)).to(device)
            outputs = self.model(inputs)
            if isinstance(outputs, tuple): # differents values returned: (prediction, *others)
                outputs, _ = outputs

        torch.save({
            "inputs": inputs.detach().cpu(),
            "outputs": outputs.detach().cpu()
        }, self.save_dir / "best_sample.pt")

    def _save_all(self):
        # Save final model weights
        torch.save(self.model.state_dict(), self.save_dir / "model.pth")

        # Save model + training config (architecture, loss, metrics)
        if hasattr(self, "config"):
            with open(self.save_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

        # Save logs
        with open(self.save_dir / "log.json", "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2)

    def train(self, num_epochs, train_loader, test_loader, device, verbose=True):
        self._resume_from_checkpoint(device)
        self.model.to(device)

        try:
            for epoch in tqdm(range(self.start_epoch, num_epochs), desc="Training", leave=False):
                self._run_epoch(epoch, train_loader, test_loader, device, verbose)
        
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted manually. Saving checkpoint...")

        self._save_all()
        print("✅ Training complete.")

    @torch.no_grad()
    def get(self, loader, device, others_keys=None):
        """
        Retourne un tuple (inputs, outputs, ~others) depuis le train ou test set.

        Paramètres :
            loader : torch.dataset.Loader
            device : torch.device
            others_keys: key of other value to return from the model
        
        Retour :
            inputs, outputs (tensors sur CPU)
            others: dict if others_keys is not None (not detach)
        """
        self.model.eval()
        inputs = next(iter(loader)).to(device)
        outputs = self.model(inputs)
        others = {}
        if isinstance(outputs, tuple): # differents values returned: (prediction, *others)
            outputs, others = outputs
        
        if others_keys is not None:
            return inputs.detach().cpu(), outputs.detach().cpu(), {k: others.get(k) for k in others_keys}
        return inputs.detach().cpu(), outputs.detach().cpu()


    def plot(self, *keys, title=None):
        """
        Affiche les courbes d'entraînement/test pour les clés spécifiées.

        Paramètres :
            keys : str
                Des clés comme "loss.train", "loss.test", "metrics.Acc<0.01.train"
            title : str ou None
                Titre du graphique (facultatif)
        """
        if not keys:
            raise ValueError("You must specify at least one log key to plot.")

        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(self.logs["loss"]["train"]) + 1)

        for key in keys:
            if key.startswith("loss."):
                _, split = key.split("loss.", 1)
                label = f"Loss {split}"
                values = self.logs["loss"][split]

            elif key.startswith("metrics."):
                try:
                    # Extract metric name and mode without breaking on inner dots
                    _, metric_and_mode = key.split("metrics.", 1)
                    *metric_parts, mode = metric_and_mode.rsplit(".", 1)
                    metric_name = ".".join(metric_parts)
                    label = f"{metric_name} {mode}"
                    values = self.logs["metrics"][metric_name][mode]
                except Exception as e:
                    raise ValueError(f"Invalid metric key format: '{key}'") from e

            else:
                raise ValueError(f"Invalid key format: '{key}'")

            ax.plot(epochs, values, label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(title or "Training Logs")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


    @staticmethod
    def load(run_name:str, base_dir="runs"):
        path_run = os.path.join(base_dir, run_name)

        with open(os.path.join(path_run, 'config.json'), 'r') as f:
            config = json.load(f)
        
        model_cls = Trainer._load_module_class(path_run, os.path.join(path_run, 'model.py'), config['model_cls'])
        loss_cls = Trainer._load_module_class(path_run, os.path.join(path_run, 'loss.py'), config['loss_cls'])

        optimizer_cls = getattr(optim, config["optimizer"]["class"])
        scheduler_cls = getattr(lr_sched, config["scheduler"]["class"])


        return Trainer(
            model_cls=model_cls,
            model_config=config['model_config'],
            loss_cls=loss_cls,
            loss_config=config['loss_config'],
            metrics=config['metrics'],

            optimizer_cls=optimizer_cls, lr=config["optimizer"]["lr"],
            scheduler_cls=scheduler_cls, scheduler_params=config["scheduler"]["params"],
            
            base_dir=base_dir,
            loading=True,
            run_name=run_name
        )
        

    @staticmethod
    def _load_module_class(path_run:str, module_path:str, class_name:str):
        """Charge dynamiquement une classe de modèle depuis un fichier Python."""

        module_name = "_dynamic_module" + path_run
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try: spec.loader.exec_module(module)
        except FileNotFoundError as e: raise ImportError(f"The run <{path_run}> doesn't have the saved version of the module <{class_name}>.")
        except Exception as e: raise e

        return getattr(module, class_name)

import inspect
import weakref
import re
from IPython import get_ipython
import torch.nn as nn

class ModuleTracker:
    _snapshots = {}

    IMPORT_BLOCK = """\
        import os
        import sys
        import torch
        import math

        import torch.nn.functional as F

        from torch import nn

        sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
        from utils.models.tracker import ModuleTracker
    """.replace('    ', '')

    @classmethod
    def track(cls, model_cls):
        """
        Décorateur à appliquer sur une classe nn.Module
        pour capturer récursivement les classes utilisées.
        """

        if not issubclass(model_cls, nn.Module): raise TypeError(f"The class <{model_cls}> doesn't inherit from nn.Module")
        original_init = model_cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            instance_id = id(self)
            classes = cls._collect_used_classes(self)
            cls._snapshots[instance_id] = (weakref.ref(self), classes)

        model_cls.__init__ = __init__
        return model_cls

    @classmethod
    def get_snapshot(cls, model_instance) -> tuple:
        return cls._snapshots.get(id(model_instance), (None, None))

    @staticmethod
    def _collect_used_classes(instance: nn.Module):
        visited = set()
        results = []

        for module in instance.modules():
            cls = module.__class__
            if cls in visited or not ModuleTracker._is_custom_class(cls):
                continue
            visited.add(cls)
            source = ModuleTracker._get_source_safe(cls)
            results.append((cls.__name__, source))
        return results

    @staticmethod
    def _is_custom_class(cls):
        mod = getattr(cls, "__module__", "")
        return not (mod.startswith("torch") or mod.startswith("builtins"))

    @staticmethod
    def _get_source_safe(cls):
        try:
            return inspect.getsource(cls)
        except (OSError, TypeError):
            pass

        # Try to extract from notebook
        try:
            ip = get_ipython()
            if ip:
                cells = ip.history_manager.input_hist_raw
                return ModuleTracker._extract_class_block_from_cells(cls.__name__, cells)
        except Exception as e:
            print(f"[get_source_safe] notebook fallback failed: {e}")

        return f"# Source unavailable for class {cls.__name__}"

    @staticmethod
    def _extract_class_block_from_cells(class_name, cells):
        pattern = re.compile(rf"^\s*class\s+{class_name}\b")

        for cell in reversed(cells):
            lines = cell.splitlines()
            for i, line in enumerate(lines):
                if pattern.match(line):
                    start_idx = i
                    while start_idx > 0 and lines[start_idx - 1].strip().startswith("@"):
                        start_idx -= 1

                    base_indent = len(lines[i]) - len(lines[i].lstrip())
                    end_idx = len(lines)

                    for j in range(i + 1, len(lines)):
                        l = lines[j]
                        stripped = l.strip()
                        indent = len(l) - len(stripped)

                        if stripped.startswith("@"):
                            if j + 1 < len(lines):
                                next_line = lines[j + 1].strip()
                                if re.match(r"^(class|def)\s+", next_line):
                                    end_idx = j
                                    break
                        elif re.match(r"^(class|def)\s+", stripped) and indent <= base_indent:
                            end_idx = j
                            break

                    return "\n".join(lines[start_idx:end_idx])
        return None

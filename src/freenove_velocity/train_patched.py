"""Wrapper that monkey-patches rsl_rl to clamp noise std, then runs training.

Prevents RuntimeError: normal expects all elements of std >= 0.0
"""

import torch
import torch.distributions as D


def _patch():
    from rsl_rl.models.mlp_model import MLPModel

    _orig = MLPModel.forward

    def _safe_forward(self, obs, **kwargs):
        result = _orig(self, obs, **kwargs)
        if hasattr(self, "distribution") and isinstance(self.distribution, D.Normal):
            self.distribution = D.Normal(
                self.distribution.loc,
                self.distribution.scale.clamp(min=0.01),
            )
        return result

    MLPModel.forward = _safe_forward
    print("[patch] MLPModel noise std clamped to min=0.01")


def main():
    _patch()
    # Import and run the real mjlab train CLI
    import importlib
    import sys

    # Find the original 'train' entry point from mjlab
    from importlib.metadata import distribution

    dist = distribution("mjlab")
    for ep in dist.entry_points:
        if ep.name == "train" and ep.group == "console_scripts":
            # Load the entry point function
            mod_path, func_name = ep.value.split(":")
            mod = importlib.import_module(mod_path)
            func = getattr(mod, func_name)
            func()
            return

    # Fallback: try common patterns
    try:
        from mjlab.cli import app
        app()
    except ImportError:
        from mjlab.cli import main as mjlab_main
        mjlab_main()


if __name__ == "__main__":
    main()

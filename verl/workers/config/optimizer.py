# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING, OmegaConf

from verl.base_config import BaseConfig

__all__ = ["OptimizerConfig", "FSDPOptimizerConfig", "McoreOptimizerConfig", "build_optimizer", "VeOmniOptimizerConfig"]

# Upstream: https://github.com/KellerJordan/Muon — MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam only.
# Sigmoid fork: https://github.com/PhiVanDat123/sigmoid_muon — ships all four class names in one `muon` module:
#   MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam_sigmoid, SingleDeviceMuonWithAuxAdam_sigmoid
MUON_AUXADAM_OPTIMIZER_NAMES = frozenset(
    {
        "MuonWithAuxAdam",
        "SingleDeviceMuonWithAuxAdam",
        "MuonWithAuxAdam_sigmoid",
        "SingleDeviceMuonWithAuxAdam_sigmoid",
    }
)


def _cfg_get(config: Any, name: str, default=None):
    """Read a field from either a dataclass config or an OmegaConf DictConfig."""
    if OmegaConf.is_config(config):
        v = OmegaConf.select(config, name, default=default)
        if v is None or v is MISSING:
            return default
        return v
    v = getattr(config, name, default)
    if v is None or v is MISSING:
        return default
    return v


def _normalize_betas(betas) -> tuple[float, float]:
    if betas is None:
        return (0.9, 0.999)
    if isinstance(betas, (list, tuple)) and len(betas) >= 2:
        return (float(betas[0]), float(betas[1]))
    return (0.9, 0.999)


def _build_muon_with_aux_adam_optimizer(module: Any, config: Any):
    """Build Muon + auxiliary AdamW param groups (upstream or sigmoid fork, same param_groups API)."""
    opt_name = _cfg_get(config, "optimizer", "AdamW")
    try:
        import muon as _muon_pkg
    except ImportError as e:
        raise ImportError(
            "Muon aux+Adam optimizers require a `muon` Python module. Recommended (all 4 optimizer classes):\n"
            "  pip install git+https://github.com/PhiVanDat123/sigmoid_muon\n"
            "Minimal upstream (MuonWithAuxAdam + SingleDeviceMuonWithAuxAdam only, no *_sigmoid):\n"
            "  pip install git+https://github.com/KellerJordan/Muon\n"
            "(Both use import name `muon`; install only one environment-wide to avoid clashes.)\n"
        ) from e

    cls = getattr(_muon_pkg, opt_name, None)
    if cls is None:
        raise AttributeError(
            f"Optimizer class `{opt_name}` not found in installed `muon` package. "
            f"Available names include: {[n for n in dir(_muon_pkg) if not n.startswith('_')]}\n"
            "For all four options (MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam, *_sigmoid), install:\n"
            "  pip install git+https://github.com/PhiVanDat123/sigmoid_muon\n"
            "Upstream KellerJordan/Muon does not define the *_sigmoid classes.\n"
            "  https://github.com/PhiVanDat123/sigmoid_muon"
        )

    excludes = _cfg_get(config, "muon_exclude_name_substrings")
    if not excludes:
        excludes = [
            "embed",
            "wte",
            "wpe",
            "lm_head",
            "tok_embeddings",
            "token_emb",
            "vision_model",
            "visual",
        ]
    excludes_l = tuple(str(s).lower() for s in excludes)

    muon_params: list = []
    adam_params: list = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        nl = name.lower()
        if any(s in nl for s in excludes_l):
            adam_params.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    adam_aux_lr_cfg = _cfg_get(config, "adam_aux_lr", None)
    lr_adam = (
        float(adam_aux_lr_cfg)
        if adam_aux_lr_cfg is not None
        else float(_cfg_get(config, "lr", 3e-4))
    )
    wd = float(_cfg_get(config, "weight_decay", 0.0))
    betas = _normalize_betas(_cfg_get(config, "betas", (0.9, 0.999)))
    eps = float(_cfg_get(config, "adam_aux_eps", 1e-10))
    muon_lr_cfg = _cfg_get(config, "muon_lr", None)
    muon_lr = float(muon_lr_cfg) if muon_lr_cfg is not None else 0.02
    muon_momentum = float(_cfg_get(config, "muon_momentum", 0.95))

    param_groups = []
    if adam_params:
        param_groups.append(
            dict(
                params=adam_params,
                lr=lr_adam,
                betas=betas,
                eps=eps,
                weight_decay=wd,
                use_muon=False,
            )
        )
    if muon_params:
        param_groups.append(
            dict(
                params=muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=wd,
                use_muon=True,
            )
        )
    if not param_groups:
        raise ValueError(f"{opt_name}: no trainable parameters found on module.")

    print(
        f"[build_optimizer] {opt_name}: {len(muon_params)} tensors -> Muon, "
        f"{len(adam_params)} tensors -> Adam (name excludes: {list(excludes_l)})"
    )

    return cls(param_groups)


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
        muon_lr (Optional[float]): LR for the Muon branch when optimizer is MuonWithAuxAdam /
            SingleDeviceMuonWithAuxAdam. Defaults to 0.02 if unset.
        muon_momentum (float): Momentum for the Muon branch.
        muon_exclude_name_substrings (Optional[list[str]]): Name substrings forcing the Adam branch.
        adam_aux_eps (float): Epsilon for the auxiliary Adam branch.
        adam_aux_lr (Optional[float]): If set, learning rate for the auxiliary Adam branch only (Muon* optimizers).
            If None, the top-level ``lr`` is used for that branch (same as standard AdamW in this config).
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None
    muon_lr: Optional[float] = None
    muon_momentum: float = 0.95
    muon_exclude_name_substrings: Optional[list[str]] = None
    adam_aux_eps: float = 1e-10
    adam_aux_lr: Optional[float] = None

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


def build_optimizer(parameters, config: FSDPOptimizerConfig, module: Any = None):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings
        module: Optional `nn.Module` (required for MuonWithAuxAdam / SingleDeviceMuonWithAuxAdam /
            MuonWithAuxAdam_sigmoid / SingleDeviceMuonWithAuxAdam_sigmoid) used to split `named_parameters()`
            into Muon vs auxiliary Adam groups.

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"

        # Muon + Adam. Requires `module=`. One install of sigmoid_muon exposes all four classes:
        #   pip install git+https://github.com/PhiVanDat123/sigmoid_muon
        # config.optimizer = "SingleDeviceMuonWithAuxAdam_sigmoid"  # or MuonWithAuxAdam / SingleDeviceMuonWithAuxAdam / MuonWithAuxAdam_sigmoid
        # build_optimizer(model.parameters(), config, module=model)
    """
    import importlib

    opt_name = _cfg_get(config, "optimizer", "AdamW")
    if opt_name in MUON_AUXADAM_OPTIMIZER_NAMES:
        if module is None:
            raise ValueError(
                f"`{opt_name}` requires `module` (the trainable `nn.Module`) so parameters can be split into "
                "Muon vs Adam groups. Call: build_optimizer(model.parameters(), config, module=model)."
            )
        return _build_muon_with_aux_adam_optimizer(module, config)

    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = str(opt_name).lower()
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    try:
        impl_mod = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(impl_mod, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(impl_mod)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)

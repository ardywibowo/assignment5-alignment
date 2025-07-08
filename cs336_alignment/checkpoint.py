import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class AppState(Stateful):
    """Container that Torch-DCP can traverse."""

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        steps: int = 0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.steps = steps

    def state_dict(self):
        model_sd = (
            get_state_dict(self.model, self.optimizer)[0]
            if self.optimizer is not None
            else get_model_state_dict(self.model)
        )
        sd: dict[str, Any] = {"model": model_sd, "steps": self.steps}
        if self.optimizer is not None:
            _, optim_sd = get_state_dict(self.model, self.optimizer)
            sd["optimizer"] = optim_sd
        if self.scheduler is not None:
            sd["scheduler"] = self.scheduler.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        opts = StateDictOptions(strict=False)  # loose load
        set_model_state_dict(self.model, state_dict["model"], options=opts)

        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

        if self.optimizer is not None and "optimizer" in state_dict:
            set_optimizer_state_dict(
                self.model, self.optimizer, state_dict["optimizer"], options=opts
            )

        self.steps = state_dict.get("steps", 0)


class CheckpointManager:
    """
    Thin wrapper around Torch-DCP.

    * Handles async CPU process-group creation
    * Works in multi-GPU and single-process modes
    * Stores model / optimiser / scheduler + `global_step`
    """

    _cpu_pg: Optional[dist.ProcessGroup] = None  # shared singleton

    def __init__(
        self,
        dir: str,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        *,
        async_pg: Optional[dist.ProcessGroup] = None,
    ):
        self.dir = Path(dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        if async_pg is not None:
            self.async_pg = async_pg
        elif dist.is_initialized() and dist.get_world_size() > 1:
            self.async_pg = self._get_or_create_cpu_pg()
        else:
            self.async_pg = None

        self._inflight_future: Optional[Any] = None

    def save_async(self, step: int, tag: str | None = None):
        ckpt_path = self._make_path(step, tag)
        self._wait_for_previous_future()

        future = dcp.async_save({"app_state": AppState(self.model,
                                                    self.optimizer,
                                                    self.scheduler,
                                                    step)},
                                checkpoint_id=ckpt_path,
                                process_group=self.async_pg)

        if self._is_primary():
            out_name = f"model_step_{step}.pt" if tag is None else f"{tag}.pt"
            def _after(_):
                self._consolidate_to_weights_only(ckpt_path, self.dir / out_name)
                print(f"[CheckpointManager] Consolidated → {out_name}")
            future = future.then(_after)

        self._inflight_future = future
        if self._is_primary():
            print(f"[CheckpointManager] async_save → {ckpt_path} (step {step})")
        return future


    def save(self, step: int, tag: str | None = None) -> str:
        ckpt_path = self._make_path(step, tag)
        self._wait_for_previous_future()

        dcp.save({"app_state": AppState(self.model,
                                        self.optimizer,
                                        self.scheduler,
                                        step)},
                checkpoint_id=ckpt_path,
                process_group=self.async_pg)

        if self._is_primary():
            out_name = f"model_step_{step}.pt" if tag is None else f"{tag}.pt"
            self._consolidate_to_weights_only(ckpt_path, self.dir / out_name)

        if self._is_primary():
            print(f"[CheckpointManager] Saved {ckpt_path} (step {step})")
        return ckpt_path

    def save_final(self, step: int) -> str:
        """Shortcut that writes `final_model.pth` synchronously."""
        return self.save(step, tag="final_model")

    def load(
        self,
        path: str | os.PathLike,
        *,
        strict: bool = True,
        weights_only: bool = False,
    ) -> int:
        """
        Restore a checkpoint produced by `save` / `save_async`.

        Returns the stored `global_step`.
        """
        app_state = (
            AppState(self.model)
            if weights_only
            else AppState(self.model, self.optimizer, self.scheduler)
        )

        if not path:
            return app_state.steps  # nothing to load

        path = str(Path(path).expanduser().resolve())
        planner = DefaultLoadPlanner(allow_partial_load=not strict)

        dcp.load(
            {"app_state": app_state},
            checkpoint_id=path,
            process_group=self.async_pg,
            planner=planner,
        )

        if self._is_primary():
            who = (
                " (model-only)"
                if weights_only or self.optimizer is None
                else " w/ optimiser & sched."
            )
            mode = "strict" if strict else "partial"
            print(
                f"[CheckpointManager] Restored {path} at step {app_state.steps}{who} ({mode})"
            )

        return app_state.steps

    def _consolidate_to_weights_only(self, ckpt_path: str, out_path: str):
        """
        Convert the Torch-DCP checkpoint shards at `ckpt_path`
        into a single .pt that holds **just** the model's state_dict.
        """
        # 1. Collapse the shards → a temporary .pt file
        tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.pt"
        dcp_to_torch_save(ckpt_path, tmp_path)

        # 2. Strip everything except the model weights
        full = torch.load(tmp_path, map_location="cpu")
        torch.save(full["app_state"]["model"], out_path)

        tmp_path.unlink(missing_ok=True)


    def _wait_for_previous_future(self):
        if self._inflight_future is not None:
            self._inflight_future.result()
            self._inflight_future = None

    def _make_path(self, step: int, tag: Optional[str]) -> str:
        filename = f"{tag or f'checkpoint_step_{step}'}"
        if not filename.endswith(".pth"):
            filename += ".pth"
        return str(self.dir / filename)

    def _is_primary(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    @classmethod
    def _get_or_create_cpu_pg(cls):
        if cls._cpu_pg is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed not initialised; call dist.init_process_group() "
                    "or pass an explicit process_group."
                )
            cls._cpu_pg = dist.new_group(backend="gloo")
        return cls._cpu_pg

    def __del__(self):
        try:
            self._wait_for_previous_future()
        except Exception:
            pass

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

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


# ══════════════════════════════════════════════════════════════════════════
#                               Checkpointing
# ══════════════════════════════════════════════════════════════════════════
class AppState(Stateful):
    """Container that Torch‑DCP can traverse."""

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

    # ---- Stateful hooks --------------------------------------------------
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
        opts = StateDictOptions(strict=False)
        set_model_state_dict(self.model, state_dict["model"], options=opts)

        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

        if self.optimizer is not None and "optimizer" in state_dict:
            set_optimizer_state_dict(
                self.model, self.optimizer, state_dict["optimizer"], options=opts
            )

        self.steps = state_dict.get("steps", 0)


# -------------------------------------------------------------------------
class CheckpointManager:
    """
    Thin wrapper around Torch‑DCP.

    * All ranks enter every save / async_save call → no collective hangs.
    * Handles both C++ (`.then`) and Python (`.add_done_callback`) futures.
    """

    _cpu_pg: Optional[dist.ProcessGroup] = None  # shared singleton

    def __init__(
        self,
        dir: str,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.dir = Path(dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # extra CPU process‑group for async I/O if running distributed
        if dist.is_initialized() and dist.get_world_size() > 1:
            self.async_pg = self._get_or_create_cpu_pg()
        else:
            self.async_pg = None

        self._inflight_future: Optional[Any] = None

    # ── public helpers ────────────────────────────────────────────────────
    def save_async(self, step: int, tag: str | None = None):
        ckpt_path = self._make_path(step, tag)
        self._wait_for_previous_future()

        future = dcp.async_save(
            {"app_state": AppState(self.model, self.optimizer, self.scheduler, step)},
            checkpoint_id=ckpt_path,
            process_group=self.async_pg,
        )

        if self._is_primary():
            out_name = f"model_step_{step}.pt" if tag is None else f"{tag}.pt"

            def _after():
                self._consolidate_to_weights_only(ckpt_path, self.dir / out_name)
                print(f"[CheckpointManager] Consolidated → {out_name}")

            self._chain_future(future, _after)

        self._inflight_future = future
        if self._is_primary():
            print(f"[CheckpointManager] async_save → {ckpt_path} (step {step})")
        return future

    def save(self, step: int, tag: str | None = None) -> str:
        ckpt_path = self._make_path(step, tag)
        self._wait_for_previous_future()

        dcp.save(
            {"app_state": AppState(self.model, self.optimizer, self.scheduler, step)},
            checkpoint_id=ckpt_path,
            process_group=self.async_pg,
        )

        if self._is_primary():
            out_name = f"model_step_{step}.pt" if tag is None else f"{tag}.pt"
            self._consolidate_to_weights_only(ckpt_path, self.dir / out_name)
            print(f"[CheckpointManager] Saved {ckpt_path} (step {step})")

        return ckpt_path

    def save_final(self, step: int) -> str:
        return self.save(step, tag="final_model")

    def load(
        self,
        path: str | os.PathLike,
        *,
        strict: bool = True,
        weights_only: bool = False,
    ) -> int:
        """Restore checkpoint; returns stored `global_step`."""
        app_state = (
            AppState(self.model)
            if weights_only
            else AppState(self.model, self.optimizer, self.scheduler)
        )
        if not path:
            return app_state.steps

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
                " (model‑only)"
                if weights_only or self.optimizer is None
                else " w/ optimiser & sched."
            )
            mode = "strict" if strict else "partial"
            print(
                f"[CheckpointManager] Restored {path} at step {app_state.steps}{who} ({mode})"
            )
        return app_state.steps

    # ── internals ─────────────────────────────────────────────────────────
    def _consolidate_to_weights_only(self, ckpt_path: str, out_path: str):
        """Collapse shards → tmp.pt → strip optimiser/sched → final .pt"""
        tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.pt"
        dcp_to_torch_save(ckpt_path, tmp_path)

        full = torch.load(tmp_path, map_location="cpu")
        torch.save(full["app_state"]["model"], out_path)
        tmp_path.unlink(missing_ok=True)

    # ---- future utilities ------------------------------------------------
    def _chain_future(self, fut: Any, cb: Callable[[], None]):
        """
        Attach `cb` to `fut`, supporting both torch._C.Future (`then`)
        and concurrent.futures.Future (`add_done_callback`).
        """
        if hasattr(fut, "then"):
            fut.then(lambda _: cb())
        elif hasattr(fut, "add_done_callback"):
            fut.add_done_callback(lambda _f: cb())
        else:  # last‑resort: block synchronously
            fut.wait() if hasattr(fut, "wait") else fut.result()
            cb()

    def _wait_for_previous_future(self):
        if self._inflight_future is not None:
            if hasattr(self._inflight_future, "wait"):
                self._inflight_future.wait()
            else:
                self._inflight_future.result()
            self._inflight_future = None

    # ---- misc helpers ----------------------------------------------------
    def _make_path(self, step: int, tag: Optional[str]) -> str:
        filename = f"{tag or f'checkpoint_step_{step}'}"
        if not filename.endswith(".pth"):
            filename += ".pth"
        return str(self.dir / filename)

    def _is_primary(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    # ---- singleton CPU group -------------------------------------------
    @classmethod
    def _get_or_create_cpu_pg(cls):
        if cls._cpu_pg is None:
            cls._cpu_pg = dist.new_group(backend="gloo")
        return cls._cpu_pg

    # ---- destructor -----------------------------------------------------
    def __del__(self):
        try:
            self._wait_for_previous_future()
        except Exception:
            pass

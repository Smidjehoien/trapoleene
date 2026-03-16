#!/usr/bin/env python3
"""
train_midihits_llama_v5_live_metrics.py

LLaMA pretrain for packed MidiTok REMI blocks, with live nerd-metrics:
- Optional torch.compile + FlashAttention2
- BF16 + TF32
- Gradient checkpointing
- TensorBoard (if installed) OR CSV fallback (always)
- GPU stats (memory always; temp/util/power if pynvml or nvidia-smi available)
- Hard fix: disables TorchDynamo "repro" that tries to run `nvcc --version` (can crash on restricted systems)

Works well on: RTX 5080 / 16GB VRAM + lots of RAM.

Typical:
  python train_midihits_llama_v5_live_metrics.py \
    --bf16 --tf32 --attn_impl flash_attention_2 --torch_compile \
    --grad_ckpt --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 3 \
    --tb_log --tb_dir out/tb_runs --log_every 10 --gpu_stats_every 50

Viewing logs:
  - If `tensorboard` command is missing, use:
      python -m tensorboard --logdir out/tb_runs --port 6006 --bind_all
  - CSV fallback:
      tail -f out/tb_runs/<run_id>/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup

try:
    from miditok import REMI  # type: ignore
except Exception:
    REMI = None


# ---------------------------
# Safety: disable nvcc repro
# ---------------------------
def disable_dynamo_repro_nvcc() -> None:
    """
    PyTorch/Inductor can try to generate a "repro" on compile failures and calls `nvcc --version`.
    On some systems nvcc exists but is not executable -> PermissionError -> crash.
    Disabling repro avoids that and also keeps error messages cleaner.
    """
    os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
    try:
        import torch._dynamo.config as dconf  # type: ignore
        dconf.repro_after = None
    except Exception:
        pass

    # Extra belt-and-suspenders: even if something flips repro back on, don't call nvcc.
    try:
        import torch._dynamo.debug_utils as dbg  # type: ignore

        def _no_cuda_system_info_comment() -> str:
            return "# [torch.compile repro disabled: nvcc not queried]\n"

        if hasattr(dbg, "_cuda_system_info_comment"):
            dbg._cuda_system_info_comment = _no_cuda_system_info_comment  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------
# Dataset
# ---------------------------
class PackedNpyDataset(Dataset):
    def __init__(self, npy_path: Path, load_to_ram: bool = True):
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"Packed .npy not found: {self.npy_path.resolve()}")
        print(f"[INFO] Loading {self.npy_path.name} -> {'RAM' if load_to_ram else 'mmap'}")
        if load_to_ram:
            arr = np.load(self.npy_path)
        else:
            arr = np.load(self.npy_path, mmap_mode="r")
        self.arr = arr
        self.n, self.t = arr.shape
        print(f"[INFO] Dataset ready: {self.n} blocks x {self.t} tokens")

    def __len__(self) -> int:
        return int(self.n)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.arr[idx]


def collate_batch(rows: List[np.ndarray]) -> torch.Tensor:
    # rows are fixed-length already
    batch = np.stack(rows, axis=0)
    return torch.from_numpy(batch)


# ---------------------------
# Tokenizer / vocab utilities
# ---------------------------
def get_vocab_size_from_tokenizer_json(tokenizer_json: Path) -> int:
    if REMI is not None:
        tok = REMI(params=str(tokenizer_json))
        # miditok exposes one of these depending on version
        for attr in ("vocab_size", "vocab_size_"):
            if hasattr(tok, attr):
                return int(getattr(tok, attr))
        # fallback
        try:
            return int(len(tok))
        except Exception:
            pass

    # generic json fallback
    with open(tokenizer_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "vocab_size" in data:
        return int(data["vocab_size"])
    # MidiTok sometimes stores merges/vocab differently; len(data.get("vocab", {})) might work:
    if isinstance(data, dict) and "vocab" in data and isinstance(data["vocab"], dict):
        return int(len(data["vocab"]))
    raise RuntimeError(f"Could not infer vocab_size from {tokenizer_json}")


# ---------------------------
# Checkpointing
# ---------------------------
def try_save_pretrained_else_fallback(
    model: LlamaForCausalLM,
    out_dir: Path,
    tokenizer_json: Optional[Path] = None,
    extra_state: Optional[dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(out_dir))
    except Exception as e:
        print(f"[WARN] save_pretrained failed ({e}) -> saving state_dict")
        model.config.to_json_file(str(out_dir / "config.json"))
        torch.save(model.state_dict(), str(out_dir / "pytorch_model.bin"))

    if tokenizer_json and tokenizer_json.exists():
        shutil.copy2(tokenizer_json, out_dir / "tokenizer.json")
    if extra_state:
        with open(out_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(extra_state, f, indent=2)


# ---------------------------
# Logging helpers
# ---------------------------
def make_run_dir(base: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def try_make_tb_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        return SummaryWriter(log_dir=str(run_dir))
    except Exception:
        return None


def csv_logger_open(run_dir: Path):
    csv_path = run_dir / "metrics.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow([
        "time_utc",
        "epoch",
        "micro_step",
        "opt_step",
        "train_loss",
        "lr",
        "tokens_per_s",
        "step_time_s",
        "gpu_mem_alloc_gb",
        "gpu_mem_reserved_gb",
        "gpu_mem_max_alloc_gb",
        "gpu_temp_c",
        "gpu_util_pct",
        "gpu_power_w",
    ])
    f.flush()
    return f, w, csv_path


def _fmt_gb(x: float) -> float:
    return float(x) / (1024**3)


def get_gpu_mem_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"alloc_gb": 0.0, "reserved_gb": 0.0, "max_alloc_gb": 0.0}
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    return {"alloc_gb": _fmt_gb(alloc), "reserved_gb": _fmt_gb(reserved), "max_alloc_gb": _fmt_gb(max_alloc)}


class GpuTelemetry:
    """
    GPU temp/util/power, best-effort:
      1) pynvml (fastest)
      2) nvidia-smi query (slower, but works without python libs)
    """
    def __init__(self) -> None:
        self.mode = "none"
        self._nvml = None
        self._handle = None

        # Try NVML
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.mode = "pynvml"
            return
        except Exception:
            self._nvml = None
            self._handle = None

        # Try nvidia-smi existence
        try:
            subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT)
            self.mode = "nvidia-smi"
        except Exception:
            self.mode = "none"

    def read(self) -> Dict[str, Optional[float]]:
        if self.mode == "pynvml" and self._nvml and self._handle:
            try:
                nvml = self._nvml
                temp = float(nvml.nvmlDeviceGetTemperature(self._handle, nvml.NVML_TEMPERATURE_GPU))
                util = float(nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                pwr_mw = float(nvml.nvmlDeviceGetPowerUsage(self._handle))
                return {"temp_c": temp, "util_pct": util, "power_w": pwr_mw / 1000.0}
            except Exception:
                return {"temp_c": None, "util_pct": None, "power_w": None}

        if self.mode == "nvidia-smi":
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.STDOUT,
                ).decode("utf-8", errors="ignore").strip()
                if not out:
                    return {"temp_c": None, "util_pct": None, "power_w": None}
                parts = [p.strip() for p in out.split(",")]
                temp = float(parts[0]) if len(parts) > 0 else None
                util = float(parts[1]) if len(parts) > 1 else None
                pwr = float(parts[2]) if len(parts) > 2 else None
                return {"temp_c": temp, "util_pct": util, "power_w": pwr}
            except Exception:
                return {"temp_c": None, "util_pct": None, "power_w": None}

        return {"temp_c": None, "util_pct": None, "power_w": None}


def sparkline(vals: List[float], width: int = 30) -> str:
    # tiny ASCII sparkline for terminal fun
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    v = vals[-width:]
    mn, mx = min(v), max(v)
    if mx - mn < 1e-9:
        return blocks[0] * len(v)
    out = []
    for x in v:
        idx = int((x - mn) / (mx - mn) * (len(blocks) - 1))
        out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    return "".join(out)


# ---------------------------
# Main
# ---------------------------
def resolve_packed_paths(packed_dir: Path) -> Tuple[Path, Path]:
    return packed_dir / "train_input_ids.npy", packed_dir / "valid_input_ids.npy"


def main() -> None:
    disable_dynamo_repro_nvcc()

    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", type=str, default="data/packed")
    ap.add_argument("--tokenizer_json", type=str, default="out/tokenizer.json")
    ap.add_argument("--out_dir", type=str, default="out/model_final")
    ap.add_argument("--resume_dir", type=str, default="")

    # Model shape (keep modest for 16GB @ 4096 ctx)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)

    # Train
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=100)

    # Perf
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--torch_compile", action="store_true")
    ap.add_argument("--torch_compile_backend", type=str, default="inductor")
    ap.add_argument("--torch_compile_mode", type=str, default="default")
    ap.add_argument("--compile_fallback_eager", action="store_true", default=True)

    # Logging
    ap.add_argument("--tb_log", action="store_true")
    ap.add_argument("--tb_dir", type=str, default="out/tb_runs")
    ap.add_argument("--log_every", type=int, default=10, help="Log every N optimizer-steps")
    ap.add_argument("--gpu_stats_every", type=int, default=50, help="Poll temp/util/power every N optimizer-steps")
    ap.add_argument("--load_to_ram", action="store_true", default=True)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    if device == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    packed_dir = Path(args.packed_dir)
    train_npy, valid_npy = resolve_packed_paths(packed_dir)

    train_ds = PackedNpyDataset(train_npy, load_to_ram=args.load_to_ram)
    valid_ds = PackedNpyDataset(valid_npy, load_to_ram=args.load_to_ram)

    tok_path = Path(args.tokenizer_json)
    vocab_size = get_vocab_size_from_tokenizer_json(tok_path)

    bos_id, eos_id, pad_id = 0, 1, 0
    print(f"[INFO] Vocab={vocab_size} | BOS={bos_id} EOS={eos_id} PAD={pad_id} | Context={args.seq_len}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # Build / load model
    if args.resume_dir:
        print(f"[INFO] Resuming from: {args.resume_dir}")
        model = LlamaForCausalLM.from_pretrained(args.resume_dir)
    else:
        cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=args.n_embd,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=4 * args.n_embd,
            max_position_embeddings=args.seq_len,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            rope_theta=10000.0,
        )
        model = LlamaForCausalLM(cfg)

    # Try to set attention impl (HF supports different attribute names across versions)
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(model.config, attr):
            setattr(model.config, attr, args.attn_impl)

    if args.grad_ckpt:
        try:
            model.gradient_checkpointing_enable()
            # Required with HF + grad ckpt
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as e:
            print(f"[WARN] grad_ckpt requested but could not enable: {e}")

    model.to(device)
    model.train()

    raw_model = model

    # torch.compile (optional)
    if args.torch_compile and device == "cuda":
        print(f"[INFO] torch.compile enabled (backend={args.torch_compile_backend}, mode={args.torch_compile_mode})")
        try:
            model = torch.compile(raw_model, backend=args.torch_compile_backend, mode=args.torch_compile_mode)
        except Exception as e:
            print(f"[WARN] torch.compile failed at wrap-time: {e}")
            model = raw_model
            args.torch_compile = False

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader) // max(1, args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)

    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=min(args.warmup_steps, max(0, total_steps - 1)),
        num_training_steps=total_steps,
    )

    print(
        f"[INFO] AMP bf16={args.bf16} | batch={args.batch_size} grad_accum={args.grad_accum} total_steps={total_steps}"
    )
    tokens_per_opt_step = int(args.batch_size) * int(args.seq_len) * int(args.grad_accum)
    print(f"[INFO] Per-optimizer-step tokens ≈ {tokens_per_opt_step:,}")

    # Log dirs
    writer = None
    run_dir = None
    csv_f = None
    csv_w = None
    csv_path = None

    if args.tb_log:
        run_dir = make_run_dir(Path(args.tb_dir))
        writer = try_make_tb_writer(run_dir)
        csv_f, csv_w, csv_path = csv_logger_open(run_dir)
        if writer is not None:
            print(f"[INFO] TensorBoard logging -> {run_dir}")
        else:
            print("[WARN] TensorBoard SummaryWriter not available. (pip install tensorboard)")
            print(f"[INFO] CSV metrics -> {csv_path}")
    else:
        # still create a run dir for CSV if you want
        run_dir = make_run_dir(Path(args.tb_dir))
        csv_f, csv_w, csv_path = csv_logger_open(run_dir)
        print(f"[INFO] CSV metrics -> {csv_path}")

    gpu_tel = GpuTelemetry()
    if gpu_tel.mode != "none":
        print(f"[INFO] GPU telemetry -> {gpu_tel.mode} (temp/util/power)")

    global_opt_step = 0
    best_valid = float("inf")

    loss_hist: List[float] = []

    # training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        optim.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_micro_steps = 0

        # reset per-epoch cuda peak stats
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for micro_step, batch in enumerate(pbar, start=1):
            step_t0 = time.time()
            epoch_micro_steps += 1

            batch = batch.to(device, non_blocking=True).long()

            # Forward+loss
            try:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(args.bf16 and device == "cuda"),
                ):
                    out = model(input_ids=batch, labels=batch)
                    loss = out.loss / max(1, args.grad_accum)
            except Exception as e:
                # If compile causes issues, fall back to eager and continue
                if args.torch_compile and args.compile_fallback_eager:
                    msg = str(e)
                    print(f"\n[WARN] torch.compile/inductor error -> fallback to eager. ({type(e).__name__}: {msg})")
                    try:
                        import torch._dynamo as dynamo  # type: ignore
                        dynamo.reset()
                    except Exception:
                        pass
                    model = raw_model
                    args.torch_compile = False
                    model.train()
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=(args.bf16 and device == "cuda"),
                    ):
                        out = model(input_ids=batch, labels=batch)
                        loss = out.loss / max(1, args.grad_accum)
                else:
                    raise

            loss.backward()
            epoch_loss_sum += float(loss.item()) * max(1, args.grad_accum)

            # Optimizer step
            if micro_step % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_opt_step += 1

                # Metrics
                lr = float(sched.get_last_lr()[0])
                mean_loss = epoch_loss_sum / max(1, global_opt_step)  # rough global avg
                loss_hist.append(float(epoch_loss_sum / max(1, epoch_micro_steps)))
                if len(loss_hist) > 200:
                    loss_hist = loss_hist[-200:]

                step_time = time.time() - step_t0
                tok_s = tokens_per_opt_step / max(1e-9, step_time)

                mem = get_gpu_mem_stats()

                # Optional telemetry
                temp = util = pwr = None
                if args.gpu_stats_every > 0 and (global_opt_step % args.gpu_stats_every == 0):
                    tel = gpu_tel.read()
                    temp, util, pwr = tel.get("temp_c"), tel.get("util_pct"), tel.get("power_w")

                # Progress bar postfix
                pbar.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tok_s=f"{tok_s:,.0f}",
                    mem=f"{mem['alloc_gb']:.2f}/{mem['reserved_gb']:.2f}GB",
                )

                # Write logs
                if args.log_every > 0 and (global_opt_step % args.log_every == 0):
                    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    # TB
                    if writer is not None:
                        writer.add_scalar("train/loss", mean_loss, global_opt_step)
                        writer.add_scalar("train/lr", lr, global_opt_step)
                        writer.add_scalar("train/tokens_per_s", tok_s, global_opt_step)
                        writer.add_scalar("train/step_time_s", step_time, global_opt_step)
                        writer.add_scalar("gpu/mem_alloc_gb", mem["alloc_gb"], global_opt_step)
                        writer.add_scalar("gpu/mem_reserved_gb", mem["reserved_gb"], global_opt_step)
                        writer.add_scalar("gpu/mem_max_alloc_gb", mem["max_alloc_gb"], global_opt_step)
                        if temp is not None:
                            writer.add_scalar("gpu/temp_c", temp, global_opt_step)
                        if util is not None:
                            writer.add_scalar("gpu/util_pct", util, global_opt_step)
                        if pwr is not None:
                            writer.add_scalar("gpu/power_w", pwr, global_opt_step)

                    # CSV (always)
                    if csv_w is not None and csv_f is not None:
                        csv_w.writerow([
                            now,
                            epoch,
                            micro_step,
                            global_opt_step,
                            mean_loss,
                            lr,
                            tok_s,
                            step_time,
                            mem["alloc_gb"],
                            mem["reserved_gb"],
                            mem["max_alloc_gb"],
                            temp,
                            util,
                            pwr,
                        ])
                        csv_f.flush()

                    # Small terminal "graph"
                    if len(loss_hist) >= 5:
                        pbar.write(f"[live] loss spark: {sparkline(loss_hist, width=40)}")

        train_time = time.time() - t0

        # Validation (quick)
        model.eval()
        vlosses: List[float] = []
        with torch.no_grad():
            vbar = tqdm(valid_loader, desc=f"valid {epoch}", dynamic_ncols=True)
            for vb, vbatch in enumerate(vbar, start=1):
                vbatch = vbatch.to(device, non_blocking=True).long()
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(args.bf16 and device == "cuda"),
                ):
                    out = model(input_ids=vbatch, labels=vbatch)
                vlosses.append(float(out.loss.item()))
                if args.eval_batches > 0 and vb >= args.eval_batches:
                    break

        valid_loss = float(np.mean(vlosses)) if vlosses else float("inf")
        ppl = math.exp(valid_loss) if valid_loss < 20 else float("inf")
        model.train()

        print(f"[INFO] Epoch {epoch}: train_time={train_time:.1f}s | valid_loss={valid_loss:.4f} | ppl={ppl:.2f}")

        # Log validation
        if writer is not None:
            writer.add_scalar("valid/loss", valid_loss, global_opt_step)
            writer.add_scalar("valid/ppl", ppl, global_opt_step)

        if csv_w is not None and csv_f is not None:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            csv_w.writerow([now, epoch, "", global_opt_step, "", "", "", "", "", "", "", "", "", ""])
            csv_f.flush()

        # Save checkpoints
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {"epoch": epoch, "global_opt_step": global_opt_step, "valid_loss": valid_loss, "args": vars(args)}
        try_save_pretrained_else_fallback(raw_model if args.torch_compile else model, out_dir / f"ckpt_epoch_{epoch}", tok_path, state)

        if valid_loss < best_valid:
            best_valid = valid_loss
            try_save_pretrained_else_fallback(raw_model if args.torch_compile else model, out_dir / "model_best", tok_path, state)

    # Final save
    out_dir = Path(args.out_dir)
    try_save_pretrained_else_fallback(raw_model if args.torch_compile else model, out_dir / "model_final", tok_path, {"global_opt_step": global_opt_step})
    print(f"[DONE] Saved to: {out_dir.resolve()}")

    if writer is not None:
        writer.flush()
        writer.close()
    if csv_f is not None:
        csv_f.close()


if __name__ == "__main__":
    main()

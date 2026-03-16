#!/usr/bin/env python3
"""
generate_ensemble_llama_cascade_v1.py

What this script does (vs the previous generate_ensemble_v1.py):
  - Uses your *trained model* to generate each target track directly (no Python-only "fake" orchestration).
  - Generates tracks in a cascade so each later instrument can "hear" the earlier ones:

        ARP1 -> ARP2 -> CLARINET -> OBOE -> FLUTE -> ALTO FLUTE -> PICCOLO FLUTE

  - Still keeps your chord-track discipline:
      * 1/16 grid alignment (configurable)
      * If a note spans a chord change, it is split at the chord boundary
      * Optional: snap notes to chord tones at each onset
  - For Claire winds: injects keyswitches (simple heuristic based on note length)

Assumptions:
  - Your model is a HuggingFace causal LM checkpoint (e.g. LlamaForCausalLM) saved in --model_dir.
  - The model was trained on Miditok token IDs matching --tokenizer_json (same vocab).
  - Input MIDI contains a chord track (notes) + bass + piano; these are used as the prompt.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
import tempfile
import zlib
import time
import gc
import os
import sys
import shutil
import threading
import atexit
import subprocess
from collections import deque
from dataclasses import dataclass

# Python 3.12 dataclasses can crash if a custom loader executes this file
# without first registering the module in sys.modules. Register a placeholder
# entry here so the script also survives odd exec_module/loaders, while still
# running normally as a plain standalone script.
if __name__ not in sys.modules or sys.modules.get(__name__) is None:
    import types as _types
    _self_module = _types.ModuleType(__name__)
    _self_module.__dict__.update(globals())
    sys.modules[__name__] = _self_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import mido  # type: ignore
import torch  # type: ignore
from transformers import AutoModelForCausalLM  # type: ignore

import miditok  # type: ignore
try:
    from miditok import TokSequence  # type: ignore
except Exception:
    try:
        from miditok.classes import TokSequence  # type: ignore
    except Exception:
        TokSequence = None  # type: ignore
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


# ----------------------------
# Logging / tqdm
# ----------------------------

def setup_logging(level: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("ensemble_gen")
    logger.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    logger.handlers.clear()

    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)

    return logger


def cuda_mem_stats() -> str:
    """Small 'nerd stats' helper for CUDA memory."""
    if not torch.cuda.is_available():
        return "CPU"
    try:
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        return f"alloc={alloc:.0f}MB res={reserved:.0f}MB peak={peak:.0f}MB"
    except Exception:
        return "CUDA"

VRAM_GC_STATE: dict = {"last_gc_chunk": -10**9}

def maybe_cuda_gc(
    logger,
    *,
    threshold: float,
    cooldown_chunks: int,
    chunk_idx: int,
    reason: str,
    state: dict,
    force: bool = False,
) -> bool:
    """Best-effort VRAM garbage collection."""
    if force:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            state["last_gc_chunk"] = int(chunk_idx)
        except Exception:
            pass
        return True

    if threshold is None or threshold <= 0:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        _free, _total = torch.cuda.mem_get_info()
        _reserved = torch.cuda.memory_reserved()
        pressure = (float(_reserved) / float(_total)) if _total else 0.0
    except Exception:
        return False
    if pressure < float(threshold):
        return False
    last = int(state.get("last_gc_chunk", -10**9))
    if int(chunk_idx) - last < int(cooldown_chunks):
        return False
    import gc as _gc
    _gc.collect()
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        pass
    state["last_gc_chunk"] = int(chunk_idx)
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    if logger:
        logger.info(f"VRAM GC | reason={reason} | pressure={pressure:.3f} | {cuda_mem_stats()}")
    return True


def debug_ids_stats(ids: Sequence[int], vocab_size: Optional[int] = None) -> str:
    if not ids:
        return "ids=[]"
    ids_list = [int(x) for x in ids]
    mn = min(ids_list)
    mx = max(ids_list)
    if vocab_size is None or vocab_size <= 0:
        return f"len={len(ids_list)} min={mn} max={mx}"
    oob = sum(1 for x in ids_list if x < 0 or x >= int(vocab_size))
    return f"len={len(ids_list)} min={mn} max={mx} oob={oob}/{len(ids_list)}"




def _tqdm_iter(iterable, *, total=None, desc="", unit="it", enabled=True, leave=True, ncols=None, mininterval=0.2):
    if (not enabled):
        return iterable
    if _use_ansi_progress():
        total_i = int(total) if total is not None else None
        bar = _AnsiProgressBar(total=total_i, desc=desc, unit=unit, leave=leave, ncols=ncols, mininterval=mininterval)

        def _gen():
            try:
                for item in iterable:
                    yield item
                    bar.update(1)
            finally:
                bar.close()

        return _gen()
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, ncols=ncols, mininterval=mininterval)


_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_first_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    m = re.search(r"-?\d+(?:\.\d+)?", str(val))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _spark(values, width: int = 20) -> str:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return "·" * max(4, int(width))
    if len(vals) > width:
        vals = vals[-width:]
    if len(vals) < width:
        vals = ([vals[0]] * (width - len(vals))) + vals
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-9:
        idx = min(len(_SPARK_BLOCKS) - 1, 3)
        return _SPARK_BLOCKS[idx] * len(vals)
    out = []
    for v in vals:
        frac = (v - lo) / (hi - lo)
        idx = int(round(frac * (len(_SPARK_BLOCKS) - 1)))
        idx = max(0, min(len(_SPARK_BLOCKS) - 1, idx))
        out.append(_SPARK_BLOCKS[idx])
    return "".join(out)


class _AnsiProgressManager:
    def __init__(self):
        self.stream = sys.stdout
        self._bars = []
        self._prev_lines = 0
        self._last_render = 0.0
        self._hide_cursor = False
        self._lock = threading.RLock()
        self._t0 = time.perf_counter()
        self._gpu_last_poll = 0.0
        self._gpu = {}
        self._hist = {
            'tok_s': deque(maxlen=48),
            'gpu_util': deque(maxlen=48),
            'gpu_temp': deque(maxlen=48),
            'gpu_power': deque(maxlen=48),
            'gpu_fan': deque(maxlen=48),
            'gpu_mem_pct': deque(maxlen=48),
        }
        self._nvsmi = self._find_nvidia_smi()
        atexit.register(self.shutdown)

    def _find_nvidia_smi(self):
        cand = shutil.which('nvidia-smi')
        if cand:
            return cand
        for p in (
            '/usr/bin/nvidia-smi',
            '/usr/local/bin/nvidia-smi',
            '/usr/lib/wsl/lib/nvidia-smi',
            '/mnt/c/Windows/System32/nvidia-smi.exe',
        ):
            if os.path.exists(p):
                return p
        return None

    def register(self, bar):
        with self._lock:
            bar._created = time.perf_counter()
            bar._last_update = bar._created
            self._bars.append(bar)
            self.render(force=True)

    def unregister(self, bar):
        with self._lock:
            if bar in self._bars:
                self._bars.remove(bar)
            self.render(force=True)

    def shutdown(self):
        with self._lock:
            if self._prev_lines and self._use_ansi():
                self._move_up(self._prev_lines)
                self._clear_block(self._prev_lines)
                self._prev_lines = 0
            if self._hide_cursor and self._use_ansi():
                try:
                    self.stream.write("[?25h")
                    self.stream.flush()
                except Exception:
                    pass
                self._hide_cursor = False

    def _use_ansi(self):
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def _move_up(self, n: int):
        if n > 0:
            self.stream.write(f"[{n}F")

    def _clear_block(self, n: int):
        for i in range(n):
            self.stream.write("[2K")
            if i != n - 1:
                self.stream.write("\n")
        if n > 1:
            self._move_up(n - 1)

    def _poll_gpu(self):
        now = time.perf_counter()
        if now - self._gpu_last_poll < 0.8:
            return
        self._gpu_last_poll = now
        info = {}
        if self._nvsmi:
            try:
                out = subprocess.check_output(
                    [
                        self._nvsmi,
                        '--query-gpu=temperature.gpu,utilization.gpu,fan.speed,power.draw,memory.used,memory.total,name',
                        '--format=csv,noheader,nounits',
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=1.5,
                ).strip().splitlines()
                if out:
                    parts = [p.strip() for p in out[0].split(',')]
                    if len(parts) >= 7:
                        temp, util, fan, power, mem_used, mem_total, name = parts[:7]
                        info = {
                            'name': name,
                            'temp': _parse_first_float(temp),
                            'util': _parse_first_float(util),
                            'fan': _parse_first_float(fan),
                            'power': _parse_first_float(power),
                            'mem_used': _parse_first_float(mem_used),
                            'mem_total': _parse_first_float(mem_total),
                        }
            except Exception:
                info = {}
        if torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                used_b = total_b - free_b
                info.setdefault('mem_used', used_b / (1024 ** 2))
                info.setdefault('mem_total', total_b / (1024 ** 2))
                info.setdefault('name', torch.cuda.get_device_name(0))
            except Exception:
                pass
        if info:
            mu = info.get('mem_used')
            mt = info.get('mem_total')
            mem_pct = None
            if mu is not None and mt not in (None, 0):
                mem_pct = 100.0 * float(mu) / float(mt)
            info['mem_pct'] = mem_pct
            self._gpu = info
            self._hist['gpu_util'].append(info.get('util'))
            self._hist['gpu_temp'].append(info.get('temp'))
            self._hist['gpu_power'].append(info.get('power'))
            self._hist['gpu_fan'].append(info.get('fan'))
            self._hist['gpu_mem_pct'].append(mem_pct)

    def _selected_bars(self, bars):
        bars = [b for b in bars if not getattr(b, 'closed', False)]
        if not bars:
            return []
        bars_sorted = sorted(bars, key=lambda b: getattr(b, '_last_update', 0.0), reverse=True)
        stage_bar = next((b for b in bars_sorted if 'bar' in str(getattr(b, 'unit', '')).lower()), None)
        samp_bar = next((b for b in bars_sorted if ('samp' in str(getattr(b, 'unit', '')).lower()) or ('samp' in str(getattr(b, 'desc', '')).lower()) or ('sampling' in str(getattr(b, 'desc', '')).lower())), None)
        other_bar = next((b for b in bars_sorted if b not in (stage_bar, samp_bar)), None)
        sel = []
        for b in (stage_bar, samp_bar, other_bar):
            if b is not None and b not in sel:
                sel.append(b)
        return sel[:3]

    def render(self, force: bool = False):
        with self._lock:
            now = time.perf_counter()
            if (not force) and (now - self._last_render < 0.08):
                return
            bars = [b for b in self._bars if not getattr(b, 'closed', False)]
            if not bars:
                if self._prev_lines:
                    if self._use_ansi():
                        self._move_up(self._prev_lines)
                        self._clear_block(self._prev_lines)
                    self._prev_lines = 0
                if self._hide_cursor and self._use_ansi():
                    self.stream.write("[?25h")
                    self._hide_cursor = False
                    self.stream.flush()
                self._last_render = now
                return

            self._poll_gpu()
            for b in bars:
                tok_s = None
                pf = getattr(b, 'postfix', {}) or {}
                if isinstance(pf, dict) and 'tok/s' in pf:
                    tok_s = _parse_first_float(pf.get('tok/s'))
                self._hist['tok_s'].append(tok_s)

            cols = shutil.get_terminal_size((160, 40)).columns
            cols = max(100, cols)
            lines = self._build_lines(cols, bars)

            if self._use_ansi() and not self._hide_cursor:
                self.stream.write("[?25l")
                self._hide_cursor = True

            if self._prev_lines:
                if self._use_ansi():
                    self._move_up(self._prev_lines)
                    self._clear_block(self._prev_lines)
                else:
                    self.stream.write("\n")

            for idx, line in enumerate(lines):
                self.stream.write(line)
                if idx != len(lines) - 1:
                    self.stream.write("\n")
            self.stream.flush()
            self._prev_lines = len(lines)
            self._last_render = now

    def _build_lines(self, cols: int, bars):
        sel = self._selected_bars(bars)
        stage_bar = sel[0] if sel else None
        samp_bar = sel[1] if len(sel) > 1 else None
        total_elapsed = _fmt_hms(time.perf_counter() - self._t0)
        title = _ansi('TTY DASHBOARD', '1;36')
        if stage_bar is not None:
            stage_name = _clip_text(str(getattr(stage_bar, 'desc', 'idle')), max(12, cols // 4))
            stage_n = max(0, int(getattr(stage_bar, 'n', 0)))
            stage_total = max(1, int(getattr(stage_bar, 'total', 1) or 1))
            stage_elapsed_s = max(0.0, time.perf_counter() - getattr(stage_bar, '_created', time.perf_counter()))
            stage_rate = (float(stage_n) / stage_elapsed_s) if stage_elapsed_s > 1e-6 and stage_n > 0 else 0.0
            stage_eta_s = ((stage_total - stage_n) / stage_rate) if stage_rate > 1e-9 and stage_total > stage_n else 0.0
            head = (
                f"{title} {_ansi('│', '2;37')} total {_ansi(total_elapsed, '1;33')} "
                f"{_ansi('│', '2;37')} stage {_ansi(stage_name, '1;37')} "
                f"{_ansi('│', '2;37')} stage elapsed {_ansi(_fmt_hms(stage_elapsed_s), '1;32')} "
                f"{_ansi('│', '2;37')} stage eta {_ansi(_fmt_hms(stage_eta_s), '1;35')}"
            )
        else:
            head = f"{title} {_ansi('│', '2;37')} total {_ansi(total_elapsed, '1;33')}"
        lines = [head[:cols]]
        if stage_bar is not None:
            lines.append(self._format_bar_line(stage_bar, cols, label='bars'))
        if samp_bar is not None:
            lines.append(self._format_bar_line(samp_bar, cols, label='sampling'))
        other_bar = sel[2] if len(sel) > 2 else None
        if other_bar is not None:
            lines.append(self._format_bar_line(other_bar, cols, label='other'))
        lines.extend(self._gpu_lines(cols))
        return lines

    def _format_bar_line(self, bar, cols: int, label: Optional[str] = None) -> str:
        total = max(1, int(bar.total) if bar.total else 1)
        n = max(0, int(bar.n))
        frac = max(0.0, min(1.0, float(n) / float(total)))
        elapsed_s = max(0.0, time.perf_counter() - getattr(bar, '_created', time.perf_counter()))
        rate = (float(n) / elapsed_s) if elapsed_s > 1e-6 and n > 0 else 0.0
        eta_s = ((total - n) / rate) if rate > 1e-9 and total > n else 0.0
        pct_txt = f"{frac*100:5.1f}%"
        count_txt = f"{n}/{total} {str(getattr(bar, 'unit', 'it'))}"
        timing_txt = f"elapsed={_fmt_hms(elapsed_s)} eta={_fmt_hms(eta_s)}"
        pf = _clip_text(bar._postfix_text(), max(10, cols // 3))
        name = _clip_text(str(getattr(bar, 'desc', 'progress')), 26)
        prefix_parts = []
        if label:
            prefix_parts.append(_ansi(f"{label:>8}", '1;36'))
        prefix_parts.append(_ansi(name, '1;37'))
        prefix_parts.append(_ansi(pct_txt, '1;35'))
        prefix_parts.append(_ansi(count_txt, '1;33'))
        prefix_parts.append(_ansi(timing_txt, '2;37'))
        if pf:
            prefix_parts.append(_ansi(pf, '2;37'))
        prefix = f" {_ansi('│', '2;37')} ".join(prefix_parts)
        used = _visual_len(prefix) + 3
        bar_w = max(18, cols - used)
        filled = int(round(bar_w * frac))
        bar_graph = _ansi('█' * filled, '1;32') + _ansi('░' * max(0, bar_w - filled), '2;32')
        return f"{prefix} {_ansi('│', '2;37')} {bar_graph}"

    def _gpu_lines(self, cols: int):
        spark_w = max(10, min(18, cols // 14))
        tok_now = None
        if self._hist['tok_s']:
            tok_now = self._hist['tok_s'][-1]
        tok_now_txt = '--' if tok_now is None else str(int(tok_now))
        gpu = dict(self._gpu)
        if not gpu:
            mem = cuda_mem_stats()
            line = (
                f"{_ansi('gpu', '1;36')} {_ansi('no live sensor access', '2;37')} {_ansi('│', '2;37')} "
                f"{_ansi(mem, '2;37')} {_ansi('│', '2;37')} "
                f"tok/s {_spark(self._hist['tok_s'], width=spark_w)} {_ansi(tok_now_txt, '1;33')}"
            )
            return [line[:cols]]
        name = _clip_text(str(gpu.get('name', 'GPU')), 18)
        util = gpu.get('util')
        temp = gpu.get('temp')
        power = gpu.get('power')
        fan = gpu.get('fan')
        mem_pct = gpu.get('mem_pct')
        mem_used = gpu.get('mem_used')
        mem_total = gpu.get('mem_total')
        mem_txt = '--/-- MB'
        if mem_used is not None and mem_total is not None:
            mem_txt = f"{int(mem_used):>5}/{int(mem_total):<5}MB"
        line1 = (
            f"{_ansi('gpu', '1;36')} {_ansi(name, '1;37')} {_ansi('│', '2;37')} "
            f"util {str(int(util)) if util is not None else '--':>3}% {_spark(self._hist['gpu_util'], spark_w)} {_ansi('│', '2;37')} "
            f"temp {str(int(temp)) if temp is not None else '--':>3}C {_spark(self._hist['gpu_temp'], spark_w)} {_ansi('│', '2;37')} "
            f"power {str(int(power)) if power is not None else '--':>4}W {_spark(self._hist['gpu_power'], spark_w)}"
        )
        line2 = (
            f"{_ansi('gpu', '1;36')} fan {str(int(fan)) if fan is not None else '--':>3}% {_spark(self._hist['gpu_fan'], spark_w)} {_ansi('│', '2;37')} "
            f"vram {str(int(mem_pct)) if mem_pct is not None else '--':>3}% {_spark(self._hist['gpu_mem_pct'], spark_w)} {_ansi('│', '2;37')} "
            f"mem {mem_txt} {_ansi('│', '2;37')} tok/s {_spark(self._hist['tok_s'], spark_w)} {_ansi(tok_now_txt, '1;33')}"
        )
        return [line1[:cols], line2[:cols]]


class _AnsiProgressBar:
    def __init__(self, *, total, desc="", unit="it", leave=False, ncols=None, mininterval=0.2):
        self.total = int(total) if total is not None else None
        self.desc = str(desc or "")
        self.unit = str(unit or "it")
        self.leave = bool(leave)
        self.ncols = ncols
        self.mininterval = float(mininterval)
        self.n = 0
        self.closed = False
        self.postfix = {}
        self._created = time.perf_counter()
        self._last_update = self._created
        _ANSI_PROGRESS.register(self)

    def update(self, n=1):
        self.n += int(n)
        self._last_update = time.perf_counter()
        _ANSI_PROGRESS.render(force=False)

    def set_postfix(self, data=None, **kwargs):
        if data is None:
            data = {}
        if isinstance(data, dict):
            self.postfix.update(data)
        else:
            self.postfix[str(data)] = ""
        if kwargs:
            self.postfix.update(kwargs)
        self._last_update = time.perf_counter()
        _ANSI_PROGRESS.render(force=False)

    def _postfix_text(self):
        if not self.postfix:
            return ""
        parts = []
        for k, v in self.postfix.items():
            if v == "":
                parts.append(str(k))
            else:
                parts.append(f"{k}={v}")
        return " | ".join(parts)

    def close(self):
        if self.closed:
            return
        self.closed = True
        _ANSI_PROGRESS.unregister(self)


def _visual_len(text: str) -> int:
    return len(re.sub(r"\[[0-9;]*m", "", str(text)))


def _clip_text(text: str, width: int) -> str:
    text = str(text)
    if width <= 3:
        return text[:width]
    if len(text) <= width:
        return text
    return text[: max(1, width - 1)] + "…"


def _use_ansi_progress() -> bool:
    if os.environ.get("NO_TTY_PROGRESS", "").strip() not in ("", "0", "false", "False"):
        return False
    term = os.environ.get("TERM", "")
    if term.lower() == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _ansi(text: str, code: str) -> str:
    if not _use_ansi_progress():
        return str(text)
    return f"[{code}m{text}[0m"


_ANSI_PROGRESS = _AnsiProgressManager()


def _tqdm_bar(*, total, desc="", unit="it", enabled=True, leave=False, ncols=None, mininterval=0.2):
    if (not enabled):
        return None
    if _use_ansi_progress():
        return _AnsiProgressBar(total=total, desc=desc, unit=unit, leave=leave, ncols=ncols, mininterval=mininterval)
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit, leave=leave, ncols=ncols, mininterval=mininterval)




# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class NoteEvent:
    start: int
    end: int
    pitch: int
    vel: int


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    program: int
    channel: int
    lo: int
    hi: int
    is_claire: bool = False
    seed_style: str = "chord"  # chord | arp_up | arp_down | sustain


# ----------------------------
# MIDI utilities
# ----------------------------

def _track_name(inst: Instrument) -> str:
    return (inst.name or "").strip()


def _first_time_signature(src: MidiFile) -> Tuple[int, int]:
    tss = list(getattr(src, "time_signature_changes", []))
    if not tss:
        return (4, 4)
    ts0 = sorted(tss, key=lambda x: x.time)[0]
    return (int(ts0.numerator), int(ts0.denominator))


def bar_ticks(tpb: int, ts: Tuple[int, int]) -> int:
    num, den = ts
    return int(round(tpb * num * (4.0 / den)))


def quantize_tick(t: int, grid_ticks: int) -> int:
    return int(round(t / max(1, grid_ticks)) * max(1, grid_ticks))


def is_probable_keyswitch(n: Note, tpb: int, low_pitch_hard: int = 11) -> bool:
    if n.pitch <= low_pitch_hard:
        return True
    dur = int(n.end - n.start)
    if n.pitch < 36 and dur <= max(1, tpb // 16):
        return True
    return False


def remove_keyswitches_from_instrument(inst: Instrument, tpb: int) -> Instrument:
    new_inst = Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
    new_inst.notes = [n for n in inst.notes if not is_probable_keyswitch(n, tpb)]
    new_inst.control_changes = list(getattr(inst, "control_changes", []))
    new_inst.pitch_bends = list(getattr(inst, "pitch_bends", []))
    return new_inst


def build_prompt_midi(
    src: MidiFile,
    logger: logging.Logger,
    disable_keyswitches: bool = True,
    chord_name_hints: Tuple[str, ...] = ("CHORD", "AKKORD", "CHORDS"),
    piano_name_hints: Tuple[str, ...] = ("PIANO", "KEYSCAPE", "KEY"),
    bass_name_hints: Tuple[str, ...] = ("EZBASS", "BASS", "DARKWALL"),
) -> MidiFile:
    """Return a new MidiFile containing only chord + (cleaned) bass + piano."""
    tpb = int(src.ticks_per_beat)

    def _match(name: str, hints: Tuple[str, ...]) -> bool:
        u = name.upper()
        return any(h in u for h in hints)

    out = MidiFile(ticks_per_beat=tpb)
    out.tempo_changes = list(getattr(src, "tempo_changes", []))
    out.time_signature_changes = list(getattr(src, "time_signature_changes", []))
    out.key_signature_changes = list(getattr(src, "key_signature_changes", []))

    kept: List[Instrument] = []

    for inst in src.instruments:
        nm = _track_name(inst) or f"Program{inst.program}" + ("_drum" if inst.is_drum else "")
        if _match(nm, chord_name_hints):
            kept.append(inst)
            logger.info(f"Prompt: keep chord track: {nm}")
            continue
        if _match(nm, piano_name_hints):
            kept.append(inst)
            logger.info(f"Prompt: keep piano track: {nm}")
            continue
        if _match(nm, bass_name_hints):
            if disable_keyswitches:
                cleaned = remove_keyswitches_from_instrument(inst, tpb)
                removed = len(inst.notes) - len(cleaned.notes)
                kept.append(cleaned)
                logger.info(f"Prompt: keep bass track: {nm} (removed keyswitch notes={removed})")
            else:
                kept.append(inst)
                logger.info(f"Prompt: keep bass track: {nm} (keyswitch removal disabled)")
            continue

    if not kept:
        logger.warning("No tracks matched hints; falling back to all non-drum tracks")
        for inst in src.instruments:
            if not inst.is_drum:
                kept.append(inst)

    out.instruments = kept
    return out


def filter_prompt_instruments(
    mid: MidiFile,
    *,
    prompt_mode: str,
    include_generated: bool,
    max_bass_tracks: int,
    max_piano_tracks: int,
    exclude_regex: Optional[str] = None,
) -> MidiFile:
    """
    Keep a compact, useful prompt to avoid hitting model context limits.
    prompt_mode: chords_only | chords_bass | chords_bass_piano
    include_generated: keep previously generated ARGENT_/CLAIRE_ tracks in the prompt (recommended for cascade)
    """
    pm = (prompt_mode or "chords_bass").strip().lower()
    if pm not in ("chords_only", "chords_bass", "chords_bass_piano"):
        pm = "chords_bass"

    rx = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None

    def is_chord(name: str) -> bool:
        u = name.upper()
        return ("CHORD" in u) or ("AKKORD" in u)

    def is_bass(name: str) -> bool:
        u = name.upper()
        return ("BASS" in u) or ("EZBASS" in u) or ("DARKWALL" in u)

    def is_piano(name: str) -> bool:
        u = name.upper()
        return ("PIANO" in u) or ("KEYSCAPE" in u) or ("KEY " in u) or ("KEYS" in u)

    def is_generated(name: str) -> bool:
        u = name.upper()
        return u.startswith("ARGENT_") or u.startswith("CLAIRE_")

    def is_seed(name: str) -> bool:
        return name.upper().endswith("_SEED")

    chords: List[Instrument] = []
    bass: List[Instrument] = []
    piano: List[Instrument] = []
    generated: List[Instrument] = []
    seed: List[Instrument] = []

    for inst in mid.instruments:
        nm = _track_name(inst) or f"Program{inst.program}"
        if rx and rx.search(nm):
            continue
        if is_seed(nm):
            seed.append(inst); continue
        if is_chord(nm):
            chords.append(inst); continue
        if include_generated and is_generated(nm):
            generated.append(inst); continue
        if pm in ("chords_bass", "chords_bass_piano") and is_bass(nm):
            bass.append(inst); continue
        if pm == "chords_bass_piano" and is_piano(nm):
            piano.append(inst); continue

    def take_largest(insts: List[Instrument], k: int) -> List[Instrument]:
        if k <= 0:
            return []
        # Sorter i revers (flest noter først)
        return sorted(insts, key=lambda x: len(getattr(x, "notes", [])), reverse=True)[:k]

    bass_keep = take_largest(bass, int(max(0, max_bass_tracks)))
    piano_keep = take_largest(piano, int(max(0, max_piano_tracks)))

    out = MidiFile(ticks_per_beat=int(mid.ticks_per_beat))
    out.tempo_changes = list(getattr(mid, "tempo_changes", []))
    out.time_signature_changes = list(getattr(mid, "time_signature_changes", []))
    out.key_signature_changes = list(getattr(mid, "key_signature_changes", []))
    out.instruments = []
    out.instruments.extend(chords)
    out.instruments.extend(bass_keep)
    out.instruments.extend(piano_keep)
    out.instruments.extend(generated)
    out.instruments.extend(seed)
    return out

def trim_prompt_to_cutoff(
    mid: MidiFile,
    *,
    cutoff_tick: int,
    chord_track_regex: str = r"(CHORD|AKKORD)",
) -> MidiFile:
    """
    Keep prompt events strictly BEFORE the target bar, but allow chord track to include the chord at the bar start (start==cutoff).
    Seed tracks (name ends with _SEED) are also allowed at the boundary.
    """
    cut = int(max(0, cutoff_tick))
    rx = re.compile(chord_track_regex, re.IGNORECASE)

    out = MidiFile(ticks_per_beat=int(mid.ticks_per_beat))
    out.tempo_changes = list(getattr(mid, "tempo_changes", []))
    out.time_signature_changes = list(getattr(mid, "time_signature_changes", []))
    out.key_signature_changes = list(getattr(mid, "key_signature_changes", []))
    out.instruments = []

    for inst in mid.instruments:
        nm = _track_name(inst) or f"Program{inst.program}"
        is_seed = nm.upper().endswith("_SEED")
        is_chord = bool(rx.search(nm))
        ni = Instrument(program=int(inst.program), is_drum=bool(inst.is_drum), name=str(inst.name))
        ni.control_changes = []
        ni.pitch_bends = []
        for n in inst.notes:
            s = int(n.start); e = int(n.end)
            if is_chord or is_seed:
                if s <= cut:
                    ni.notes.append(Note(start=s, end=e, pitch=int(n.pitch), velocity=int(n.velocity)))
            else:
                if s < cut:
                    ni.notes.append(Note(start=s, end=e, pitch=int(n.pitch), velocity=int(n.velocity)))
        for cc in getattr(inst, "control_changes", []):
            t = int(getattr(cc, "time", getattr(cc, "tick", 0)))
            if t < cut:
                ni.control_changes.append(type(cc)(time=t, number=int(cc.number), value=int(cc.value)))
        for pb in getattr(inst, "pitch_bends", []):
            t = int(getattr(pb, "time", getattr(pb, "tick", 0)))
            if t < cut:
                ni.pitch_bends.append(type(pb)(time=t, pitch=int(pb.pitch)))
        if ni.notes or ni.control_changes or ni.pitch_bends:
            out.instruments.append(ni)

    return out


def slice_miditoolkit_midi(mid: MidiFile, start_tick: int, end_tick: int, shift_to_zero: bool = True) -> MidiFile:
    """Slice MidiFile to [start_tick, end_tick) and optionally shift times so start_tick=0."""
    tpb = int(mid.ticks_per_beat)
    out = MidiFile(ticks_per_beat=tpb)

    offset = start_tick if shift_to_zero else 0

    out.tempo_changes = []
    out.time_signature_changes = []
    out.key_signature_changes = list(getattr(mid, "key_signature_changes", []))

    # tempos
    tempos = list(getattr(mid, "tempo_changes", []))
    if tempos:
        tempos_sorted = sorted(tempos, key=lambda x: x.time)
        prev = None
        for t in tempos_sorted:
            if t.time <= start_tick:
                prev = t
            elif start_tick < t.time < end_tick:
                out.tempo_changes.append(TempoChange(tempo=t.tempo, time=int(t.time - offset)))
        if prev is not None:
            out.tempo_changes.insert(0, TempoChange(tempo=prev.tempo, time=0))

    # time signatures
    tss = list(getattr(mid, "time_signature_changes", []))
    if tss:
        tss_sorted = sorted(tss, key=lambda x: x.time)
        prev = None
        for ts in tss_sorted:
            if ts.time <= start_tick:
                prev = ts
            elif start_tick < ts.time < end_tick:
                out.time_signature_changes.append(TimeSignature(numerator=ts.numerator, denominator=ts.denominator, time=int(ts.time - offset)))
        if prev is not None:
            out.time_signature_changes.insert(0, TimeSignature(numerator=prev.numerator, denominator=prev.denominator, time=0))

    out.instruments = []
    for inst in mid.instruments:
        ni = Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        ni.control_changes = []
        ni.pitch_bends = []

        for n in inst.notes:
            s = int(n.start)
            e = int(n.end)
            if e <= start_tick or s >= end_tick:
                continue
            s2 = max(start_tick, s) - offset
            e2 = min(end_tick, e) - offset
            if e2 <= s2:
                continue
            ni.notes.append(Note(start=int(s2), end=int(e2), pitch=int(n.pitch), velocity=int(n.velocity)))

        # CC + PB
        for cc in getattr(inst, "control_changes", []):
            t = int(getattr(cc, "time", getattr(cc, "tick", 0)))
            if start_tick <= t < end_tick:
                cc2 = type(cc)(time=int(t - offset), number=int(cc.number), value=int(cc.value))
                ni.control_changes.append(cc2)
        for pb in getattr(inst, "pitch_bends", []):
            t = int(getattr(pb, "time", getattr(pb, "tick", 0)))
            if start_tick <= t < end_tick:
                pb2 = type(pb)(time=int(t - offset), pitch=int(pb.pitch))
                ni.pitch_bends.append(pb2)

        if ni.notes or ni.control_changes or ni.pitch_bends:
            out.instruments.append(ni)

    return out


def extract_chord_tones_from_prompt(mid: MidiFile) -> List[Tuple[int, List[int]]]:
    chord_inst = None
    for inst in mid.instruments:
        nm = _track_name(inst).upper()
        if "CHORD" in nm or "AKKORD" in nm:
            chord_inst = inst
            break
    if chord_inst is None:
        chord_inst = max(mid.instruments, key=lambda i: len(i.notes)) if mid.instruments else None
    if chord_inst is None or not chord_inst.notes:
        return []
    starts: Dict[int, List[int]] = {}
    for n in chord_inst.notes:
        starts.setdefault(int(n.start), []).append(int(n.pitch))
    out: List[Tuple[int, List[int]]] = []
    for t in sorted(starts.keys()):
        out.append((t, sorted(set(starts[t]))))
    return out


def find_chord_at_time(chords: List[Tuple[int, List[int]]], t: int) -> List[int]:
    if not chords:
        return []
    lo, hi = 0, len(chords) - 1
    best = chords[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        ct, cp = chords[mid]
        if ct <= t:
            best = cp
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ----------------------------
# Melody extraction (optional constraints)
# ----------------------------

def extract_melody_notes(src: MidiFile, melody_track_regex: Optional[str]) -> List[NoteEvent]:
    if not melody_track_regex:
        return []
    rx = re.compile(melody_track_regex, re.IGNORECASE)
    out: List[NoteEvent] = []
    for inst in src.instruments:
        nm = _track_name(inst)
        if not nm:
            continue
        if not rx.search(nm):
            continue
        for n in inst.notes:
            out.append(NoteEvent(start=int(n.start), end=int(n.end), pitch=int(n.pitch), vel=int(n.velocity)))
    return sorted(out, key=lambda x: (x.start, x.end, x.pitch))


def melody_active_at(melody: List[NoteEvent], t: int) -> Optional[NoteEvent]:
    # Melody is usually small; linear scan is ok. If you want faster, build an index.
    for m in melody:
        if m.start <= t < m.end:
            return m
    return None


# ----------------------------
# Tokenizer / model
# ----------------------------

def load_tokenizer(tokenizer_json: Path):
    tok = miditok.REMI(params=str(tokenizer_json))
    # try to keep the model's "program separation" tokens
    try:
        tok.one_token_stream = True
        tok.one_token_stream_for_programs = True
    except Exception:
        pass
    return tok


def encode_midi_to_ids(tok, midi_path: Path) -> List[int]:
    # miditok API varies by version; this is the most compatible path.
    seqs = tok(str(midi_path))
    seq = seqs[0] if isinstance(seqs, list) else seqs
    return list(getattr(seq, "ids", seq))


def _decode_ids_to_score(tok, ids: Sequence[int]):
    """
    Robust decoder for miditok>=3.
    Tries multiple call patterns because API differs across versions / backends.
    Returns a Score/Midi-like object or None.
    """
    ids_list = [int(x) for x in ids]

    # Build TokSequence if available
    seq = None
    if TokSequence is not None:
        try:
            seq = TokSequence(ids=ids_list)  # type: ignore
        except Exception:
            seq = None

    # 1) decode path (preferred on newer MidiTok versions)
    if hasattr(tok, "decode"):
        for obj in (seq, [seq] if seq is not None else None, ids_list, [ids_list]):
            if obj is None:
                continue
            try:
                out = tok.decode(obj)  # type: ignore
                if out is not None:
                    return out
            except Exception:
                pass

    # 2) legacy tokens_to_midi / tokens_to_score path
    if hasattr(tok, "tokens_to_midi"):
        for obj in (seq, [seq] if seq is not None else None, ids_list, [ids_list]):
            if obj is None:
                continue
            try:
                out = tok_decode_compat(tok, obj)  # type: ignore
                if out is not None:
                    return out
            except Exception:
                pass

    # 3) __call__ decode is not a thing; nothing else to try
    return None


def decode_ids_to_notes_by_program(tok, ids, program, lo: int | None = None, hi: int | None = None):
    """
    Decode token IDs and return notes for the requested MIDI program.

    Robustness:
      - If the decoded score has no track explicitly marked with `program`,
        we fall back to selecting the "best" non-drum track by:
          1) program match bonus (if any),
          2) in-range ratio (if lo/hi provided),
          3) note count.
    This prevents the common failure mode where the model emits musical content
    but forgets/omits the program token, yielding 0 notes for the target.
    """
    score = _decode_ids_to_score(tok, ids)
    # Collect candidate tracks (symusic.Score uses .tracks; MidiFile backend uses .instruments)
    tracks = []
    if hasattr(score, "tracks"):
        tracks = list(score.tracks)
    elif hasattr(score, "instruments"):
        tracks = list(score.instruments)
    best_notes = []
    best_s = -1e18

    def _track_program(tr):
        p = getattr(tr, "program", None)
        try:
            return int(p) if p is not None else None
        except Exception:
            return None

    for tr in tracks:
        # Skip drums if possible
        if getattr(tr, "is_drum", False):
            continue

        notes = []
        for n in getattr(tr, "notes", []):
            # symusic.Note has time/duration; miditoolkit.Note has start/end
            if hasattr(n, "time"):
                st = int(n.time)
                en = int(n.time + n.duration)
                pitch = int(n.pitch)
                vel = int(getattr(n, "velocity", 100))
            else:
                st = int(n.start)
                en = int(n.end)
                pitch = int(n.pitch)
                vel = int(getattr(n, "velocity", 100))
            if en <= st:
                en = st + 1
            notes.append(NoteEvent(pitch=pitch, start=st, end=en, vel=vel))

        if not notes:
            continue

        tr_prog = _track_program(tr)
        match = (tr_prog == int(program))

        inr = 0.0
        if lo is not None and hi is not None and hi >= lo:
            inr = sum(1 for ev in notes if lo <= ev.pitch <= hi) / max(1, len(notes))

        # Scoring: big bonus for explicit program match, then prefer in-range + density
        s = 0.0
        if match:
            s += 1e6
        s += len(notes)
        s += inr * 1000.0

        # Tiny preference for named tracks that look like our target
        name = (getattr(tr, "name", "") or "").lower()
        if name and (str(program) in name or "argent" in name or "claire" in name):
            s += 0.5

        if s > best_s:
            best_s = s
            best_notes = notes

    return best_notes


def decode_ids_to_notes(tok, ids: Sequence[int]) -> List[NoteEvent]:
    """Decode notes from ALL tracks / programs (fallback when program-specific decode yields nothing)."""
    score = _decode_ids_to_score(tok, ids)
    if score is None:
        return []
    tracks = getattr(score, "tracks", None) or getattr(score, "instruments", None) or getattr(score, "parts", None)
    if not tracks:
        return []
    out: List[NoteEvent] = []
    for tr in tracks:
        if getattr(tr, "is_drum", False):
            continue
        tr_notes = getattr(tr, "notes", None)
        if tr_notes is None:
            getter = getattr(tr, "get_notes", None)
            tr_notes = getter() if callable(getter) else None
        if not tr_notes:
            continue
        for n in tr_notes:
            try:
                start = int(getattr(n, "time", getattr(n, "start", 0)))
                dur = getattr(n, "duration", getattr(n, "dur", None))
                end = start + int(dur) if dur is not None else int(getattr(n, "end", start + 1))
                pitch = int(getattr(n, "pitch", getattr(n, "note", getattr(n, "key", 60))))
                vel = int(getattr(n, "velocity", getattr(n, "vel", 90)))
                if end <= start:
                    end = start + 1
                out.append(NoteEvent(start=start, end=end, pitch=pitch, vel=vel))
            except Exception:
                continue
    return sorted(out, key=lambda x: (x.start, x.end, x.pitch))


def subtract_prompt_notes(gen: List[NoteEvent], prompt: List[NoteEvent], tol: int) -> List[NoteEvent]:
    """Remove notes from gen that look like they were copied from the prompt."""
    if not gen or not prompt:
        return gen
    tol = max(1, int(tol))

    # Hold track of which prompt notes have been "consumed" to prevent deleting valid generated overlaps
    prompt_by_pitch = {}
    for p in prompt:
        prompt_by_pitch.setdefault(int(p.pitch), []).append({"note": p, "used": False})

    out = []
    for gn in gen:
        cands = prompt_by_pitch.get(int(gn.pitch), [])
        is_prompt = False
        for cand in cands:
            if not cand["used"]:
                pn = cand["note"]
                if abs(gn.start - pn.start) <= tol and abs(gn.end - pn.end) <= tol:
                    is_prompt = True
                    cand["used"] = True  # Consume the prompt note!
                    break
        if not is_prompt:
            out.append(gn)
    return out


def _dtype_from_arg(dtype: str) -> torch.dtype:
    d = (dtype or "fp16").lower()
    if d == "bf16":
        return torch.bfloat16
    if d == "fp32":
        return torch.float32
    return torch.float16


def get_model_max_pos(model) -> int:
    cfg = getattr(model, "config", None)
    for k in ("max_position_embeddings", "max_seq_len", "seq_length", "n_positions"):
        v = getattr(cfg, k, None) if cfg is not None else None
        if v is not None:
            try:
                vi = int(v)
                if vi > 0:
                    return vi
            except Exception:
                pass
    return 4096


def load_model(
    model_dir: Path, 
    device: str, 
    dtype: str, 
    attn_impl: str, 
    logger: logging.Logger,
    gpu_vram_limit_gb: Optional[float] = None,
    cuda_compile: bool = False,
    quantization: str = "none"
):
    """Load HF causal LM with settings optimized for long-context generation on RTX class GPUs."""
    model_dtype = _dtype_from_arg(dtype)

    # GPU perf toggles (safe no-ops on CPU)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    kwargs = {
        "dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }

    # Håndter kvantisering (INT8 / INT4) via bitsandbytes
    if quantization in ("int8", "int4"):
        try:
            from transformers import BitsAndBytesConfig
            if quantization == "int8":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Aktiverer 8-bit (INT8) kvantisering for ekstrem hastighet/minne-besparelse!")
            elif quantization == "int4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Aktiverer 4-bit (INT4) kvantisering (maksimal hastighet/minne)!")
        except ImportError:
            logger.error("bitsandbytes er ikke installert! Kjør 'pip install bitsandbytes'. Faller tilbake til standard.")

    # attn impl: "auto" tries flash_attention_2 -> sdpa
    attn_impl = (attn_impl or "auto").strip().lower()
    if attn_impl != "auto":
        kwargs["attn_implementation"] = attn_impl
    else:
        for candidate in ("flash_attention_2", "sdpa"):
            try:
                kwargs["attn_implementation"] = candidate
                break
            except Exception:
                continue

    # HÅNDTERING AV SHARED MEMORY / VRAM LIMIT
    use_device_map = False
    if device == "cuda" and torch.cuda.is_available():
        # BitsAndBytes krever ofte device_map
        if (gpu_vram_limit_gb is not None and gpu_vram_limit_gb > 0) or quantization != "none":
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                cpu_mem = f"{ram_gb * 0.8:.0f}GiB"
            except ImportError:
                cpu_mem = "64GiB"
            
            # Hvis vi ikke har noen VRAM limit, men bruker kvantisering, bare la device_map="auto" fikse det
            if gpu_vram_limit_gb is not None and gpu_vram_limit_gb > 0:
                max_memory = {
                    0: f"{gpu_vram_limit_gb}GiB", 
                    "cpu": cpu_mem 
                }
                kwargs["max_memory"] = max_memory
                logger.info(f"VRAM Limit satt til {gpu_vram_limit_gb}GB.")
            
            kwargs["device_map"] = "auto"
            use_device_map = True

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    except TypeError as e:
        if "dtype" in str(e):
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["model_dtype"] = fallback_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), **fallback_kwargs)
        else:
            raise
    model.eval()

    if not use_device_map:
        if device == "cuda" and torch.cuda.is_available():
            model.to("cuda")
        else:
            model.to("cpu")

    # OPTIMERING MED TORCH.COMPILE
    if cuda_compile:
        if hasattr(torch, "compile"):
            logger.info("Kompilerer modellen med torch.compile (dette tar 1-2 minutter ved første generering, men blir lynraskt etterpå)...")
            try:
                model = torch.compile(model)
                logger.info("Modellen ble kompilert.")
            except Exception as e:
                logger.warning(f"torch.compile feilet: {e}. Fortsetter uten.")
        else:
            logger.warning("torch.compile mangler (krever PyTorch >= 2.0).")

    logger.info(
        f"Loaded model from {model_dir} | dtype={str(model_dtype).replace('torch.', '')} | device={device} | attn={kwargs.get('attn_implementation','(default)')} | max_pos={get_model_max_pos(model)}"
    )
    return model


@torch.inference_mode()
def sample_model_batch(
    model,
    input_ids: torch.Tensor,
    gen_batch: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    use_cache: bool = True,
    cache_implementation: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """Generate gen_batch continuations for a single prompt."""
    if gen_batch <= 1:
        in_ids = input_ids
    else:
        in_ids = input_ids.repeat(int(gen_batch), 1)

    gen_kwargs = dict(
        do_sample=True,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        use_cache=bool(use_cache),
    )

    # Avoid early stopping if eos_token_id is unset / meaningless for your training.
    if cache_implementation and str(cache_implementation).lower() != "auto":
        gen_kwargs["cache_implementation"] = cache_implementation


    # For worst-case OOM recovery we may disable KV cache entirely.
    if not bool(use_cache):
        gen_kwargs.pop("cache_implementation", None)
        gen_kwargs["use_cache"] = False

    try:
        gen_kwargs["eos_token_id"] = None
    except Exception:
        pass

    try:
        out = model.generate(in_ids, **gen_kwargs)
    except TypeError:
        # Older transformers: eos_token_id=None may not be accepted
        gen_kwargs.pop("eos_token_id", None)
        out = model.generate(in_ids, **gen_kwargs)

    return out


# ----------------------------
# Post-processing helpers (chord discipline + anti-repeat)
# ----------------------------

def clip_notes(notes: List[NoteEvent], start_tick: int, end_tick: int) -> List[NoteEvent]:
    out: List[NoteEvent] = []
    for n in notes:
        if n.end <= start_tick or n.start >= end_tick:
            continue
        s = max(start_tick, n.start)
        e = min(end_tick, n.end)
        if e > s:
            out.append(NoteEvent(start=s, end=e, pitch=n.pitch, vel=n.vel))
    return out


def clip_notes_to_range(notes: List[NoteEvent], start_tick: int, end_tick: int) -> List[NoteEvent]:
    """Backward-compatible alias used by some generation paths."""
    return clip_notes(notes, start_tick, end_tick)


def split_on_times(notes: List[NoteEvent], split_times: Sequence[int]) -> List[NoteEvent]:
    """Split notes at each split_time that falls strictly inside [start,end)."""
    if not notes or not split_times:
        return notes
    times = sorted(set(int(t) for t in split_times))
    out: List[NoteEvent] = []
    for n in notes:
        cur_s = int(n.start)
        cur_e = int(n.end)
        cur_p = int(n.pitch)
        cur_v = int(n.vel)
        cuts = [t for t in times if cur_s < t < cur_e]
        if not cuts:
            out.append(n)
            continue
        s = cur_s
        for t in cuts:
            out.append(NoteEvent(start=s, end=t, pitch=cur_p, vel=cur_v))
            s = t
        out.append(NoteEvent(start=s, end=cur_e, pitch=cur_p, vel=cur_v))
    return out


def _infer_decoded_time_scale(
    decoded_notes: List[NoteEvent],
    win_bars: int,
    grid_per_bar: int,
    grid_ticks: int,
) -> int:
    """Infer whether decoded note times are in 'grid steps' or in MIDI ticks.

    Some Miditok tokenizers can decode note times as position indices (0..bars*grid_per_bar)
    instead of absolute MIDI ticks. In that case, multiply by `grid_ticks` to map to ticks.

    This heuristic is intentionally conservative to avoid accidental double-scaling.
    """
    if not decoded_notes:
        return 1
    gt = int(grid_ticks)
    if gt <= 0:
        return 1

    win_bars_i = max(1, int(win_bars))
    gpb_i = max(1, int(grid_per_bar))
    expected_steps = win_bars_i * gpb_i

    times: List[int] = []
    for n in decoded_notes:
        times.append(int(n.start))
        times.append(int(n.end))
    times = [t for t in times if t > 0]
    if not times:
        return 1

    max_t = max(times)

    # If the decoded times look grid-aligned in ticks, assume they're already ticks.
    mult = sum(1 for t in times if (t % gt) == 0)
    frac_mult = float(mult) / float(len(times))

    # If values are tiny relative to expected step span AND not tick-aligned, assume steps.
    # 'Tiny' is deliberately low to avoid misclassifying short tick segments.
    if max_t <= expected_steps * 2 + gpb_i and frac_mult < 0.20:
        return gt

    # Classic step-domain: max time stays near expected steps (with small spill margin)
    if max_t <= expected_steps * 4 + gpb_i * 8 and frac_mult < 0.35:
        return gt

    return 1

def _scale_note_times_to_ticks(notes: List[NoteEvent], scale: int) -> List[NoteEvent]:
    """Multiply note start/end by `scale` (if scale != 1)."""
    if scale == 1 or not notes:
        return notes
    s = int(scale)
    if s <= 0:
        return notes
    out: List[NoteEvent] = []
    for n in notes:
        st = int(n.start) * s
        en = int(n.end) * s
        if en <= st:
            en = st + s
        out.append(NoteEvent(start=int(st), end=int(en), pitch=int(n.pitch), vel=int(n.vel)))
    return out


def analyze_chord(pitches):
    if not pitches: return 0, False, False
    pcs = set(p % 12 for p in pitches)
    best_root, best_score = min(pitches) % 12, -1
    is_maj, is_min = False, False
    for root in pcs:
        score = 0
        has_M3, has_m3, has_P5 = (root+4)%12 in pcs, (root+3)%12 in pcs, (root+7)%12 in pcs
        if has_P5: score += 3
        if has_M3: score += 2
        if has_m3: score += 2
        if score > best_score:
            best_score, best_root, is_maj, is_min = score, root, has_M3, has_m3
    if best_score <= 0: is_min = True
    return best_root, is_maj, is_min

def get_local_scale(chords: List[Tuple[int, List[int]]], t: int) -> List[int]:
    if not chords: return [0, 2, 4, 5, 7, 9, 11]
    idx = 0
    for i, (ct, _) in enumerate(chords):
        if ct <= t: idx = i
        else: break
    start_idx, end_idx = max(0, idx - 2), min(len(chords) - 1, idx + 2)
    scale_pcs = set()
    for i in range(start_idx, end_idx + 1):
        r, ma, mi = analyze_chord(chords[i][1])
        ivs = [0, 2, 3, 5, 7, 8, 10] if mi else [0, 2, 4, 5, 7, 9, 11]
        for iv in ivs: scale_pcs.add((r + iv) % 12)
    if len(scale_pcs) < 5:
        r, ma, mi = analyze_chord(chords[idx][1])
        ivs = [0, 2, 3, 5, 7, 8, 10] if mi else [0, 2, 4, 5, 7, 9, 11]
        return [(r + iv) % 12 for iv in ivs]
    return sorted(list(scale_pcs))

def snap_pitch_to_chord(pitch: int, chord_pitches: List[int], lo: int, hi: int, local_scale_pcs: Optional[List[int]] = None) -> int:
    if not chord_pitches:
        return int(max(lo, min(hi, pitch)))
    pcs = sorted({p % 12 for p in chord_pitches})
    if (pitch % 12) in pcs and lo <= pitch <= hi:
        return pitch
    if local_scale_pcs and (pitch % 12) in local_scale_pcs and lo <= pitch <= hi:
        return pitch
    target_pcs = local_scale_pcs if local_scale_pcs else pcs
    best, best_dist = None, 10**9
    for spc in target_pcs:
        base = spc + 12 * int(round((pitch - spc) / 12.0))
        for cand in (base - 12, base, base + 12):
            if cand < lo or cand > hi: continue
            dist = abs(cand - pitch)
            if dist < best_dist:
                best_dist, best = dist, cand
    return int(best) if best is not None else int(max(lo, min(hi, pitch)))

def apply_phrase_transposition(notes: List[NoteEvent], chords: List[Tuple[int, List[int]]], lo: int, hi: int, phrase_gap: int) -> List[NoteEvent]:
    if not notes or not chords: return notes
    notes = sorted(notes, key=lambda x: x.start)
    phrases, cur_phrase, last_end = [], [], -1
    for n in notes:
        if not cur_phrase or n.start - last_end <= phrase_gap:
            cur_phrase.append(n); last_end = max(last_end, n.end)
        else:
            phrases.append(cur_phrase); cur_phrase = [n]; last_end = n.end
    if cur_phrase: phrases.append(cur_phrase)
    out = []
    for phrase in phrases:
        best_shift, best_score, best_tie = 0, -99999, 999
        for shift in range(-11, 12):
            score, oor = 0, 0
            for n in phrase:
                p = n.pitch + shift
                if p < lo or p > hi: oor += 1; continue
                c = find_chord_at_time(chords, n.start)
                s = get_local_scale(chords, n.start)
                if (p % 12) in s: score += 2
                if (p % 12) in [x % 12 for x in c]: score += 1
            total = score - (oor * 10)
            if total > best_score or (total == best_score and abs(shift) < best_tie):
                best_score, best_shift, best_tie = total, shift, abs(shift)
        for n in phrase:
            p_s = n.pitch + best_shift
            c = find_chord_at_time(chords, n.start)
            s = get_local_scale(chords, n.start)
            p2 = snap_pitch_to_chord(p_s, c, lo, hi, local_scale_pcs=s)
            out.append(NoteEvent(start=n.start, end=n.end, pitch=p2, vel=n.vel))
    return sorted(out, key=lambda x: (x.start, x.end))

def reduce_repetitions(
    notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    lo: int,
    hi: int,
    grid_ticks: int,
    max_same_in_row: int = 2,
) -> List[NoteEvent]:
    """If the model spits the same pitch repeatedly on consecutive grid steps, nudge to another chord tone."""
    if not notes:
        return notes
    out: List[NoteEvent] = []
    notes = sorted(notes, key=lambda x: (x.start, x.end, x.pitch))
    last_pitch = None
    streak = 0
    for n in notes:
        p = n.pitch
        if last_pitch is not None and p == last_pitch and (abs(n.start - out[-1].start) <= grid_ticks):
            streak += 1
        else:
            streak = 1
        if streak > max_same_in_row:
            chord = find_chord_at_time(chords, n.start)
            # pick a different chord tone near current pitch
            pcs = sorted({pp % 12 for pp in chord}) if chord else []
            if pcs:
                # try all pcs except current
                alt_pcs = [pc for pc in pcs if pc != (p % 12)] or pcs
                cand_p = p
                best = None
                best_dist = 10**9
                for pc in alt_pcs:
                    base = pc + 12 * int(round((p - pc) / 12))
                    for cand in (base - 12, base, base + 12):
                        if lo <= cand <= hi and cand != p:
                            dist = abs(cand - p)
                            if dist < best_dist:
                                best_dist = dist
                                best = cand
                if best is not None:
                    cand_p = int(best)
                    p = cand_p
                    streak = 1
        out.append(NoteEvent(start=n.start, end=n.end, pitch=int(max(lo, min(hi, p))), vel=n.vel))
        last_pitch = out[-1].pitch
    return out


def postprocess_track_notes(
    notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    song_end: int,
    lo: int,
    hi: int,
    grid_ticks: int,
    split_on_chord_changes: bool,
    snap_to_chord_tones: bool,
    repetition_fix: bool,
) -> List[NoteEvent]:
    if not notes:
        return []
    # clip to song
    notes = clip_notes(notes, 0, song_end)

    # quantize start/end
    q: List[NoteEvent] = []
    for n in notes:
        s = quantize_tick(n.start, grid_ticks)
        e = quantize_tick(n.end, grid_ticks)
        if e <= s:
            e = s + max(1, grid_ticks)
        p = int(max(lo, min(hi, n.pitch)))
        q.append(NoteEvent(start=s, end=min(song_end, e), pitch=p, vel=int(max(1, min(127, n.vel)))))
    q.sort(key=lambda x: (x.start, x.end, x.pitch))

    if split_on_chord_changes and chords:
        split_times = [t for t, _ in chords]
        q = split_on_times(q, split_times)

    if snap_to_chord_tones and chords:
        q = apply_phrase_transposition(q, chords, lo, hi, phrase_gap=grid_ticks * 8)

    if repetition_fix:
        q = reduce_repetitions(q, chords=chords, lo=lo, hi=hi, grid_ticks=grid_ticks)

    # final clamp + sort
    q = [n for n in q if n.end > n.start]
    q.sort(key=lambda x: (x.start, x.end, x.pitch))
    return q

# ----------------------------
# Claire (woodwinds) shaping: longer notes + monophonic + chord-safe
# ----------------------------

def transpose_notes_block_to_range(notes: List[NoteEvent], lo: int, hi: int) -> List[NoteEvent]:
    """Transpose entire block by a *constant* octave shift (±12n) to fit into [lo, hi].

    Uses `compute_octave_shift_12`. If no all-fit shift exists, remaining outliers are clipped
    as a last resort (keeps MIDI valid).
    """
    if not notes:
        return []
    shift = compute_octave_shift_12(notes, lo, hi)
    out: List[NoteEvent] = []
    for n in notes:
        p = int(n.pitch) + int(shift)
        if p < lo:
            p = lo
        elif p > hi:
            p = hi
        out.append(NoteEvent(start=int(n.start), end=int(n.end), pitch=p, vel=int(n.vel)))
    return out

def _next_chord_change_tick(chords: List[Tuple[int, List[int]]], t: int, song_end: int) -> int:
    """Return the next chord start tick strictly after t, else song_end."""
    if not chords:
        return song_end
    lo, hi = 0, len(chords) - 1
    best = song_end
    while lo <= hi:
        mid = (lo + hi) // 2
        ct = int(chords[mid][0])
        if ct <= t:
            lo = mid + 1
        else:
            best = ct
            hi = mid - 1
    return int(best)

def _dedupe_same_start_keep_best(notes: List[NoteEvent]) -> List[NoteEvent]:
    """If multiple notes share the same start tick, keep the most 'important' one."""
    if not notes:
        return []
    groups: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        groups.setdefault(int(n.start), []).append(n)
    out: List[NoteEvent] = []
    for t in sorted(groups.keys()):
        g = groups[t]
        best = max(g, key=lambda x: (int(x.vel), int(x.end) - int(x.start), -abs(int(x.pitch) - 72)))
        out.append(best)
    out.sort(key=lambda x: (x.start, x.end, x.pitch))
    return out

def _make_monophonic_by_trimming(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Trim overlaps so no note begins before the previous ends (single-note line)."""
    if not notes:
        return []
    notes = sorted(notes, key=lambda x: (x.start, x.end, x.pitch))
    out: List[NoteEvent] = []
    for n in notes:
        if not out:
            if int(n.end) > int(n.start):
                out.append(n)
            continue
        prev = out[-1]
        if int(n.start) < int(prev.end):
            pe = min(int(prev.end), int(n.start))
            if pe <= int(prev.start):
                out.pop()
            else:
                out[-1] = NoteEvent(start=int(prev.start), end=pe, pitch=int(prev.pitch), vel=int(prev.vel))
        if int(n.end) > int(n.start):
            out.append(NoteEvent(start=int(n.start), end=int(n.end), pitch=int(n.pitch), vel=int(n.vel)))
    out = [n for n in out if int(n.end) > int(n.start)]
    out.sort(key=lambda x: (x.start, x.end, x.pitch))
    return out


def _closest_chord_pitch_dir(anchor: int, chord_pitches: List[int], lo: int, hi: int, direction: int) -> Optional[int]:
    """Return nearest chord tone strictly above (direction=+1) or below (direction=-1) anchor."""
    if not chord_pitches:
        return None
    pcs = sorted({int(p) % 12 for p in chord_pitches})
    if not pcs:
        return None
    anchor = int(anchor); lo = int(lo); hi = int(hi)
    # generate all chord-tone candidates within range across octaves
    cand_list: List[int] = []
    o0 = int(lo // 12) - 1
    o1 = int(hi // 12) + 1
    for o in range(o0, o1 + 1):
        for pc in pcs:
            cand = int(12 * o + pc)
            if lo <= cand <= hi:
                cand_list.append(cand)
    if not cand_list:
        return None
    if direction > 0:
        ups = [c for c in cand_list if c > anchor]
        return int(min(ups, key=lambda x: (x - anchor, x))) if ups else None
    downs = [c for c in cand_list if c < anchor]
    return int(max(downs, key=lambda x: (x, -(anchor - x)))) if downs else None


def rewrite_hammer_repetitions_pattern(
    notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    lo: int,
    hi: int,
    *,
    gap_ticks: int,
    min_run: int = 3,
) -> List[NoteEvent]:
    """
    Rewrite "hammering" (many repeats of same pitch) into a simple musical oscillation:
        up, stay, down, stay, up, stay, ...
    where up/down are adjacent chord tones (within range).
    """
    if not notes or min_run <= 1:
        return notes
    notes = sorted(notes, key=lambda x: (int(x.start), int(x.end), int(x.pitch)))
    out: List[NoteEvent] = []
    i = 0
    gap_ticks = int(max(1, gap_ticks))
    while i < len(notes):
        base = notes[i]
        base_pitch = int(base.pitch)
        chord0 = find_chord_at_time(chords, int(base.start)) if chords else []
        pcs0 = tuple(sorted({p % 12 for p in chord0})) if chord0 else tuple()

        j = i + 1
        # group consecutive repeats of same pitch within gap, and same chord pcs
        while j < len(notes):
            prev = notes[j - 1]
            cur = notes[j]
            if int(cur.pitch) != base_pitch:
                break
            if int(cur.start) - int(prev.start) > gap_ticks:
                break
            chordj = find_chord_at_time(chords, int(cur.start)) if chords else []
            pcsj = tuple(sorted({p % 12 for p in chordj})) if chordj else tuple()
            if pcsj != pcs0:
                break
            j += 1

        group = notes[i:j]
        if len(group) >= int(min_run) and chord0:
            up_p = _closest_chord_pitch_dir(base_pitch, chord0, lo, hi, direction=+1)
            dn_p = _closest_chord_pitch_dir(base_pitch, chord0, lo, hi, direction=-1)
            pattern = ("up", "stay", "down", "stay")
            for k, n in enumerate(group):
                if k == 0:
                    new_p = base_pitch
                else:
                    mode = pattern[(k - 1) % len(pattern)]
                    if mode == "up" and up_p is not None:
                        new_p = int(up_p)
                    elif mode == "down" and dn_p is not None:
                        new_p = int(dn_p)
                    else:
                        new_p = base_pitch
                out.append(NoteEvent(start=int(n.start), end=int(n.end), pitch=int(new_p), vel=int(n.vel)))
        else:
            out.extend(group)

        i = j

    out.sort(key=lambda x: (int(x.start), int(x.end), int(x.pitch)))
    return out


def postprocess_claire_notes(
    notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    song_end: int,
    *,
    lo: int,
    hi: int,
    grid_ticks: int,
    quantize_mult: int = 2,
    min_note_steps: int = 2,
    max_note_steps: int = 24,
    sustain: bool = True,
    monophonic: bool = True,
    snap_to_chord_tones: bool = True,
    hammer_alt: bool = True,
    hammer_min_run: int = 3,
    hammer_gap_steps: int = 1,
) -> List[NoteEvent]:
    """
    Claire-specific shaping:
      - quantize to a coarser grid (default 1/16 when grid_per_bar=32 and quantize_mult=2)
      - preserve model contour by block-transposing into playable range first
      - THEN (optionally) snap to nearest chord tone
      - remove duplicates at same start
      - extend notes to next note or chord change (sustain)
      - split at chord changes (chord-safe)
      - enforce monophonic line (no overlaps / no stacked notes)
      - rewrite "hammering" (repeated same pitch) into up/stay/down/stay chord-tone oscillation
    """
    if not notes:
        return []
    lo = int(lo); hi = int(hi)
    notes = clip_notes(notes, 0, song_end)
    
    # Blokk-transponer alt slik at melodien bevares!
    notes = transpose_notes_block_to_range(notes, lo, hi)

    if snap_to_chord_tones and chords:
        notes = apply_phrase_transposition(notes, chords, lo, hi, phrase_gap=grid_ticks * 8)

    qstep = max(1, int(grid_ticks) * max(1, int(quantize_mult)))
    min_len = max(1, int(grid_ticks) * max(1, int(min_note_steps)))
    max_len = max(1, int(grid_ticks) * max(1, int(max_note_steps)))

    # Quantize starts lightly (default = grid_ticks), keep model rhythm; map pitch by octave-to-range then chord-snap.
    qnotes: List[NoteEvent] = []

    start_step = max(1, int(grid_ticks))  # preserve rhythmic intent
    end_step = max(1, int(qstep))        # optional coarser end grid via --claire_quantize_mult

    def _current_chord_start_tick(chs: List[Tuple[int, List[int]]], t: int) -> int:
        if not chs:
            return 0
        last = 0
        for tt, _ in chs:
            if int(tt) <= int(t):
                last = int(tt)
            else:
                break
        return int(last)

    for n in notes:
        s_raw = int(n.start)
        e_raw = int(n.end)

        # Light quantize of start, but NEVER jump across a chord boundary.
        s = quantize_tick(s_raw, start_step)
        if chords:
            cur_start = _current_chord_start_tick(chords, s_raw)
            nxt = _next_chord_change_tick(chords, s_raw, song_end)
            if s < cur_start or s >= nxt:
                s = s_raw

        # End quantize (optional), but keep end after start.
        e = quantize_tick(e_raw, end_step)
        if e <= s:
            e = s + max(1, start_step)

        p = int(max(lo, min(hi, int(n.pitch))))

        qnotes.append(NoteEvent(start=int(s), end=int(min(song_end, e)), pitch=int(p), vel=int(max(1, min(127, n.vel)))))

    qnotes.sort(key=lambda x: (x.start, x.end, x.pitch))
    qnotes = _dedupe_same_start_keep_best(qnotes)
    if sustain:
        # IMPORTANT: do NOT override the model's rhythm.
        # We only extend *too-short* notes up to min_len, and we cap notes at next note / chord change.
        sustained: List[NoteEvent] = []
        for i, n in enumerate(qnotes):
            s0 = int(n.start)
            e0 = int(n.end)
            next_start = int(qnotes[i + 1].start) if i + 1 < len(qnotes) else song_end
            chord_cap = _next_chord_change_tick(chords, s0, song_end) if chords else song_end

            dur0 = max(0, e0 - s0)
            if dur0 < min_len:
                e = min(s0 + min_len, s0 + max_len, next_start, chord_cap, song_end)
            else:
                e = min(e0, s0 + max_len, next_start, chord_cap, song_end)

            if e <= s0:
                continue
            sustained.append(NoteEvent(start=s0, end=e, pitch=int(n.pitch), vel=int(n.vel)))
        qnotes = sustained

    # Always ensure chord-safe (split) at chord starts (except at 0)
    if chords:
        split_times = [int(t) for t, _ in chords if int(t) > 0]
        qnotes = split_on_times(qnotes, split_times)

    qnotes = _dedupe_same_start_keep_best(qnotes)
    if monophonic:
        qnotes = _make_monophonic_by_trimming(qnotes)

    # Rewrite repeated hammering AFTER mono/split so we don't create overlaps
    if hammer_alt and chords:
        gap_ticks = qstep * max(1, int(hammer_gap_steps))
        qnotes = rewrite_hammer_repetitions_pattern(
            qnotes, chords=chords, lo=lo, hi=hi, gap_ticks=gap_ticks, min_run=int(hammer_min_run)
        )
        qnotes = _dedupe_same_start_keep_best(qnotes)
        if monophonic:
            qnotes = _make_monophonic_by_trimming(qnotes)

    qnotes = [n for n in qnotes if int(n.end) > int(n.start)]
    qnotes.sort(key=lambda x: (x.start, x.end, x.pitch))
    return qnotes



# ----------------------------
# Argent ARP postprocessing (4-string hits per step)
# ----------------------------

def _select_spread(pitches: List[int], k: int) -> List[int]:
    """Pick k pitches spread across the sorted list (deterministic)."""
    if k <= 0:
        return []
    if len(pitches) <= k:
        return list(pitches)
    idxs = np.linspace(0, len(pitches) - 1, num=k)
    sel: List[int] = []
    for i in idxs:
        sel.append(pitches[int(round(float(i)))])
    out: List[int] = []
    for p in sel:
        if p not in out:
            out.append(p)
    if len(out) < k:
        for p in pitches:
            if p not in out:
                out.append(p)
            if len(out) >= k:
                break
    return out[:k]


def enforce_argent_4hits_per_step(
    notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    song_end: int,
    lo: int,
    hi: int,
    grid_ticks: int,
    *,
    target_notes: int = 4,
    top_pitch_hint: Optional[int] = None,
    step_len_mult: float = 1.0,
    max_dur_steps: int = 2,
    roll_frac: float = 0.18,
) -> List[NoteEvent]:
    """
    Ensures each grid step that has any activity becomes a 4-note 'hit' (approx 4 strings),
    using chord tones as fillers when needed.

    The model still decides *where* hits happen; we only enforce per-hit polyphony and
    keep fills inside the chord.
    """
    if not notes:
        return []

    grouped: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        s = quantize_tick(n.start, grid_ticks)
        grouped.setdefault(s, []).append(NoteEvent(start=s, end=n.end, pitch=n.pitch, vel=n.vel))

    step_len = max(1, int(round(grid_ticks * float(step_len_mult))))
    max_len = max(step_len, int(max(1, max_dur_steps)) * grid_ticks)

    roll_step = 0
    if target_notes > 1 and roll_frac > 0:
        roll_step = max(0, int(round((grid_ticks * float(roll_frac)) / max(1, target_notes - 1))))

    out: List[NoteEvent] = []
    for s in sorted(grouped.keys()):
        g = grouped[s]
        pitches = sorted({int(max(lo, min(hi, n.pitch))) for n in g})
        if not pitches:
            continue

        picked = _select_spread(pitches, int(target_notes))

        # Fill missing notes from chord tones (near hint)
        if len(picked) < target_notes:
            chord = find_chord_at_time(chords, s) if chords else []
            hint = int(top_pitch_hint) if top_pitch_hint is not None else int(max(lo, min(hi, (lo + hi) // 2)))
            fill = choose_voicing(chord, top_pitch=hint, target_size=int(target_notes), lo=lo, hi=hi) if chord else []
            for p in fill:
                if p not in picked:
                    picked.append(int(p))
                if len(picked) >= target_notes:
                    break

        # Last resort: octave duplicates in range
        while len(picked) < target_notes:
            base = picked[-1] if picked else int(max(lo, min(hi, 60)))
            cand = base - 12 if base - 12 >= lo else (base + 12 if base + 12 <= hi else base)
            if cand in picked:
                cand = int(max(lo, min(hi, cand + 1)))
            picked.append(int(cand))

        picked = sorted(picked[:target_notes])

        v = int(np.median([n.vel for n in g])) if g else 96
        v = int(max(1, min(127, v)))

        cap_end = min(song_end, s + max_len)

        # Don't roll past next chord boundary if one is super-close
        next_ch = None
        if chords:
            for t, _ in chords:
                if t > s:
                    next_ch = t
                    break
        cap_roll_end = (max(s, next_ch - 1) if next_ch is not None else (s + grid_ticks))

        for i, p in enumerate(picked):
            ss = s + i * roll_step
            if ss > cap_roll_end:
                ss = cap_roll_end
            ee = min(cap_end, ss + step_len)
            if ee <= ss:
                ee = min(song_end, ss + max(1, grid_ticks))
            out.append(NoteEvent(start=int(ss), end=int(ee), pitch=int(p), vel=int(v)))

    out.sort(key=lambda x: (x.start, x.end, x.pitch))
    return out


# ----------------------------
# Shreddage 3.5 Argent keyswitch injection
# ----------------------------

def inject_argent_keyswitches_and_poly(
    notes: List[NoteEvent],
    *,
    grid_ticks: int,
    ks_mute: int = 1,        # C#-2 (usually MIDI 1)
    ks_staccato: int = 2,    # D-2  (usually MIDI 2)
    poly_note: int = 113,    # F7 (MIDI 113) - user-requested "polyphonic mode" / reset behavior
    ks_len_ticks: int = 24,
    poly_len_ticks: int = 36,
    advance_ticks: int = 10,
    mode: str = "auto",      # auto | mute | staccato | none
    include_poly_note: bool = True,
) -> List[NoteEvent]:
    """
    Output-only injection for Shreddage:
      - Adds articulation keyswitches (Mute / Staccato) before hits.
      - Adds an initial "F7 (113)" behavioral keyswitch at time 0 if enabled.

    IMPORTANT: These keyswitch notes are NOT fed back into the model prompt (we append gen_clean to prompt).
    """
    if not notes:
        # still inject poly if requested
        if include_poly_note:
            t0 = 0  # start of song (ticks)
            return [NoteEvent(start=t0, end=t0 + max(1, int(poly_len_ticks)), pitch=int(poly_note), vel=110)]
        return []

    m = (mode or "auto").strip().lower()
    if m not in ("auto", "mute", "staccato", "none"):
        m = "auto"

    # group by start (assumes chords/arps are already quantized to grid)
    grouped: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        grouped.setdefault(int(n.start), []).append(n)

    starts = sorted(grouped.keys())
    out: List[NoteEvent] = []

    if include_poly_note:
        out.append(NoteEvent(start=0, end=max(1, int(poly_len_ticks)), pitch=int(poly_note), vel=110))

    last_art = None
    adv = max(0, int(advance_ticks))
    ks_len = max(1, int(ks_len_ticks))

    for i, s0 in enumerate(starts):
        g = grouped[s0]
        if not g:
            continue
        end0 = max(n.end for n in g)
        dur = int(max(1, end0 - s0))
        vel = int(np.median([n.vel for n in g]))
        avg_pitch = float(np.mean([n.pitch for n in g]))

        # choose articulation
        art = None
        if m == "none":
            art = None
        elif m == "mute":
            art = "mute"
        elif m == "staccato":
            art = "staccato"
        else:
            # auto: chug = mute; more open short = staccato
            if dur <= grid_ticks * 2 and avg_pitch <= 58:
                art = "mute"
            elif dur <= max(1, int(grid_ticks * 1.25)):
                art = "staccato"
            else:
                # keep previous (no switch spam)
                art = None

        # inject KS slightly before hit, only if changed (or if we want first)
        if art is not None and art != last_art:
            ks_pitch = int(ks_mute) if art == "mute" else int(ks_staccato)
            tks = max(0, int(s0) - adv)
            out.append(NoteEvent(start=tks, end=tks + ks_len, pitch=ks_pitch, vel=110))
            last_art = art

        out.extend(g)

    # Sort so that keyswitches (low pitches) naturally come before playable notes at the same tick
    out.sort(key=lambda x: (x.start, x.pitch, x.end))
    return out


# ----------------------------
# Claire keyswitch + "breath" CC curves
# ----------------------------

# In Claire manuals they explicitly recommend riding CC1 (Dynamics) + CC11 (Expression)
# to make notes "breathe" and avoid static playback.

CLAIRE_KS_ORDERS: Dict[str, List[Tuple[str, int]]] = {
    # 8 slots: C0, C#0, D0, D#0, E0, F0, F#0, G0  (offsets 0..7 from claire_c0_midi)
    "FLUTE": [
        ("natural", 0),
        ("medium1", 1),
        ("medium2", 2),
        ("strong1", 3),
        ("strong2", 4),
        ("sus_xfade", 5),
        ("staccattissimo", 6),
        ("marcato", 7),
    ],
    "OBOE": [
        ("natural", 0),
        ("medium1", 1),
        ("medium2", 2),
        ("strong1", 3),
        ("strong2", 4),
        ("sus_xfade", 5),
        ("staccattissimo", 6),
        ("marcato", 7),
    ],
    "PICCOLO": [
        ("natural", 0),
        ("soft1", 1),
        ("medium1", 2),
        ("medium2", 3),
        ("strong", 4),
        ("sus_xfade", 5),
        ("staccattissimo", 6),
        ("marcato", 7),
    ],
    "ALTO_FLUTE": [
        ("natural", 0),
        ("soft1", 1),
        ("soft2", 2),
        ("medium", 3),
        ("strong", 4),
        ("sus_xfade", 5),
        ("staccattissimo", 6),
        ("marcato", 7),
    ],
    "CLARINET": [
        ("natural", 0),
        ("medium_arc", 1),
        ("strong_arc1", 2),
        ("strong_arc2", 3),
        ("vibrato", 4),
        ("heavy_vibrato", 5),
        ("staccattissimo", 6),
        ("marcato", 7),
    ],
}

CLAIRE_DEFAULT_ORDER: List[Tuple[str, int]] = [
    ("natural", 0),
    ("medium1", 1),
    ("medium2", 2),
    ("strong1", 3),
    ("strong2", 4),
    ("sus_xfade", 5),
    ("staccattissimo", 6),
    ("marcato", 7),
]


def _claire_kind_from_spec_name(spec_name: str) -> str:
    u = (spec_name or "").upper()
    if "PICCOLO" in u:
        return "PICCOLO"
    if "ALTO" in u and "FLUTE" in u:
        return "ALTO_FLUTE"
    if "CLARINET" in u:
        return "CLARINET"
    if "OBOE" in u:
        return "OBOE"
    if "FLUTE" in u:
        return "FLUTE"
    return "DEFAULT"


def build_claire_keyswitch_map(spec_name: str, c0_midi: int) -> Dict[str, int]:
    kind = _claire_kind_from_spec_name(spec_name)
    order = CLAIRE_KS_ORDERS.get(kind, CLAIRE_DEFAULT_ORDER)
    return {name: int(c0_midi) + int(off) for name, off in order}


def choose_claire_articulation(
    *,
    spec_name: str,
    note_len_ticks: int,
    vel: int,
    grid_ticks: int,
    is_downbeat: bool,
    allow_vibrato: bool,
) -> str:
    kind = _claire_kind_from_spec_name(spec_name)
    note_len_ticks = int(note_len_ticks)
    vel = int(max(1, min(127, vel)))

    # short notes
    if note_len_ticks <= max(1, grid_ticks // 2):
        return "staccattissimo"

    # accented short-ish downbeats
    if is_downbeat and note_len_ticks <= grid_ticks * 2:
        return "marcato"

    # long notes -> crossfade sustain if available
    if note_len_ticks >= grid_ticks * 6:
        return "sus_xfade"

    # otherwise choose dynamic "arc" based on velocity
    if kind == "PICCOLO":
        if vel <= 55:
            return "soft1"
        if vel <= 90:
            return "medium1"
        return "strong"

    if kind == "ALTO_FLUTE":
        if vel <= 50:
            return "soft1"
        if vel <= 72:
            return "soft2"
        if vel <= 96:
            return "medium"
        return "strong"

    if kind == "CLARINET":
        # optional vibrato on longer notes if enabled
        if allow_vibrato and note_len_ticks >= grid_ticks * 4:
            return "heavy_vibrato" if vel >= 105 else "vibrato"
        if vel <= 78:
            return "medium_arc"
        if vel <= 104:
            return "strong_arc1"
        return "strong_arc2"

    # flute/oboe/default
    if vel <= 80:
        return "medium1"
    if vel <= 104:
        return "strong1"
    return "strong2"


def inject_claire_keyswitches(
    notes: List[NoteEvent],
    *,
    spec_name: str,
    claire_c0_midi: int,
    grid_ticks: int,
    allow_vibrato: bool = False,
    phrase_gap_steps: int = 2,
    seed: int = 1234,
    force_on_phrase: bool = True,
    force_every_note: bool = False,
) -> List[NoteEvent]:
    if not notes:
        return []
    ks_map = build_claire_keyswitch_map(spec_name, claire_c0_midi)
    out: List[NoteEvent] = []
    last_art: Optional[str] = None

    advance = min(24, max(1, int(grid_ticks // 2)))
    phrase_gap = max(1, int(phrase_gap_steps)) * max(1, int(grid_ticks))
    rng = random.Random(int(seed) + (zlib.adler32(spec_name.encode('utf-8')) & 0xFFFF))

    def _maybe_variant(a: str, vel: int) -> str:
        # Gentle variation between adjacent arc articulations if both exist.
        if a.startswith('medium'):
            opts = [x for x in ('medium1', 'medium2') if x in ks_map]
            if len(opts) >= 2:
                p2 = 0.25 + 0.35 * (max(0, min(127, vel)) / 127.0)
                return opts[1] if rng.random() < p2 else opts[0]
        if a.startswith('strong'):
            opts = [x for x in ('strong1', 'strong2') if x in ks_map]
            if len(opts) >= 2:
                p2 = 0.35 + 0.40 * (max(0, min(127, vel)) / 127.0)
                return opts[1] if rng.random() < p2 else opts[0]
        if a == 'soft1' and 'soft2' in ks_map:
            return 'soft2' if rng.random() < 0.28 else 'soft1'
        return a

    grouped: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        grouped.setdefault(int(n.start), []).append(n)

    prev_end: Optional[int] = None

    for s0 in sorted(grouped.keys()):
        g = grouped[s0]
        if not g:
            continue
        end0 = max(n.end for n in g)
        vel = int(np.median([n.vel for n in g]))
        is_downbeat = (s0 % (grid_ticks * 8) == 0)

        new_phrase = prev_end is None or (int(s0) - int(prev_end) >= phrase_gap)

        art = choose_claire_articulation(
            spec_name=spec_name,
            note_len_ticks=int(end0 - s0),
            vel=vel,
            grid_ticks=grid_ticks,
            is_downbeat=is_downbeat,
            allow_vibrato=allow_vibrato,
        )
        art = _maybe_variant(str(art), vel)

        if art not in ks_map:
            art = 'natural' if 'natural' in ks_map else list(ks_map.keys())[0]

        if new_phrase and force_on_phrase:
            last_art = None

        if bool(force_every_note) or art != last_art:
            ks_pitch = ks_map.get(art)
            if ks_pitch is not None:
                t = max(0, int(s0) - int(advance))
                out.append(NoteEvent(start=t, end=t + max(1, int(advance)), pitch=int(ks_pitch), vel=110))
            last_art = art

        out.extend(g)
        prev_end = max(int(prev_end) if prev_end is not None else 0, int(end0))

    out.sort(key=lambda x: (x.start, x.pitch, x.end))
    return out


def _merge_cc_events(events: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Keep last value per (time, cc)."""
    if not events:
        return []
    events = [(int(t), int(cc), int(max(0, min(127, v)))) for (t, cc, v) in events]
    events.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, int, int]] = []
    last_key = None
    for t, cc, v in events:
        key = (t, cc)
        if out and key == last_key:
            out[-1] = (t, cc, v)
        else:
            out.append((t, cc, v))
            last_key = key
    return out


def generate_claire_breath_cc(
    notes: List[NoteEvent],
    *,
    tpb: int,
    grid_ticks: int,
    cc1_base: int = 35,
    cc1_peak: int = 92,
    cc11_base: int = 70,
    cc11_peak: int = 112,
    jitter: int = 4,
    seed: int = 1234,
    phrase_gap_steps: int = 2,
) -> List[Tuple[int, int, int]]:
    """
    Create rolling CC1/CC11 curves per phrase and per note onset.
    Keeps density low (few points) so it stays musical and DAW-friendly.
    """
    if not notes:
        return []

    rng = random.Random(int(seed))

    # group by start (monophonic/mostly monophonic, but safe)
    grouped: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        grouped.setdefault(int(n.start), []).append(n)
    starts = sorted(grouped.keys())

    # initial set
    events: List[Tuple[int, int, int]] = [
        (0, 1, int(max(0, min(127, cc1_base)))),
        (0, 11, int(max(0, min(127, cc11_base)))),
    ]

    prev_end = None

    for i, s0 in enumerate(starts):
        g = grouped[s0]
        if not g:
            continue
        end0 = max(n.end for n in g)
        dur = int(max(1, end0 - s0))
        vel_med = float(np.median([n.vel for n in g]))
        vel_norm = max(0.05, min(1.0, vel_med / 127.0))

        # phrase detection (gap before this onset)
        new_phrase = False
        if prev_end is None:
            new_phrase = True
        else:
            if (s0 - prev_end) >= int(max(1, phrase_gap_steps) * grid_ticks):
                new_phrase = True

        # next onset for legato overlap handling
        next_start = starts[i + 1] if i + 1 < len(starts) else None

        # scale peaks with velocity
        p1 = int(cc1_base + (cc1_peak - cc1_base) * (vel_norm ** 0.75))
        p11 = int(cc11_base + (cc11_peak - cc11_base) * (vel_norm ** 0.65))

        # tiny human jitter
        p1 = int(max(0, min(127, p1 + rng.randint(-jitter, jitter))))
        p11 = int(max(0, min(127, p11 + rng.randint(-jitter, jitter))))

        # attack / release as fraction of duration but capped
        atk = max(1, min(int(grid_ticks), int(dur * 0.18)))
        rel = max(1, min(int(grid_ticks), int(dur * 0.22)))

        pre = max(0, int(s0) - min(12, max(1, grid_ticks // 4)))
        t_peak = int(s0)
        t_atk = int(min(end0, s0 + atk))
        t_rel = int(max(s0, end0 - rel))
        t_end = int(end0)

        # if legato-ish (next starts immediately), do not drop fully to base
        drop_base_1 = cc1_base
        drop_base_11 = cc11_base
        if next_start is not None and next_start <= (end0 + max(1, grid_ticks // 2)):
            drop_base_1 = int(min(cc1_peak, cc1_base + 18))
            drop_base_11 = int(min(cc11_peak, cc11_base + 14))

        # phrase start gets a small "in-breath"
        if new_phrase:
            events.append((pre, 1, int(max(0, min(127, cc1_base - 8)))))
            events.append((pre, 11, int(max(0, min(127, cc11_base - 6)))))
        else:
            events.append((pre, 1, int(max(0, min(127, cc1_base)))))
            events.append((pre, 11, int(max(0, min(127, cc11_base)))))

        # main swell
        events.append((t_peak, 1, p1))
        events.append((t_peak, 11, p11))

        # slight decay / shaping
        events.append((t_atk, 1, int(max(0, min(127, round(p1 * 0.92))))))
        events.append((t_atk, 11, int(max(0, min(127, round(p11 * 0.96))))))

        # tail
        events.append((t_rel, 1, int(max(0, min(127, round(p1 * 0.82))))))
        events.append((t_rel, 11, int(max(0, min(127, round(p11 * 0.88))))))

        events.append((t_end, 1, int(max(0, min(127, drop_base_1)))))
        events.append((t_end, 11, int(max(0, min(127, drop_base_11)))))

        prev_end = end0

    return _merge_cc_events(events)

# ----------------------------
# Candidate scoring
# ----------------------------

def compute_octave_shift_12(notes: List[NoteEvent], lo: int, hi: int, max_steps: int = 6) -> int:
    """Compute a *constant* octave shift (multiple of 12) that best fits a note-block into [lo, hi].

    Preference:
      1) Any shift that makes *all* notes fit.
      2) Otherwise maximize notes-in-range.
    Ties: mean pitch closest to center, then smaller |shift|.
    """
    if not notes:
        return 0
    pitches = [int(n.pitch) for n in notes]
    mn, mx = min(pitches), max(pitches)
    mean = sum(pitches) / float(len(pitches))
    center = 0.5 * (lo + hi)

    all_fit = []
    for k in range(-max_steps, max_steps + 1):
        s = 12 * k
        if mn + s >= lo and mx + s <= hi:
            all_fit.append(s)
    if all_fit:
        return min(all_fit, key=lambda s: (abs((mean + s) - center), abs(s)))

    best_s = 0
    best_in = -1
    best_tie = float("inf")
    for k in range(-max_steps, max_steps + 1):
        s = 12 * k
        in_cnt = sum(1 for p in pitches if lo <= p + s <= hi)
        tie = abs((mean + s) - center)
        if (in_cnt > best_in) or (in_cnt == best_in and (tie < best_tie - 1e-9 or (abs(tie - best_tie) < 1e-9 and abs(s) < abs(best_s)))):
            best_in = in_cnt
            best_tie = tie
            best_s = s
    return best_s


def score_candidate(
    seg_notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    seg_start: int,
    seg_end: int,
    grid_ticks: int,
    bar_ticks: int,
    lo: int,
    hi: int,
    melody: Optional[List[NoteEvent]] = None,
    melody_overlap: float = 0.25,
    avoid_melody_active: bool = False,
    min_notes_in_seg: int = 0,
    seed: int = 0,
) -> float:
    """Heuristic scoring for candidate notes in [seg_start, seg_end).

    Notes and chord times are assumed to be absolute ticks.
    """
    total = len(seg_notes)
    if total == 0:
        return -1e9

    in_range = sum(1 for n in seg_notes if lo <= int(n.pitch) <= hi)
    range_ratio = in_range / float(max(1, total))

    chord_hits = 0
    chord_notes = 0
    for n in seg_notes:
        chord_pitches = find_chord_at_time(chords, int(n.start))
        if chord_pitches:
            pcs = {p % 12 for p in chord_pitches}
            chord_notes += 1
            if int(n.pitch) % 12 in pcs:
                chord_hits += 1
    chord_ratio = (chord_hits / float(chord_notes)) if chord_notes else 0.0

    chord_changes = [ct for (ct, _) in chords if seg_start <= ct < seg_end]
    if not chord_changes:
        cover_ratio = 1.0 if chord_notes else 0.0
    else:
        covered = 0
        for i, ct in enumerate(chord_changes):
            nt = chord_changes[i + 1] if i + 1 < len(chord_changes) else seg_end
            if any(ct <= int(n.start) < nt for n in seg_notes):
                covered += 1
        cover_ratio = covered / float(max(1, len(chord_changes)))

    gt = max(1, int(grid_ticks))
    steps = [int(n.start) // gt for n in seg_notes]
    unique_steps = len(set(steps))
    seg_steps = max(1, (seg_end - seg_start) // gt)
    disp_ratio = unique_steps / float(seg_steps)

    counts: Dict[int, int] = {}
    for n in seg_notes:
        p = int(n.pitch)
        counts[p] = counts.get(p, 0) + 1
    max_rep = max(counts.values()) / float(max(1, total))

    bticks = max(1, int(bar_ticks))
    n_seg_bars = max(1, int(math.ceil(max(1, (seg_end - seg_start)) / float(bticks))))
    occupied_bars = set()
    for n in seg_notes:
        if seg_start <= int(n.start) < seg_end:
            bi = int((int(n.start) - int(seg_start)) // bticks)
            if 0 <= bi < n_seg_bars:
                occupied_bars.add(bi)
    bar_fill_ratio = len(occupied_bars) / float(max(1, n_seg_bars))
    longest_empty_run = 0
    cur_empty_run = 0
    for bi in range(n_seg_bars):
        if bi in occupied_bars:
            longest_empty_run = max(longest_empty_run, cur_empty_run)
            cur_empty_run = 0
        else:
            cur_empty_run += 1
    longest_empty_run = max(longest_empty_run, cur_empty_run)
    gap_pen = longest_empty_run / float(max(1, n_seg_bars))

    overlap_pen = 0.0
    if avoid_melody_active and melody:
        mel = [mn for mn in melody if not (int(mn.end) <= seg_start or int(mn.start) >= seg_end)]
        if mel:
            ov = 0
            for n in seg_notes:
                ns, ne = int(n.start), int(n.end)
                for mn in mel:
                    ms, me = int(mn.start), int(mn.end)
                    inter = max(0, min(ne, me) - max(ns, ms))
                    ov += inter
            total_dur = sum(max(1, int(n.end) - int(n.start)) for n in seg_notes)
            ov_ratio = ov / float(max(1, total_dur))
            if ov_ratio > melody_overlap:
                overlap_pen = min(1.0, (ov_ratio - melody_overlap) / max(1e-9, (1.0 - melody_overlap)))

    density_pen = 0.0
    if min_notes_in_seg > 0 and total < min_notes_in_seg:
        density_pen = min(1.0, (min_notes_in_seg - total) / float(max(1, min_notes_in_seg)))

    score = 0.0
    score += 6.0 * chord_ratio
    score += 1.5 * cover_ratio
    score += 1.0 * min(1.0, disp_ratio / 0.35)
    score += 1.0 * range_ratio
    score += 2.5 * bar_fill_ratio

    score -= 2.0 * overlap_pen
    score -= 1.5 * density_pen
    score -= 2.0 * gap_pen
    if max_rep > 0.55:
        score -= 1.0 * (max_rep - 0.55) / 0.45

    return float(score)


def choose_voicing(chord_pitches: List[int], top_pitch: int, target_size: int, lo: int, hi: int) -> List[int]:
    if not chord_pitches:
        return []
    pcs = sorted({p % 12 for p in chord_pitches})
    pool: List[int] = []
    for octave in range(-2, 12):
        for pc in pcs:
            p = pc + 12 * octave
            if lo <= p <= hi:
                pool.append(p)
    pool = sorted(set(pool))
    if not pool:
        return []
    top = min(pool, key=lambda p: abs(p - top_pitch))
    below = [p for p in pool if p <= top and p != top]
    chosen = [top]
    i_lo, i_hi = 0, len(below) - 1
    toggle = True
    while len(chosen) < target_size and i_lo <= i_hi:
        if toggle:
            chosen.append(below[i_lo]); i_lo += 1
        else:
            chosen.append(below[i_hi]); i_hi -= 1
        toggle = not toggle
    chosen = sorted(set(chosen))
    if chosen and chosen[-1] != top:
        chosen = [p for p in chosen if p != top] + [top]
    return chosen


def add_seed_pattern(
    win_mid: MidiFile,
    chords: List[Tuple[int, List[int]]],
    win_start_tick: int,
    spec: InstrumentSpec,
    tpb: int,
    grid_ticks: int,
    seed_vel: int,
    seed_len_ticks: int,
    insert_at_tick: int = 0,
) -> None:
    """Add a tiny example snippet on spec.program to bias the model toward the intended style."""
    chord_now = find_chord_at_time(chords, win_start_tick) if chords else []
    if not chord_now:
        chord_now = [60, 64, 67]  # C triad fallback

    inst = Instrument(program=int(spec.program), is_drum=False, name=f"{spec.name}_SEED")
    base_t = max(0, int(insert_at_tick))

    seed_len = max(1, int(seed_len_ticks))

    if spec.seed_style == "arp_up":
        pitches = choose_voicing(chord_now, top_pitch=min(spec.hi, max(spec.lo, 64)), target_size=4, lo=spec.lo, hi=spec.hi)
        pitches = sorted(pitches)[:4] or [60, 64, 67, 72]
        t0 = base_t  # FIKSET BUG: Var 0!
        for p in pitches:
            inst.notes.append(Note(velocity=int(seed_vel), pitch=int(p), start=int(t0), end=int(t0 + seed_len)))
            t0 += max(1, grid_ticks // 2)

    elif spec.seed_style == "arp_down":
        pitches = choose_voicing(chord_now, top_pitch=min(spec.hi, max(spec.lo, 76)), target_size=4, lo=spec.lo, hi=spec.hi)
        pitches = sorted(pitches, reverse=True)[:4] or [72, 67, 64, 60]
        t0 = base_t  # FIKSET BUG: Var 0!
        for p in pitches:
            inst.notes.append(Note(velocity=int(seed_vel), pitch=int(p), start=int(t0), end=int(t0 + seed_len)))
            t0 += max(1, grid_ticks // 2)

    elif spec.seed_style == "argent_4hit":
        # Demonstrate a 4-note 'hit' repeated across 2 steps (biases model to poly hits)
        pitches = choose_voicing(
            chord_now,
            top_pitch=min(spec.hi, max(spec.lo, 72)),
            target_size=4,
            lo=spec.lo,
            hi=spec.hi,
        )
        pitches = sorted(pitches)[:4] or [60, 64, 67, 72]
        for k in range(2):
            t0 = int(base_t + k * grid_ticks)  # FIKSET BUG: Var k*grid_ticks
            for p in pitches:
                inst.notes.append(Note(velocity=int(seed_vel), pitch=int(p), start=t0, end=int(t0 + seed_len)))

    elif spec.seed_style == "sustain":
        pitches = choose_voicing(chord_now, top_pitch=min(spec.hi, max(spec.lo, 72)), target_size=1, lo=spec.lo, hi=spec.hi)
        p = pitches[-1] if pitches else int(max(spec.lo, min(spec.hi, 72)))
        inst.notes.append(Note(velocity=int(seed_vel), pitch=int(p), start=base_t, end=int(base_t + max(seed_len * 4, grid_ticks * 2))))

    else:  # "chord"
        pitches = choose_voicing(chord_now, top_pitch=min(spec.hi, max(spec.lo, 68)), target_size=3, lo=spec.lo, hi=spec.hi)
        if not pitches:
            pitches = [60, 64, 67]
        for p in pitches:
            inst.notes.append(Note(velocity=int(seed_vel), pitch=int(p), start=base_t, end=base_t + max(seed_len, grid_ticks)))  # FIKSET BUG: Var 0

    win_mid.instruments.append(inst)


# ----------------------------
# Output MIDI writer
# ----------------------------

def note_events_to_mido_track(
    notes: List[NoteEvent],
    name: str,
    program: int,
    channel: int,
    pitch_bends: Optional[List[Tuple[int, int]]] = None,
    control_changes: Optional[List[Tuple[int, int, int]]] = None,
) -> mido.MidiTrack:
    pitch_bends = pitch_bends or []
    control_changes = control_changes or []

    events: List[Tuple[int, mido.Message]] = []
    events.append((0, mido.MetaMessage("track_name", name=name, time=0)))
    events.append((0, mido.Message("program_change", program=int(program) % 128, channel=int(channel) % 16, time=0)))
    events.append((0, mido.Message("pitchwheel", pitch=0, channel=int(channel) % 16, time=0)))

    for t, cc, val in control_changes:
        events.append((int(t), mido.Message("control_change", control=int(cc), value=int(val), channel=int(channel) % 16, time=0)))
    for t, bend in pitch_bends:
        b = int(max(-8192, min(8191, int(bend))))
        events.append((int(t), mido.Message("pitchwheel", pitch=b, channel=int(channel) % 16, time=0)))
    for n in notes:
        # mido uses the attribute name "velocity" (not "vel")
        events.append((int(n.start), mido.Message("note_on", note=int(n.pitch), velocity=int(n.vel), channel=int(channel) % 16, time=0)))
        events.append((int(n.end), mido.Message("note_off", note=int(n.pitch), velocity=0, channel=int(channel) % 16, time=0)))

    def _order(msg: mido.Message) -> int:
        t = getattr(msg, "type", "")
        if t in ("track_name", "set_tempo", "time_signature"):
            return 0
        if t == "program_change":
            return 1
        if t == "control_change":
            return 2
        if t == "pitchwheel":
            return 3
        if t == "note_off":
            return 4
        if t == "note_on":
            return 5
        return 6

    events_sorted = sorted(events, key=lambda x: (x[0], _order(x[1])))
    tr = mido.MidiTrack()
    last = 0
    for t, msg in events_sorted:
        t = int(max(0, t))
        if t < last:
            t = last
        d = t - last
        last = t
        tr.append(msg.copy(time=int(d)))
    return tr


def _meta_events_to_mido_track(
    tempo_changes: List[TempoChange],
    time_sigs: List[TimeSignature],
) -> mido.MidiTrack:
    events: List[Tuple[int, mido.MetaMessage]] = []
    events.append((0, mido.MetaMessage("track_name", name="META", time=0)))

    # Keep the full tempo map from the source MIDI, not just the first event.
    seen_tempo: set[Tuple[int, int]] = set()
    for tc in sorted(tempo_changes, key=lambda x: int(getattr(x, "time", 0))):
        tick = max(0, int(getattr(tc, "time", 0)))
        bpm = float(getattr(tc, "tempo", 120.0))
        tempo_us = int(mido.bpm2tempo(bpm))
        key = (tick, tempo_us)
        if key in seen_tempo:
            continue
        seen_tempo.add(key)
        events.append((tick, mido.MetaMessage("set_tempo", tempo=tempo_us, time=0)))

    seen_ts: set[Tuple[int, int, int]] = set()
    for ts in sorted(time_sigs, key=lambda x: int(getattr(x, "time", 0))):
        tick = max(0, int(getattr(ts, "time", 0)))
        num = int(getattr(ts, "numerator", 4))
        den = int(getattr(ts, "denominator", 4))
        key = (tick, num, den)
        if key in seen_ts:
            continue
        seen_ts.add(key)
        events.append((
            tick,
            mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0),
        ))

    def _order(msg: mido.MetaMessage) -> int:
        t = getattr(msg, "type", "")
        if t == "track_name":
            return 0
        if t == "time_signature":
            return 1
        if t == "set_tempo":
            return 2
        return 3

    events_sorted = sorted(events, key=lambda x: (x[0], _order(x[1])))
    tr = mido.MidiTrack()
    last = 0
    for t, msg in events_sorted:
        t = int(max(0, t))
        if t < last:
            t = last
        d = t - last
        last = t
        tr.append(msg.copy(time=int(d)))
    return tr


def write_output_midi(
    out_path: Path,
    tpb: int,
    tempo_changes: List[TempoChange],
    time_sigs: List[TimeSignature],
    specs: List[InstrumentSpec],
    notes_by_name: Dict[str, List[NoteEvent]],
    control_changes_by_name: Optional[Dict[str, List[Tuple[int, int, int]]]],
    logger: logging.Logger,
) -> None:
    mid = mido.MidiFile(ticks_per_beat=int(tpb))
    mid.tracks.append(_meta_events_to_mido_track(tempo_changes=tempo_changes, time_sigs=time_sigs))

    for spec in specs:
        notes = notes_by_name.get(spec.name, [])
        ccs = control_changes_by_name.get(spec.name, []) if control_changes_by_name else []
        mid.tracks.append(
            note_events_to_mido_track(
                notes=notes,
                name=spec.name,
                program=spec.program,
                channel=spec.channel,
                pitch_bends=[],
                control_changes=ccs,
            )
        )

    mid.save(str(out_path))
    logger.info(f"Wrote: {out_path.resolve()}")


# ----------------------------
# Generation (cascade)
# ----------------------------

def instrument_to_miditoolkit(spec: InstrumentSpec, notes: List[NoteEvent]) -> Instrument:
    inst = Instrument(program=int(spec.program), is_drum=False, name=str(spec.name))
    inst.notes = [Note(start=int(n.start), end=int(n.end), pitch=int(n.pitch), velocity=int(n.vel)) for n in notes]
    return inst


def determine_song_end(mid: MidiFile) -> int:
    end_tick = 0
    for inst in mid.instruments:
        for n in inst.notes:
            end_tick = max(end_tick, int(n.end))
    return int(end_tick)


def generate_one_instrument(
    *,
    spec: InstrumentSpec,
    prompt_mid_full: MidiFile,
    chords: List[Tuple[int, List[int]]],
    melody: List[NoteEvent],
    model,
    tok,
    tpb: int,
    ts: Tuple[int, int],
    song_end: int,
    grid_per_bar: int,
    anchor_ticks: int,
    argent_notes_per_step: int,
    argent_roll_frac: float,
    argent_step_len_mult: float,
    argent_max_dur_steps: int,
    argent_rhythm_mode: str,
    argent_arp_density: float,
    argent_arp_dur_steps: int,
    argent_strums_per_bar: int,
    argent_strum_notes: int,
    argent_strum_roll_frac: float,
    argent_strum_dur_steps: int,
    argent_arp1_top_pitch: int,
    argent_arp2_top_pitch: int,
    bars_per_chunk: int,
    context_before_bars: int,
    context_after_bars: int,
    prompt_mode: str,
    prompt_include_generated: bool,
    prompt_max_bass_tracks: int,
    prompt_max_piano_tracks: int,
    prompt_exclude_regex: Optional[str],
    auto_context_clamp: bool,
    min_gen_new_tokens: int,
    samples_per_bar: int,
    samples_per_chunk: Optional[int],
    max_new_tokens_per_call: int,
    gen_batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    melody_overlap: float,
    avoid_melody_active: bool,
    split_on_chord_changes: bool,
    snap_to_chord_tones: bool,
    repetition_fix: bool,
    seed_vel: int,
    seed_ticks: int,
    progress_enabled: bool,
    progress_ncols: Optional[int],
    progress_mininterval: float,
    time_map: str = "auto",
    time_map_min_notes: int = 1,
    claire_min_notes_per_bar: int = 8,
    bar_from: int = 0,
    bar_to: int = -1,
    logger: logging.Logger = logging.getLogger(__name__),
    cache_implementation: Optional[str] = None,
    vram_gc_threshold: float = 0.0,
    vram_gc_cooldown_chunks: int = 4,
    vram_gc_state: Optional[dict] = None,
) -> List[NoteEvent]:
    bt = bar_ticks(tpb, ts)
    grid_ticks = max(1, bt // max(1, int(grid_per_bar)))

    n_bars = max(1, int(math.ceil(song_end / bt)))
    chunk_start0 = max(0, int(bar_from))
    chunk_end0 = n_bars if int(bar_to) < 0 else min(n_bars, int(bar_to))
    chunk_starts = list(range(chunk_start0, chunk_end0, max(1, int(bars_per_chunk))))

    combined: List[NoteEvent] = []
    claire_fixed_oct_shift: Optional[int] = None  # constant per-instrument (Claire only)

    p_bars = _tqdm_bar(
        total=(chunk_end0 - chunk_start0),
        desc=f"{spec.name} bars",
        unit="bar",
        enabled=progress_enabled,
        leave=True,
        ncols=progress_ncols,
        mininterval=progress_mininterval,
    )

    # Deterministic seeds per instrument
    base_seed = int(seed) + (int(spec.program) * 1009) + (int(spec.channel) * 17)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    max_pos = get_model_max_pos(model)
    _vram_gc_state = vram_gc_state if vram_gc_state is not None else VRAM_GC_STATE
    _cache_impl = cache_implementation
    if _cache_impl is not None and str(_cache_impl).lower() == "auto":
        _cache_impl = None

    for ci, start_bar in enumerate(chunk_starts):
        chunk_bars = min(int(bars_per_chunk), n_bars - start_bar)
        seg_start = start_bar * bt
        seg_end = min(song_end, (start_bar + chunk_bars) * bt)

        win_start_bar = max(0, start_bar - int(max(0, context_before_bars)))
        win_end_bar = min(n_bars, start_bar + chunk_bars + int(max(0, context_after_bars)))
        win_start = win_start_bar * bt
        win_end = min(song_end, win_end_bar * bt)

        # Minimum notes per segment (especially for Claire so we don't pick 0–1 note candidates)
        min_notes_in_seg = int(claire_min_notes_per_bar * chunk_bars) if getattr(spec, "is_claire", False) else 0

        # --- Build + clamp prompt window so we always leave room for generation ---
        cb = int(max(0, context_before_bars))
        ca = int(max(0, context_after_bars))

        min_new = max(16, int(min_gen_new_tokens))
        max_prompt_len_target = max(64, int(max_pos) - int(min_new) - 1)

        last_prompt_len: Optional[int] = None
        clamped = False

        while True:
            win_start_bar = max(0, start_bar - cb)
            # Important: end prompt at the START of the target segment so generation lands inside this bar.
            # For the very first bar (start_bar==0) we include bar 0 as 'primer' because there is no past context.
            if int(start_bar) == 0:
                win_end_bar = min(n_bars, 1)
            else:
                win_end_bar = int(start_bar)
            if win_end_bar <= win_start_bar:
                win_end_bar = min(n_bars, win_start_bar + 1)
            win_start = win_start_bar * bt
            # Anchor: include a small slice of the target bar so boundary notes (chords/seed) survive quantization.
            # If anchor_ticks==0 we use grid_ticks (1/16 for grid_per_bar=32 in 4/4).
            _anch = int(anchor_ticks) if int(anchor_ticks) > 0 else int(grid_ticks)
            _anch = max(1, _anch)
            if int(start_bar) == 0:
                win_end = min(song_end, bt)
            else:
                win_end = min(song_end, int(seg_start) + _anch)
            cutoff_local = int(max(0, int(seg_start) - int(win_start)))

            win_mid = slice_miditoolkit_midi(prompt_mid_full, win_start, win_end, shift_to_zero=True)

            seed_dur = int(seed_ticks) if int(seed_ticks) > 0 else max(1, int(tpb // 16))
            # Place the seed exactly at the target bar start (local time).
            add_seed_pattern(
                win_mid=win_mid,
                chords=chords,
                win_start_tick=int(seg_start),
                spec=spec,
                tpb=tpb,
                grid_ticks=grid_ticks,
                seed_vel=int(seed_vel),
                seed_len_ticks=seed_dur,
                insert_at_tick=max(0, cutoff_local - grid_ticks),
            )

            # Filter prompt content (reduce token bloat)
            win_mid = filter_prompt_instruments(
                win_mid,
                prompt_mode=str(prompt_mode),
                include_generated=bool(prompt_include_generated),
                max_bass_tracks=int(prompt_max_bass_tracks),
                max_piano_tracks=int(prompt_max_piano_tracks),
                exclude_regex=prompt_exclude_regex,
            )

            # Trim prompt so it ends at target bar start (chords + seed allowed at boundary).
            win_mid = trim_prompt_to_cutoff(win_mid, cutoff_tick=cutoff_local)

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp_path = Path(f.name)
            try:
                win_mid.dump(str(tmp_path))
                prompt_ids = encode_midi_to_ids(tok, tmp_path)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

            if not prompt_ids:
                logger.warning(f"{spec.name} chunk {ci}: empty prompt ids, skipping")
                break

            last_prompt_len = len(prompt_ids)

            if (not auto_context_clamp) or (last_prompt_len <= max_prompt_len_target):
                break

            # Too long: shrink context bars (prefer shrinking after-bars first)
            if ca > 0:
                ca -= 1
                clamped = True
                continue
            if cb > 0:
                cb -= 1
                clamped = True
                continue

            # Still too long: downgrade prompt_mode
            if str(prompt_mode).strip().lower() != "chords_only":
                prompt_mode = "chords_only"
                clamped = True
                continue

            logger.warning(
                f"{spec.name} chunk {ci}: prompt_len={last_prompt_len} still > target={max_prompt_len_target} even after clamping; generation may be weak."
            )
            break

        if not prompt_ids:
            continue

        if clamped and last_prompt_len is not None:
            logger.debug(
                f"{spec.name} chunk {ci+1}: prompt clamped | cb={cb} ca={ca} mode={prompt_mode} prompt_len={last_prompt_len} target<={max_prompt_len_target}"
            )

        # Determine sampling budget
        if samples_per_chunk is None:
            n_samples = max(1, int(samples_per_bar) * int(chunk_bars))
        else:
            n_samples = max(1, int(samples_per_chunk))

        eff_max_new = max(16, int(max_new_tokens_per_call))
        prompt_len = len(prompt_ids)
        eff_max_new = max(1, min(eff_max_new, max(1, int(max_pos) - prompt_len - 1)))
        # Sanity clamp: for 1-bar chunks, generating thousands of tokens is almost always accidental
        # and can explode VRAM. Keep a hard cap unless you intentionally raise it.
        if int(chunk_bars) <= 1 and eff_max_new > 1024:
            logger.warning(f"{spec.name} bar {start_bar}: eff_new={eff_max_new} looks too high for 1-bar chunk; clamping to 1024 to avoid VRAM blowups.")
            eff_max_new = 1024

        batch = max(1, int(gen_batch_size))
        # VRAM guard: longer (prompt+gen) contexts scale memory roughly with batch * seq_len.
        # When the prompt is huge, proactively cap micro-batch to avoid OOM/allocator asserts.
        seq_len = int(prompt_len) + int(eff_max_new)
        if torch.cuda.is_available():
            if seq_len > 3600:
                batch = min(batch, 4)
            elif seq_len > 3200:
                batch = min(batch, 8)
        rounds = int(math.ceil(n_samples / batch))
        logger.info(f"{spec.name} chunk {ci+1}/{len(chunk_starts)} | bars {start_bar}-{start_bar+chunk_bars-1} | win {win_start_bar}-{win_end_bar-1} | prompt_ids={prompt_len} | eff_new={eff_max_new} | samples={n_samples} | batch={batch} | rounds={rounds} | {cuda_mem_stats()}")

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=("cuda" if torch.cuda.is_available() else "cpu"))

        # Decode ALL prompt notes across all tracks to subtract later
        prompt_notes_all = decode_ids_to_notes(tok, prompt_ids)

        # Infer whether decoded note times are in 'grid steps' (position indices) and scale to ticks.
        win_bars = max(1, int(win_end_bar - win_start_bar))
        decoded_scale = _infer_decoded_time_scale(
            prompt_notes_all,
            win_bars=win_bars,
            grid_per_bar=grid_per_bar,
            grid_ticks=grid_ticks,
        )
        
        if decoded_scale != 1:
            prompt_notes_all = _scale_note_times_to_ticks(prompt_notes_all, decoded_scale)
        tol = max(1, int(tpb // 16))

        def _looks_like_cuda_oom(err: BaseException) -> bool:
            s = str(err).lower()
            return ("out of memory" in s) and ("cuda" in s or "cublas" in s or "cudnn" in s or "hip" in s)

        def _looks_like_cuda_allocator_assert(err: BaseException) -> bool:
            s = str(err).lower()
            if "cudacachingallocator" not in s:
                return False
            return (
                ("internal assert failed" in s)
                or ("handles_.at(" in s)
                or ("cudacachingallocator.cpp" in s and "assert" in s)
            )

        # Stream-score candidates to avoid large RAM spikes.
        best_score = -1e18
        best: List[NoteEvent] = []
        n_generated = 0
        cache_impl_cur = _cache_impl
        use_cache_cur = True
        chunk_gen_tokens = 0
        chunk_gen_time = 0.0

        p_samp = _tqdm_bar(
            total=n_samples,
            desc=f"{spec.name} chunk {ci+1}/{len(chunk_starts)} sampling",
            unit="samp",
            enabled=progress_enabled,
            leave=False,
            ncols=progress_ncols,
            mininterval=progress_mininterval,
        )

        maybe_cuda_gc(
            logger,
            threshold=vram_gc_threshold,
            cooldown_chunks=vram_gc_cooldown_chunks,
            chunk_idx=ci,
            reason=f"{spec.name} pre-generate",
            state=_vram_gc_state,
        )
        for r in range(rounds):
            remaining = n_samples - n_generated
            if remaining <= 0:
                break

            bsz = min(batch, remaining)
            _tgen0 = time.perf_counter()
            cur_eff_new = eff_max_new
            attempt = 0

            while True:
                try:
                    out_ids = sample_model_batch(
                        model=model,
                        input_ids=input_ids,
                        gen_batch=bsz,
                        max_new_tokens=cur_eff_new,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        use_cache=use_cache_cur,
                        cache_implementation=cache_impl_cur,
                        logger=logger,
                    )
                    break
                except Exception as e:
                    # If this transformers build doesn't support a cache mode, fall back gracefully.
                    if isinstance(e, ValueError) and ("cache_implementation" in str(e).lower()):
                        if cache_impl_cur == "offloaded":
                            logger.warning("cache_implementation='offloaded' not supported here; falling back to 'dynamic'.")
                            cache_impl_cur = "dynamic"
                            continue
                    if isinstance(e, RuntimeError) and (_looks_like_cuda_oom(e) or _looks_like_cuda_allocator_assert(e)):
                        attempt += 1
                        logger.warning(
                            f"CUDA alloc failure (attempt={attempt}, batch={bsz}, new={cur_eff_new}, cache={cache_impl_cur}, use_cache={use_cache_cur}). Recovering..."
                        )
                        try:
                            maybe_cuda_gc(
                                threshold=vram_gc_threshold,
                                cooldown_chunks=0,
                                chunk_idx=ci,
                                reason=f"{spec.name} OOM-recover",
                                state=_vram_gc_state,
                                logger=logger,
                                force=True,
                            )
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        gc.collect()
                        # --- NY FIX: Slett cache-objektet HELT fra modellen ---
                        try:
                            if hasattr(model, "_cache"):
                                delattr(model, "_cache")
                        except Exception:
                            pass
                        # ----------------------------------------------------------------
                        # Recover strategy: first shrink batch/new to avoid repeated allocator failures,
                        # then fall back to alternative cache implementations, and finally disable KV cache entirely.
                        if bsz > 1:
                            bsz = max(1, bsz // 2)
                            # FIX: Static cache takler ikke at batch size endres dynamisk. Tving dynamic!
                            if cache_impl_cur == "static":
                                cache_impl_cur = "dynamic"
                            continue
                        if cur_eff_new > 128:
                            cur_eff_new = max(64, int(cur_eff_new * 0.75))
                            if cache_impl_cur == "static":
                                cache_impl_cur = "dynamic"
                            continue
                        if cache_impl_cur == "static":
                            cache_impl_cur = "dynamic"
                            continue
                        if cache_impl_cur in (None, "dynamic"):
                            cache_impl_cur = "offloaded"
                            continue
                        if use_cache_cur:
                            use_cache_cur = False
                            continue
                    raise

            _tgen1 = time.perf_counter()
            try:
                _new = int(out_ids.shape[1]) - int(input_ids.shape[1])
                _tok = max(0, _new) * int(out_ids.shape[0])
                _dt = max(1e-6, _tgen1 - _tgen0)
                _tok_s = _tok / _dt
                chunk_gen_tokens += _tok
                chunk_gen_time += _dt
            except Exception:
                _tok_s = 0.0

            if p_samp is not None:
                try:
                    p_samp.update(int(bsz))
                    p_samp.set_postfix({"bar": f"{start_bar+1}/{n_bars}", "prompt": prompt_len, "new": cur_eff_new, "batch": bsz, "tok/s": f"{_tok_s:.0f}", "mem": cuda_mem_stats()})
                except Exception:
                    pass

            out_np = out_ids.detach().cpu().numpy()
            del out_ids

            round_offset = int(n_generated)
            for ri, row in enumerate(out_np):
                ids = list(row)
                
                # EXTRACT ALL NOTES ignoring program numbers!
                all_out_notes = decode_ids_to_notes(tok, ids)
                
                if decoded_scale != 1:
                    all_out_notes = _scale_note_times_to_ticks(all_out_notes, decoded_scale)
                    
                # Subtract the full prompt context to leave ONLY the newly generated notes
                notes = subtract_prompt_notes(all_out_notes, prompt_notes_all, tol=tol)

                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        mn_s = min((n.start for n in notes), default=-1)
                        mx_s = max((n.start for n in notes), default=-1)
                        logger.debug(f"{spec.name} bar {start_bar}: raw_decoded={len(notes)} raw_range={mn_s}-{mx_s} win_start={win_start} seg={seg_start}-{seg_end}")
                    except Exception:
                        pass
                # Time mapping: build both candidates in auto mode and KEEP THE ONE WITH BEST SEGMENT SCORE.
                # The previous logic only fell back when absolute mapping had too few notes. That still allowed
                # candidates with enough notes overall but several empty bars inside the chunk.
                shifted_abs = [NoteEvent(start=n.start + win_start, end=n.end + win_start, pitch=n.pitch, vel=n.vel) for n in notes]
                shifted_abs = clip_notes(shifted_abs, seg_start, seg_end)

                mapped_variants: List[Tuple[str, List[NoteEvent]]] = [("absolute", shifted_abs)]
                if time_map in ("segment_bar", "auto") and len(notes) > 0:
                    shifted_bar = []
                    base = min(int(n.start) for n in notes)
                    for n in notes:
                        s = (int(n.start) - base) + seg_start
                        e = (int(n.end) - base) + seg_start
                        if e <= s:
                            e = min(seg_end, s + max(1, int(grid_ticks)))
                        shifted_bar.append(NoteEvent(start=s, end=e, pitch=n.pitch, vel=n.vel))
                    shifted_bar = clip_notes(shifted_bar, seg_start, seg_end)
                    mapped_variants.append(("segment_rebase", shifted_bar))

                pp_lo = 0 if getattr(spec, "is_claire", False) else spec.lo
                pp_hi = 127 if getattr(spec, "is_claire", False) else spec.hi
                pp_snap = False if getattr(spec, "is_claire", False) else snap_to_chord_tones
                pp_rep = False if getattr(spec, "is_claire", False) else repetition_fix

                global_i = round_offset + int(ri)
                local_best_variant_name = "absolute"
                local_best_variant_notes: List[NoteEvent] = []
                local_best_variant_score = -1e18

                for variant_name, variant_notes in mapped_variants:
                    cand = variant_notes

                    # Claire: compute octave shift once (first non-empty segment) and keep it constant.
                    if getattr(spec, "is_claire", False) and len(cand) > 0:
                        if claire_fixed_oct_shift is None:
                            claire_fixed_oct_shift = compute_octave_shift_12(cand, spec.lo, spec.hi)
                            logger.info(f"{spec.name}: fixed octave shift set to {claire_fixed_oct_shift:+d} semitones (computed from first non-empty segment)")
                        if claire_fixed_oct_shift:
                            cand = [NoteEvent(start=int(n.start), end=int(n.end), pitch=int(n.pitch) + int(claire_fixed_oct_shift), vel=int(n.vel)) for n in cand]
                        cand = [n for n in cand if spec.lo <= int(n.pitch) <= spec.hi]

                    cand = postprocess_track_notes(
                        cand,
                        chords=chords,
                        song_end=song_end,
                        lo=pp_lo,
                        hi=pp_hi,
                        grid_ticks=grid_ticks,
                        split_on_chord_changes=split_on_chord_changes,
                        snap_to_chord_tones=pp_snap,
                        repetition_fix=pp_rep,
                    )

                    cand_score = score_candidate(
                        cand,
                        chords=chords,
                        melody=melody,
                        seg_start=seg_start,
                        seg_end=seg_end,
                        grid_ticks=grid_ticks,
                        bar_ticks=bt,
                        lo=spec.lo,
                        hi=spec.hi,
                        melody_overlap=float(melody_overlap),
                        avoid_melody_active=avoid_melody_active,
                        min_notes_in_seg=min_notes_in_seg,
                        seed=base_seed + global_i * 19,
                    )
                    if cand_score > local_best_variant_score:
                        local_best_variant_score = cand_score
                        local_best_variant_name = variant_name
                        local_best_variant_notes = cand

                shifted = local_best_variant_notes
                used_time_map = local_best_variant_name
                sc = local_best_variant_score

                if logger.isEnabledFor(logging.DEBUG) and used_time_map != "absolute":
                    if shifted:
                        mn = min(n.start for n in shifted)
                        mx = max(n.start for n in shifted)
                        logger.debug(f"{spec.name} bar {start_bar}: time_map={used_time_map} -> seg_range={mn}-{mx}")
                    else:
                        logger.debug(f"{spec.name} bar {start_bar}: time_map={used_time_map} produced 0 notes")

                if sc > best_score:
                    best_score = sc
                    best = shifted

            # Track how many candidates we've generated so far (used for seeding and to clamp bsz).
            n_generated += int(out_np.shape[0])
            del out_np
            
        if p_samp is not None:
            try:
                p_samp.close()
            except Exception:
                pass

        # Clean up input_ids to prevent VRAM fragmentation before the next chunk
        try:
            del input_ids
        except NameError:
            pass

        # Pick best candidate
        # (best_score/best are selected on-the-fly during generation)

        avg_tok_s = chunk_gen_tokens / chunk_gen_time if chunk_gen_time > 0 else 0.0
        logger.info(f"{spec.name} chunk {ci+1}/{len(chunk_starts)} | bars {start_bar}-{start_bar+chunk_bars-1} | best_score={best_score:.3f} | notes={len(best)} | tok/s={avg_tok_s:.0f}")
        if not best:
            logger.warning(f"{spec.name} chunk {ci+1}/{len(chunk_starts)} returned 0 notes (try --prompt_mode chords_only or lower context bars or raise --min_gen_new_tokens).")
            try:
                vs = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
            except Exception:
                vs = 0
            logger.debug(f"{spec.name} chunk {ci+1}: prompt_ids_stats=" + debug_ids_stats(prompt_ids, vs))

        if p_bars is not None:
            try:
                p_bars.update(int(chunk_bars))
                p_bars.set_postfix({"chunk": f"{ci+1}/{len(chunk_starts)}", "bar": f"{min(start_bar+chunk_bars, n_bars)}/{n_bars}", "best": f"{best_score:.2f}", "notes": len(best), "tok/s": f"{avg_tok_s:.0f}"})
            except Exception:
                pass

        combined.extend(best)

    combined.sort(key=lambda x: (x.start, x.end, x.pitch))

    # final global postprocess (same rules)
    combined = postprocess_track_notes(
        combined,
        chords=chords,
        song_end=song_end,
        lo=spec.lo,
        hi=spec.hi,
        grid_ticks=grid_ticks,
        split_on_chord_changes=split_on_chord_changes,
        snap_to_chord_tones=snap_to_chord_tones,
        repetition_fix=repetition_fix,
    )


    if p_bars is not None:
        try:
            p_bars.close()
        except Exception:
            pass


    return combined


# ----------------------------
# Main
# ----------------------------

def pitch_to_name(pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[pitch % 12]

def print_chord_timeline(chords: List[Tuple[int, List[int]]], tpb: int, ts: Tuple[int, int], logger: logging.Logger):
    if not chords:
        return
    bt = int(round(tpb * ts[0] * (4.0 / ts[1])))
    logger.info("--- MUSIKALSK TIDSLINJE (AKKORDER & SKALAER) ---")
    
    # Viser ALLE akkorder (fordi vi har skjermoppløsning til det!)
    for item in chords:
        tick, pitches = item
            
        bar = (tick // bt) + 1
        beat = ((tick % bt) // tpb) + 1

        root, is_major, is_minor = analyze_chord(pitches)

        if is_minor:
            scale_type = "Moll"
            intervals = [0, 2, 3, 5, 7, 8, 10]
        elif is_major:
            scale_type = "Dur"
            intervals = [0, 2, 4, 5, 7, 9, 11]
        else:
            scale_type = "Moll/Power"
            intervals = [0, 2, 3, 5, 7, 8, 10]

        scale_pcs = [(root + iv) % 12 for iv in intervals]

        chord_names = [pitch_to_name(p) for p in sorted(set(p % 12 for p in pitches))]
        scale_names = [pitch_to_name(p) for p in scale_pcs]
        root_name = pitch_to_name(root)

        chord_lbl = f"{root_name} {scale_type}"
        logger.info(f"Takt {bar:03d} Slag {beat} (T:{tick:05d}) | {chord_lbl:<13} | Toner: {','.join(chord_names):<12} | Skala: {','.join(scale_names)}")
    logger.info("------------------------------------------------")


def tok_decode_compat(tok, obj):
    fn = getattr(tok, "decode", None)
    if callable(fn):
        return fn(obj)
    fn = getattr(tok, "tokens_to_midi", None)
    if callable(fn):
        return fn(obj)  # older MidiTok fallback
    raise AttributeError("Tokenizer has neither decode() nor tokens_to_midi()")


def tok_encode_compat(tok, midi_obj_or_path):
    for name in ("encode", "midi_to_ids", "midi_to_tokens"):
        fn = getattr(tok, name, None)
        if not callable(fn):
            continue
        try:
            return fn(midi_obj_or_path)
        except TypeError:
            continue
    raise AttributeError("Tokenizer has no compatible encode method")


def main() -> None:
    ap = argparse.ArgumentParser()

    # I/O
    ap.add_argument("--in_midi", required=True, type=Path)
    ap.add_argument("--out_midi", required=True, type=Path)

    # Model / tokenizer
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--tokenizer_json", required=True, type=Path)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cuDNN on CUDA for FP32 runs")
    ap.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        help="Transformers attention implementation. 'sdpa' is a good default; 'auto' tries flash -> sdpa.",
    )

    ap.add_argument(
        "--cache_implementation",
        type=str,
        default="auto",
        choices=["auto", "dynamic", "static", "offloaded"],
        help=(
            "KV-cache strategy passed to transformers.generate (auto=do not pass). "
            "static can be faster but uses much more VRAM."
        ),
    )
    ap.add_argument(
        "--gpu_vram_limit_gb", 
        type=float, 
        default=0.0, 
        help="Maks GB VRAM modellen får bruke til vekter. Resten offloades til RAM for å gi plass til KV-cache spikes. 0 = deaktivert."
    )
    ap.add_argument(
        "--cuda_compile", 
        action="store_true", 
        help="Bruk torch.compile for å JIT-kompilere modellen (raskere generering etter en kort oppvarmingsperiode)."
    )
    ap.add_argument(
        "--quantization", 
        type=str, 
        default="none", 
        choices=["none", "int8", "int4"], 
        help="Bruk bitsandbytes for å kjøre modellen i 8-bit eller 4-bit. Massiv fartsøkning og minnebesparelse på RTX-kort."
    )
    ap.add_argument(
        "--vram_gc_threshold",
        type=float,
        default=0.0,
        help="If >0, triggers torch.cuda.empty_cache when reserved/total >= threshold (e.g. 0.90).",
    )
    ap.add_argument(
        "--vram_gc_cooldown_chunks",
        type=int,
        default=4,
        help="Minimum number of chunks between VRAM GC calls.",
    )

    # Generation
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.06)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--grid_per_bar", type=int, default=32)
    ap.add_argument(
        "--anchor_ticks",
        type=int,
        default=0,
        help="How many ticks of the target bar to include in the prompt for time-anchoring. 0=auto (grid_ticks).",
    )
    ap.add_argument(
        "--time_map",
        type=str,
        default="auto",
        choices=["auto", "absolute", "segment_bar"],
        help="How to map decoded note times into the target bar. absolute=use window start; segment_bar=map notes into current bar (mod bar_ticks); auto=try absolute then fall back if empty.",
    )
    ap.add_argument(
        "--time_map_min_notes",
        type=int,
        default=1,
        help="When --time_map auto, keep absolute mapping if it yields at least this many notes in the target segment; otherwise fall back to segment_bar.",
    )


    ap.add_argument(
        "--claire_min_notes_per_bar",
        type=int,
        default=8,
        help="Minimum number of Claire notes per bar when selecting best candidate (prevents overly sparse wind 'stabs').",
    )


    ap.add_argument(
        "--bar_from",
        type=int,
        default=0,
        help="Start bar index (0-based) to process. Useful for quick tests / resume.",
    )
    ap.add_argument(
        "--bar_to",
        type=int,
        default=-1,
        help="End bar index (exclusive). -1 means process until song end.",
    )

    ap.add_argument("--bars_per_chunk", type=int, default=4)

    ap.add_argument("--context_before_bars", type=int, default=1)
    ap.add_argument("--context_after_bars", type=int, default=0)
    ap.add_argument("--prompt_mode", type=str, default="chords_bass", choices=["chords_only", "chords_bass", "chords_bass_piano"],
                    help="Which base tracks to keep in the prompt window. Use chords_only if prompt gets too long.")
    ap.add_argument("--prompt_include_generated", action="store_true", default=True,
                    help="Include previously generated ARGENT_/CLAIRE_ tracks in the prompt window for cascade.")
    ap.add_argument("--prompt_max_bass_tracks", type=int, default=1,
                    help="Keep at most this many bass tracks in the prompt (smallest by note-count).")
    ap.add_argument("--prompt_max_piano_tracks", type=int, default=1,
                    help="Keep at most this many piano tracks in the prompt (smallest by note-count).")
    ap.add_argument("--prompt_exclude_regex", type=str, default=None,
                    help="Regex of track names to exclude from the prompt (e.g. 'DARKWALL|Keyscape').")
    ap.add_argument("--min_gen_new_tokens", type=int, default=96,
                    help="Target minimum new tokens available per chunk. Auto clamps context/prompt_mode if needed.")
    ap.add_argument("--auto_context_clamp", action="store_true", default=True,
                    help="Auto-shrink prompt window and downgrade prompt_mode to ensure room for generation.")

    ap.add_argument("--samples_per_bar", type=int, default=4)
    ap.add_argument("--samples_per_chunk", type=int, default=None)
    ap.add_argument("--max_new_tokens_per_call", type=int, default=256)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    # Argent ARP shaping
    ap.add_argument("--argent_notes_per_step", type=int, default=2, help="How many simultaneous notes (strings) per ARP step.")
    ap.add_argument("--argent_roll_frac", type=float, default=0.18, help="Within-step strum roll as fraction of grid step (0 disables).")
    ap.add_argument("--argent_step_len_mult", type=float, default=1.0, help="Base note length as multiple of grid step (e.g. 1.0 = 1/16).")
    ap.add_argument("--argent_max_dur_steps", type=int, default=2, help="Cap note lengths to this many grid steps (keeps arps tight).")

    ap.add_argument("--argent_rhythm_mode", choices=["chug", "arp", "mixed"], default="arp",

                    help="ARGENT_ARP* rhythm shaping. chug=legacy dense chord-hits each grid step; arp=single-note arpeggio rhythm with end-of-bar strums; mixed=arp with occasional mini-chugs.")

    ap.add_argument("--argent_arp_density", type=float, default=0.55,

                    help="In arp/mixed mode: probability (0..1) of emitting an arp note at each candidate slot (templates are ~1/16 by default).")

    ap.add_argument("--argent_arp_dur_steps", type=int, default=2,

                    help="In arp/mixed mode: typical duration (in grid steps) for arp notes. Post-processing will still split on chord changes.")

    ap.add_argument("--argent_strums_per_bar", type=int, default=2,

                    help="In arp/mixed mode: how many full strums to place near the end of each bar (0 disables).")

    ap.add_argument("--argent_strum_notes", type=int, default=9,

                    help="In arp/mixed mode: chord tones per strum (clamped to available tones in range).")

    ap.add_argument("--argent_strum_roll_frac", type=float, default=0.25,

                    help="In arp/mixed mode: within-strum roll fraction (0..1 of grid_ticks) across notes.")

    ap.add_argument("--argent_strum_dur_steps", type=int, default=6,

                    help="In arp/mixed mode: duration (in grid steps) for strum notes (longer ring).")
    ap.add_argument("--argent_arp1_top_pitch", type=int, default=78, help="Voicing target top pitch hint for ARP1 (higher strings 1-6 bias).")
    ap.add_argument("--argent_arp2_top_pitch", type=int, default=62, help="Voicing target top pitch hint for ARP2 (lower strings 4-9 bias).")
    # Argent keyswitches (output-only, Shreddage 3.5)
    ap.add_argument("--argent_ks_mode", type=str, default="auto", choices=["auto", "mute", "staccato", "none"],
                    help="Which articulation KS to inject before hits (output-only).")
    ap.add_argument("--argent_ks_mute", type=int, default=1, help="Mute articulation keyswitch pitch (default C#-2 ~= MIDI 1).")
    ap.add_argument("--argent_ks_staccato", type=int, default=2, help="Staccato articulation keyswitch pitch (default D-2 ~= MIDI 2).")
    ap.add_argument("--argent_ks_len_ticks", type=int, default=24, help="Length of KS notes in ticks.")
    ap.add_argument("--argent_ks_advance_ticks", type=int, default=10, help="How many ticks before hit to place KS.")
    ap.add_argument("--argent_poly_note", type=int, default=113, help="Behavioral KS note to send at start (F7 = 113).")
    ap.add_argument("--argent_poly_len_ticks", type=int, default=36, help="Length of the behavioral KS note at start.")
    ap.add_argument("--no_argent_poly_note", dest="argent_poly_note_on", action="store_false", help="Disable initial F7 (113) note.")
    ap.set_defaults(argent_poly_note_on=True)

    # Constraints / scoring
    ap.add_argument("--melody_track_regex", type=str, default=None)
    ap.add_argument("--melody_overlap", type=float, default=0.25)
    ap.add_argument("--avoid_melody_active", action="store_true")

    ap.add_argument("--split_on_chord_changes", action="store_true", default=True)
    ap.add_argument("--no_split_on_chord_changes", dest="split_on_chord_changes", action="store_false")

    ap.add_argument("--snap_to_chord_tones", action="store_true", default=False)
    ap.add_argument("--repetition_fix", action="store_true", default=True)
    ap.add_argument("--no_repetition_fix", dest="repetition_fix", action="store_false")

    # Seed pattern knobs
    ap.add_argument("--seed_vel", type=int, default=70)
    ap.add_argument("--seed_ticks", type=int, default=0)

    # Claire keyswitches
    ap.add_argument("--claire_c0_midi", type=int, default=24)

    # Claire "breath" curves (CC1 dynamics + CC11 expression)
    ap.add_argument("--no_claire_cc", action="store_true", help="Disable CC1/CC11 breath curves for Claire winds.")
    ap.add_argument("--claire_cc1_base", type=int, default=35, help="CC1 base value (0-127).")
    ap.add_argument("--claire_cc1_peak", type=int, default=92, help="CC1 peak value (0-127).")
    ap.add_argument("--claire_cc11_base", type=int, default=70, help="CC11 base value (0-127).")
    ap.add_argument("--claire_cc11_peak", type=int, default=112, help="CC11 peak value (0-127).")
    ap.add_argument("--claire_cc_jitter", type=int, default=4, help="Random jitter amount applied to peaks.")
    ap.add_argument("--claire_phrase_gap_steps", type=int, default=2, help="Gap in grid steps to treat as new phrase.")
    ap.add_argument("--claire_force_ks_every_note", type=int, default=1, help="(Claire) If 1, emit articulation keyswitches on every note onset for flute-family instruments (flute/alto/piccolo).")
    ap.add_argument("--claire_allow_vibrato", action="store_true", help="Allow Clarinet vibrato/heavy vibrato selection on long notes.")
    # Claire note shaping (monophonic + longer sustains)
    ap.add_argument("--claire_quantize_mult", type=int, default=1,
                    help="Extra quantize factor for Claire notes vs grid_ticks (e.g. 2 => 1/16 if grid_per_bar=32).")
    ap.add_argument("--claire_min_note_steps", type=int, default=2,
                    help="Minimum Claire note length in grid steps (grid_ticks units). 2 => 1/16 if grid_per_bar=32.")
    ap.add_argument("--claire_max_note_steps", type=int, default=24,
                    help="Maximum Claire sustain length in grid steps before capping (still capped by next note / chord change).")
    ap.add_argument("--no_claire_sustain", action="store_true",
                    help="Disable Claire sustain extension (keeps model durations after base postprocess).")
    ap.add_argument("--no_claire_monophonic", action="store_true",
                    help="Disable Claire monophonic enforcement / de-duplication.")
    # Claire repetition rewrite (avoid hammering same pitch)
    ap.add_argument("--no_claire_hammer_alt", action="store_true",
                    help="Disable Claire repeated-note alternation (up/stay/down/stay on chord tones).")
    ap.add_argument("--claire_hammer_min_run", type=int, default=3,
                    help="Minimum length of a repeated-note run (same pitch) before alternation applies.")
    ap.add_argument("--claire_hammer_gap_steps", type=int, default=1,
                    help="Max gap (in coarse quant steps) between repeated starts to be considered hammering.")


    # Track programs (unique per track by default)
    ap.add_argument("--arp1_program", type=int, default=29)
    ap.add_argument("--arp2_program", type=int, default=30)
    ap.add_argument("--clarinet_program", type=int, default=71)
    ap.add_argument("--oboe_program", type=int, default=68)
    ap.add_argument("--flute_program", type=int, default=73)
    ap.add_argument("--alto_flute_program", type=int, default=74)
    ap.add_argument("--piccolo_program", type=int, default=72)

    # Channels (DAW routing)
    ap.add_argument("--channels", type=str, default="0,1,2,3,4,5,6")

    # Logging / progress
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--log_file", default=None)
    ap.add_argument("--progress", action="store_true", default=True)
    ap.add_argument("--no_progress", dest="progress", action="store_false")
    ap.add_argument("--progress_ncols", type=int, default=0)
    ap.add_argument("--progress_mininterval", type=float, default=0.05)
    ap.add_argument("--write_intermediate", action="store_true",
                    help="Write intermediate .mid after each cascade step (so you can start mixing earlier).")
    ap.add_argument("--intermediate_dir", type=str, default="",
                    help="Directory for intermediate .mid files (default: folder of --out_midi).")
    ap.add_argument("--intermediate_prefix", type=str, default="",
                    help="Filename prefix for intermediate files (default: stem of --out_midi).")
    ap.add_argument("--intermediate_mode", type=str, default="sofar", choices=["sofar","step_only"],
                    help="sofar: include all generated tracks so far; step_only: only current step track (others empty).")

    args = ap.parse_args()
    logger = setup_logging(args.log_level, args.log_file)

    t0 = time.perf_counter()

    progress_enabled = bool(args.progress)
    progress_ncols = None if int(args.progress_ncols or 0) <= 0 else int(args.progress_ncols)
    progress_mininterval = float(args.progress_mininterval or 0.2)

    # Seed (global)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    if device == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        logger.info("TF32 enabled for CUDA matmul/cuDNN")
    src = MidiFile(str(args.in_midi))
    tpb = int(src.ticks_per_beat)
    ts = _first_time_signature(src)
    bt = bar_ticks(tpb, ts)
    grid_ticks = max(1, bt // max(1, int(args.grid_per_bar)))

    logger.info(f"TPB={tpb} | TS={ts[0]}/{ts[1]} | bar_ticks={bt} | grid_ticks={grid_ticks}")

    melody = extract_melody_notes(src, args.melody_track_regex)
    if melody:
        logger.info(f"Extracted melody notes: {len(melody)} (regex={args.melody_track_regex})")

    # Base prompt: chord + piano + bass (keyswitch-stripped)
    prompt_mid = build_prompt_midi(src, logger, disable_keyswitches=True)

    # Use the source song length, not the filtered prompt length, so generation covers the full arrangement.
    song_end = determine_song_end(src)
    if song_end <= 0:
        raise RuntimeError("Prompt MIDI appears empty after filtering. Check track name hints.")

    chords = extract_chord_tones_from_prompt(prompt_mid)
    if not chords:
        logger.warning("Could not extract chord track tones; chord-awareness will be weaker.")

    logger.info(f"Song end tick={song_end} | bars~{math.ceil(song_end / bt)} | chords={len(chords)}")

    # Skriv ut tidslinjen for debugging
    try:
        print_chord_timeline(chords, tpb, ts, logger)
    except Exception as e:
        logger.warning(f"Kunne ikke skrive ut akkord-tidslinje: {e}")

    # Load tokenizer + model
    tok = load_tokenizer(args.tokenizer_json)
    model = load_model(
        args.model_dir, 
        device=device, 
        dtype=args.dtype, 
        attn_impl=args.attn_impl, 
        logger=logger,
        gpu_vram_limit_gb=args.gpu_vram_limit_gb,
        cuda_compile=args.cuda_compile,
        quantization=args.quantization
    )

    # Parse channels
    chan_parts = [p.strip() for p in str(args.channels).split(",") if p.strip()]
    chans = [int(p) for p in chan_parts] if chan_parts else list(range(7))
    while len(chans) < 7:
        chans.append(len(chans))
    chans = chans[:7]

    # Track order = your requested cascade order
    specs: List[InstrumentSpec] = [
        InstrumentSpec("ARGENT_ARP1", int(args.arp1_program), int(chans[0]), lo=40, hi=96, is_claire=False, seed_style="argent_4hit"),
        InstrumentSpec("ARGENT_ARP2", int(args.arp2_program), int(chans[1]), lo=25, hi=84, is_claire=False, seed_style="argent_4hit"),
        InstrumentSpec("CLAIRE_CLARINET", int(args.clarinet_program), int(chans[2]), lo=50, hi=94, is_claire=True, seed_style="sustain"),
        InstrumentSpec("CLAIRE_OBOE", int(args.oboe_program), int(chans[3]), lo=58, hi=92, is_claire=True, seed_style="sustain"),
        InstrumentSpec("CLAIRE_FLUTE", int(args.flute_program), int(chans[4]), lo=60, hi=96, is_claire=True, seed_style="sustain"),
        InstrumentSpec("CLAIRE_ALTO_FLUTE", int(args.alto_flute_program), int(chans[5]), lo=55, hi=88, is_claire=True, seed_style="sustain"),
        InstrumentSpec("CLAIRE_PICCOLO_FLUTE", int(args.piccolo_program), int(chans[6]), lo=74, hi=108, is_claire=True, seed_style="sustain"),
    ]

    # Cascade generation: keep growing prompt_mid by appending previously generated tracks (WITHOUT keyswitches).
    notes_by_name: Dict[str, List[NoteEvent]] = {}
    cc_by_name: Dict[str, List[Tuple[int, int, int]]] = {}

    for idx, spec in enumerate(specs):
        logger.info(f"=== Cascade step {idx+1}/{len(specs)}: GENERATE {spec.name} (program={spec.program}, ch={spec.channel}) ===")

        track_seed = int(args.seed) + (zlib.adler32(spec.name.encode('utf-8')) & 0x7FFFFFFF)

        gen_clean = generate_one_instrument(
            spec=spec,
            prompt_mid_full=prompt_mid,
            chords=chords,
            melody=melody,
            model=model,
            tok=tok,
            tpb=tpb,
            ts=ts,
            song_end=song_end,
            grid_per_bar=int(args.grid_per_bar),
            anchor_ticks=int(args.anchor_ticks),
            argent_notes_per_step=int(args.argent_notes_per_step),
            argent_roll_frac=float(args.argent_roll_frac),
            argent_step_len_mult=float(args.argent_step_len_mult),
            argent_max_dur_steps=int(args.argent_max_dur_steps),
            argent_rhythm_mode=str(args.argent_rhythm_mode),
            argent_arp_density=float(args.argent_arp_density),
            argent_arp_dur_steps=int(args.argent_arp_dur_steps),
            argent_strums_per_bar=int(args.argent_strums_per_bar),
            argent_strum_notes=int(args.argent_strum_notes),
            argent_strum_roll_frac=float(args.argent_strum_roll_frac),
            argent_strum_dur_steps=int(args.argent_strum_dur_steps),
            argent_arp1_top_pitch=int(args.argent_arp1_top_pitch),
            argent_arp2_top_pitch=int(args.argent_arp2_top_pitch),
            bars_per_chunk=int(args.bars_per_chunk),
            context_before_bars=int(args.context_before_bars),
            context_after_bars=int(args.context_after_bars),
            prompt_mode=str(args.prompt_mode),
            prompt_include_generated=bool(args.prompt_include_generated),
            prompt_max_bass_tracks=int(args.prompt_max_bass_tracks),
            prompt_max_piano_tracks=int(args.prompt_max_piano_tracks),
            prompt_exclude_regex=(None if args.prompt_exclude_regex is None else str(args.prompt_exclude_regex)),
            auto_context_clamp=bool(args.auto_context_clamp),
            min_gen_new_tokens=int(args.min_gen_new_tokens),
            samples_per_bar=int(args.samples_per_bar),
            samples_per_chunk=(None if args.samples_per_chunk is None else int(args.samples_per_chunk)),
            max_new_tokens_per_call=int(args.max_new_tokens_per_call),
            gen_batch_size=int(args.gen_batch_size),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            repetition_penalty=float(args.repetition_penalty),
            seed=int(track_seed),
            melody_overlap=float(args.melody_overlap),
            avoid_melody_active=bool(args.avoid_melody_active),
            split_on_chord_changes=bool(args.split_on_chord_changes),
            snap_to_chord_tones=bool(args.snap_to_chord_tones),
            repetition_fix=bool(args.repetition_fix),
            seed_vel=int(args.seed_vel),
            seed_ticks=int(args.seed_ticks),
            progress_enabled=progress_enabled,
            progress_ncols=progress_ncols,
            progress_mininterval=progress_mininterval,
            time_map=str(args.time_map),
            time_map_min_notes=int(args.time_map_min_notes),
            claire_min_notes_per_bar=int(args.claire_min_notes_per_bar),
            bar_from=int(args.bar_from),
            bar_to=int(args.bar_to),
            logger=logger,
            cache_implementation=args.cache_implementation,
            vram_gc_threshold=args.vram_gc_threshold,
            vram_gc_cooldown_chunks=args.vram_gc_cooldown_chunks,
            vram_gc_state=VRAM_GC_STATE,
        )

        # Claire winds: make notes longer / monophonic (post generation)
        if spec.is_claire:
            gen_clean = postprocess_claire_notes(
                gen_clean,
                chords=chords,
                song_end=song_end,
                lo=spec.lo,
                hi=spec.hi,
                grid_ticks=grid_ticks,
                quantize_mult=int(args.claire_quantize_mult),
                min_note_steps=int(args.claire_min_note_steps),
                max_note_steps=int(args.claire_max_note_steps),
                sustain=(not bool(args.no_claire_sustain)),
                monophonic=(not bool(args.no_claire_monophonic)),
                snap_to_chord_tones=bool(args.snap_to_chord_tones),
                hammer_alt=(not bool(getattr(args, "no_claire_hammer_alt", False))),
                hammer_min_run=int(getattr(args, "claire_hammer_min_run", 3)),
                hammer_gap_steps=int(getattr(args, "claire_hammer_gap_steps", 1)),
            )

        # Output version: add articulation keyswitches / CC curves (output-only)
        control_changes: List[Tuple[int, int, int]] = []

        if spec.is_claire:
            gen_out = inject_claire_keyswitches(
                gen_clean,
                spec_name=spec.name,
                claire_c0_midi=int(args.claire_c0_midi),
                grid_ticks=grid_ticks,
                allow_vibrato=bool(args.claire_allow_vibrato),
                phrase_gap_steps=int(args.claire_phrase_gap_steps),
                seed=int(track_seed) + 991,
                force_on_phrase=True,
                force_every_note=(bool(args.claire_force_ks_every_note) and ("FLUTE" in spec.name.upper())),
            )
            if not bool(args.no_claire_cc):
                control_changes = generate_claire_breath_cc(
                    gen_clean,
                    tpb=tpb,
                    grid_ticks=grid_ticks,
                    cc1_base=int(args.claire_cc1_base),
                    cc1_peak=int(args.claire_cc1_peak),
                    cc11_base=int(args.claire_cc11_base),
                    cc11_peak=int(args.claire_cc11_peak),
                    jitter=int(args.claire_cc_jitter),
                    seed=int(track_seed) + int(spec.program) * 97,
                    phrase_gap_steps=int(args.claire_phrase_gap_steps),
                )

        elif spec.name.startswith("ARGENT_ARP"):
            gen_out = inject_argent_keyswitches_and_poly(
                gen_clean,
                grid_ticks=grid_ticks,
                ks_mute=int(args.argent_ks_mute),
                ks_staccato=int(args.argent_ks_staccato),
                poly_note=int(args.argent_poly_note),
                ks_len_ticks=int(args.argent_ks_len_ticks),
                poly_len_ticks=int(args.argent_poly_len_ticks),
                advance_ticks=int(args.argent_ks_advance_ticks),
                mode=str(args.argent_ks_mode),
                include_poly_note=bool(args.argent_poly_note_on),
            )

        else:
            gen_out = gen_clean

        notes_by_name[spec.name] = gen_out
        if control_changes:
            cc_by_name[spec.name] = control_changes

        # Append CLEAN notes to prompt for next instruments (avoid conditioning on keyswitch spam)

        # Optional: write intermediate MIDI after each cascade step (no effect on generation)
        if bool(getattr(args, "write_intermediate", False)):
            try:
                out_dir = (getattr(args, "intermediate_dir", "") or "").strip()
                out_dir = out_dir if out_dir else str(Path(args.out_midi).parent)
                prefix = (getattr(args, "intermediate_prefix", "") or "").strip() or Path(args.out_midi).stem
                safe_name = re.sub(r"[^A-Za-z0-9_\\-]+", "_", str(spec.name))
                step_path = Path(out_dir) / f"{prefix}.step{idx+1:02d}_{safe_name}.mid"
                mode = str(getattr(args, "intermediate_mode", "sofar"))
                if mode == "step_only":
                    tmp_notes = {spec.name: gen_out}
                    tmp_cc = {spec.name: control_changes} if control_changes else {}
                else:
                    tmp_notes = dict(notes_by_name)
                    tmp_notes[spec.name] = gen_out
                    tmp_cc = dict(cc_by_name)
                    if control_changes:
                        tmp_cc[spec.name] = control_changes
                write_output_midi(
                    out_path=step_path,
                    tpb=tpb,
                    tempo_changes=list(getattr(src, "tempo_changes", [])),
                    time_sigs=list(getattr(src, "time_signature_changes", [])),
                    specs=specs,
                    notes_by_name=tmp_notes,
                    control_changes_by_name=tmp_cc,
                    logger=logger,
                )
                logger.info(f"Wrote intermediate MIDI: {step_path}")
            except Exception:
                logger.exception("Failed to write intermediate MIDI (continuing)")
        prompt_mid.instruments.append(instrument_to_miditoolkit(spec, gen_clean))

        logger.info(f"{spec.name}: clean_notes={len(gen_clean)} | out_notes={len(gen_out)} | prompt_tracks_now={len(prompt_mid.instruments)}")

    # Write output midi
    write_output_midi(
        out_path=Path(args.out_midi),
        tpb=tpb,
        tempo_changes=list(getattr(src, "tempo_changes", [])),
        time_sigs=list(getattr(src, "time_signature_changes", [])),
        specs=specs,
        notes_by_name=notes_by_name,
        control_changes_by_name=cc_by_name,
        logger=logger,
    )

    t1 = time.perf_counter()
    logger.info(f"Done. Total runtime: {t1 - t0:.2f}s")




# ----------------------------
# Argent post-processing patch: DYADS in ARP + 9-string STRUM voicings
# ----------------------------
# Shreddage places strum / pick / performance keyswitches above the playable range (e.g. F#5=78+),
# so we keep generated musical notes below that by default.
ARGENT_PLAYABLE_HI_DEFAULT = 77  # highest "safe" musical note (<= F5). Override via spec.hi if you remap KS.

def _argent_hi_music(lo: int, hi: int) -> int:
    return int(max(lo, min(int(hi), int(ARGENT_PLAYABLE_HI_DEFAULT))))

def _argent_pool_from_pcs(pcs: List[int], lo: int, hi: int) -> List[int]:
    pcs_set = {int(pc) % 12 for pc in pcs}
    if not pcs_set:
        return []
    out: List[int] = []
    for p in range(int(lo), int(hi) + 1):
        if (p % 12) in pcs_set:
            out.append(int(p))
    return out

def _argent_pick_evenly(pool: List[int], k: int) -> List[int]:
    if not pool:
        return []
    pool = sorted(set(int(p) for p in pool))
    if k <= 1:
        return [pool[len(pool) // 2]]
    if len(pool) <= k:
        # pad (rare) by octave shifts while staying in range; duplicates are fine in this fallback
        out = pool[:]
        while len(out) < k:
            out.append(out[-1])
        return out[:k]

    idxs = [int(round(i * (len(pool) - 1) / (k - 1))) for i in range(k)]
    # make strictly increasing indices
    chosen: List[int] = []
    last = -1
    for idx in idxs:
        idx = max(idx, last + 1)
        if idx >= len(pool):
            idx = len(pool) - 1
        chosen.append(pool[idx])
        last = idx

    # ensure uniqueness if possible
    used = set()
    fixed: List[int] = []
    for i, p in enumerate(chosen):
        if p not in used:
            fixed.append(p); used.add(p); continue
        # scan nearby for an unused pitch
        base = idxs[i]
        found = None
        for d in range(1, 64):
            for j in (base - d, base + d):
                if 0 <= j < len(pool) and pool[j] not in used:
                    found = pool[j]
                    break
            if found is not None:
                break
        if found is None:
            found = p
        fixed.append(found); used.add(found)
    return fixed[:k]

def _argent_choose_wide_voicing(chord_pitches: List[int], lo: int, hi: int, k: int) -> List[int]:
    if not chord_pitches:
        return []
    pcs = sorted({int(p) % 12 for p in chord_pitches})
    pool = _argent_pool_from_pcs(pcs, lo, hi)
    return _argent_pick_evenly(pool, int(max(1, k)))

def _argent_choose_dyad_companion(main: int, chord_pitches: List[int], lo: int, hi: int, prefer_above: bool) -> Optional[int]:
    if not chord_pitches:
        return None
    pcs = {int(p) % 12 for p in chord_pitches}
    if not pcs:
        return None

    # Try consonant-ish intervals first (within an octave)
    intervals = [7, 4, 3, 5, 9, 10, 12, 2, 6, 8, 11, 1]
    dirs = [1, -1] if prefer_above else [-1, 1]

    for d in dirs:
        for iv in intervals:
            p = int(main + d * iv)
            if lo <= p <= hi and (p % 12) in pcs and p != main and abs(p - main) >= 3:
                return p

    # Fallback: nearest chord tone not equal to main
    pool = _argent_pool_from_pcs(sorted(pcs), lo, hi)
    if not pool:
        return None
    pool2 = [p for p in pool if p != main]
    if not pool2:
        return None
    if prefer_above:
        above = [p for p in pool2 if p > main]
        if above:
            return min(above, key=lambda p: p - main)
    below = [p for p in pool2 if p < main]
    if below:
        return max(below, key=lambda p: main - p)
    return min(pool2, key=lambda p: abs(p - main))

def _argent_resolve_same_pitch_overlaps(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Ensure we don't stack note-ons of the same pitch without an intervening note-off."""
    if not notes:
        return []
    by_pitch: Dict[int, List[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(int(n.pitch), []).append(n)

    out: List[NoteEvent] = []
    for p, arr in by_pitch.items():
        arr = sorted(arr, key=lambda n: (n.start, n.end))
        fixed: List[NoteEvent] = []
        for i, n in enumerate(arr):
            start = int(n.start)
            end = int(n.end)
            if i + 1 < len(arr):
                nxt = arr[i + 1]
                if end > int(nxt.start):
                    end = max(start + 1, int(nxt.start))
            if end > start:
                fixed.append(NoteEvent(start=start, end=end, pitch=int(n.pitch), vel=int(n.vel)))
        out.extend(fixed)

    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    return out

# OVERRIDE: enforce_argent_arp_with_strums now emits DYADS for arp hits and 9-note WIDE STRUMS.
def enforce_argent_arp_with_strums(
    notes: List[NoteEvent],
    chords: List[Tuple[int, List[int]]],
    song_end: int,
    lo: int,
    hi: int,
    grid_ticks: int,
    grid_per_bar: int,
    top_pitch_hint: int,
    rhythm_mode: str = "arp",   # "arp" | "mixed"
    arp_density: float = 0.55,
    arp_dur_steps: int = 2,
    strums_per_bar: int = 2,
    strum_notes: int = 9,
    strum_roll_frac: float = 0.25,
    strum_dur_steps: int = 6,
    range_start: Optional[int] = None,
    range_end: Optional[int] = None,
) -> List[NoteEvent]:
    """
    DYAD ARP:
      - for every arp hit, output 2 notes (dyad) on chord tones, slightly rolled for realism.

    9-STRING STRUM:
      - near the end of each bar, output a wide voicing across the whole playable range (k=strum_notes, default 9),
        rolled as a strum.

    IMPORTANT:
      - skips arp hits on the strum slots to avoid duplicate stacks at the same tick/pitch.
      - keeps musical notes below F#5 (78) by default to avoid Shreddage strum/pick keyswitch ranges.
    """
    if song_end <= 0 or grid_ticks <= 0 or grid_per_bar <= 0:
        return list(notes)

    bar_ticks = int(grid_ticks * grid_per_bar)
    if bar_ticks <= 0:
        return list(notes)

    hi_music = _argent_hi_music(int(lo), int(hi))

    # Infer time-span if not provided
    if range_start is None or range_end is None:
        try:
            min_s = min(int(n.start) for n in notes) if notes else 0
            max_e = max(int(n.end) for n in notes) if notes else 0
        except Exception:
            min_s, max_e = 0, 0
        if range_start is None:
            range_start = (min_s // bar_ticks) * bar_ticks
        if range_end is None:
            range_end = ((max_e + bar_ticks - 1) // bar_ticks) * bar_ticks

    range_start = max(0, min(int(range_start), int(song_end)))
    range_end = max(range_start, min(int(range_end), int(song_end)))

    from collections import defaultdict
    step_pitches = defaultdict(list)  # (bar_idx, step_idx) -> [pitch]
    for n in notes:
        s = quantize_tick(int(n.start), int(grid_ticks))
        b = int(s // bar_ticks)
        si = int((s - b * bar_ticks) // grid_ticks)
        step_pitches[(b, si)].append(int(n.pitch))

    # pick a 16th-based template (keeps old behavior)
    def pick_template() -> List[int]:
        if grid_per_bar >= 32:
            n16 = 16
            interval_16 = max(1, grid_per_bar // n16)
        else:
            n16 = max(4, min(16, grid_per_bar))
            interval_16 = max(1, grid_per_bar // n16)

        mode = str(rhythm_mode).lower()
        slots: List[int] = []
        if mode == "mixed":
            # favor offbeats + some anchors
            for i in range(n16):
                if i % 2 == 1 and random.random() < 0.65:
                    slots.append(i)
                if i % 4 == 0 and random.random() < 0.35:
                    slots.append(i)
        else:
            # "arp": more even grid, slightly sparse
            for i in range(n16):
                if random.random() < 0.55:
                    slots.append(i)

        # Always keep a couple of anchors
        if 0 not in slots:
            slots.append(0)
        if n16 - 2 not in slots:
            slots.append(n16 - 2)

        steps = [min(grid_per_bar - 1, max(0, int(sl * interval_16))) for sl in slots]
        seen = set()
        out_steps = []
        for s in steps:
            if s not in seen:
                seen.add(s)
                out_steps.append(s)
        return out_steps

    out: List[NoteEvent] = []
    n_bars = int(math.ceil(float(song_end) / float(bar_ticks)))
    bar0 = int(range_start) // bar_ticks
    bar1 = (int(range_end) + bar_ticks - 1) // bar_ticks
    bar1 = min(bar1, n_bars)

    arp_idx = 0
    arp_dir = 1
    dyad_flip = False

    # helper for strum slots (match old behavior)
    def strum_steps_for_bar() -> List[int]:
        if strums_per_bar <= 0:
            return []
        if grid_per_bar >= 32:
            n16 = 16
            interval_16 = max(1, grid_per_bar // n16)
        else:
            n16 = max(4, min(16, grid_per_bar))
            interval_16 = max(1, grid_per_bar // n16)
        steps: List[int] = []
        for k in range(int(strums_per_bar)):
            slot = max(0, n16 - int(strums_per_bar) + k)
            s = min(grid_per_bar - 1, int(slot * interval_16))
            steps.append(int(s))
        return sorted(set(steps))

    for b in range(bar0, bar1):
        bar_start = int(b * bar_ticks)
        bar_end = min(int(song_end), int(bar_start + bar_ticks), int(range_end))
        if bar_end <= bar_start:
            continue

        reserved_strum_steps = set(strum_steps_for_bar())

        candidate_steps = pick_template()
        for s in candidate_steps:
            if int(s) in reserved_strum_steps:
                continue  # avoid stacking with strums

            t = int(bar_start + int(s) * int(grid_ticks))
            if t < int(range_start) or t >= int(range_end) or t >= int(bar_end):
                continue
            if random.random() > max(0.0, min(1.0, float(arp_density))):
                continue

            chord_pitches = find_chord_at_time(chords, t)
            if not chord_pitches:
                continue
            chord_tones = [int(p) for p in chord_pitches if int(lo) <= int(p) <= int(hi_music)]
            if not chord_tones:
                chord_tones = [int(p) for p in chord_pitches]
            chord_tones = sorted(set(chord_tones))
            if not chord_tones:
                continue

            src = step_pitches.get((b, int(s)))
            if src:
                main = snap_pitch_to_chord(int(random.choice(src)), chord_tones, int(lo), int(hi_music))
            else:
                main = chord_tones[arp_idx % len(chord_tones)]
                arp_idx += arp_dir
                if len(chord_tones) > 2 and random.random() < 0.08:
                    arp_dir *= -1

            # duration + jitter (keep old feel)
            dur_steps = max(1, int(arp_dur_steps) + random.choice([-1, 0, 0, 1]))
            dur_ticks = int(grid_ticks * dur_steps)
            jitter = random.randint(0, max(0, int(grid_ticks * 0.10)))
            ns = int(t + jitter)
            ne = int(min(bar_end, ns + dur_ticks))
            if ne <= ns:
                continue

            # Build dyad (2 notes) on chord tones
            dyad_flip = not dyad_flip
            prefer_above = ((main < (lo + hi_music) // 2) ^ dyad_flip)
            comp = _argent_choose_dyad_companion(int(main), chord_tones, int(lo), int(hi_music), bool(prefer_above))
            if comp is None or int(comp) == int(main):
                comp = int(main + 7) if int(main + 7) <= int(hi_music) else int(main - 7)

            dyad_roll = max(0, int(grid_ticks * 0.08))
            v_main = 84
            v_comp = max(1, int(v_main * 0.85))
            out.append(NoteEvent(pitch=int(main), start=int(ns), end=int(ne), vel=int(v_main)))
            out.append(NoteEvent(pitch=int(comp), start=int(min(ne - 1, ns + dyad_roll)), end=int(ne), vel=int(v_comp)))

        # Strums at end of bar (wide 9-string voicing)
        for s in strum_steps_for_bar():
            t = int(bar_start + int(s) * int(grid_ticks))
            if t >= bar_end or t < int(range_start) or t >= int(range_end):
                continue
            chord_pitches = find_chord_at_time(chords, t)
            if not chord_pitches:
                continue

            chord_tones = [int(p) for p in chord_pitches if int(lo) <= int(p) <= int(hi_music)]
            if not chord_tones:
                chord_tones = [int(p) for p in chord_pitches]
            chord_tones = sorted(set(chord_tones))
            if not chord_tones:
                continue

            hit = int(max(2, int(strum_notes)))
            picked = _argent_choose_wide_voicing(chord_tones, int(lo), int(hi_music), hit)
            if not picked:
                continue

            roll = max(0, int(grid_ticks * max(0.0, min(1.0, float(strum_roll_frac)))))
            per = max(0, int(roll // max(1, len(picked) - 1)))
            dur_ticks = int(grid_ticks * max(2, int(strum_dur_steps)))
            for i, p in enumerate(picked):
                ns = int(t + i * per)
                ne = int(min(bar_end, ns + dur_ticks))
                if ne > ns:
                    out.append(NoteEvent(pitch=int(p), start=int(ns), end=int(ne), vel=110))

    out = clip_notes(out, int(range_start), int(range_end))
    out.sort(key=lambda n: (n.start, n.pitch, n.end))
    out = _argent_resolve_same_pitch_overlaps(out)
    return out

if __name__ == "__main__":
    main()
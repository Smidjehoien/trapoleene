#!/usr/bin/env python3
"""
prepare_dataset_preprocess_v8_HYBRID_CHUNKED.py

Fixer 0-blocks-problemet ved å støtte MidiTok-versjoner som kun aksepterer file path (str),
ikke miditoolkit.MidiFile objekt.

Ytelse:
- Chunker filer per worker (mindre IPC/pickling)
- preprocess_midi() kjøres én gang per fil (ikke per shift)
- In-place transpose + transpose tilbake (ingen deepcopy)
- Skipper shifts som havner utenfor tokenizer pitch_range
- Bruker per-worker tempfil i /dev/shm (RAM disk) hvis tilgjengelig

Støtter:
- sanitize kvalitetsfilter
- remove_drums
- normalize_keypair
- augment_pitch (shift-list)
- valgfri BPE-trening (kan være 0/off)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
import warnings
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from miditok import REMI  # type: ignore
from miditoolkit import MidiFile  # type: ignore

# Less noise
warnings.filterwarnings("ignore", message="Attribute controls are not compatible")
warnings.filterwarnings("ignore", message="miditok: The `midi_to_tokens` method had been renamed")


PITCH_CLASS_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)

# ---- worker globals ----
_WORKER_TOK: Optional[REMI] = None
_WORKER_ARGS: Optional[dict] = None

_WORKER_FN_NAME: Optional[str] = None          # "encode" / "midi_to_ids" / "midi_to_tokens"
_WORKER_INPUT_MODE: Optional[str] = None       # "obj" or "path"
_WORKER_TOKENS_TO_IDS: bool = False

_WORKER_PITCH_RANGE: Tuple[int, int] = (21, 109)
_WORKER_TMP_DIR: Optional[Path] = None
_WORKER_TMP_MID: Optional[Path] = None


# ---------- logging ----------
def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("prepare_dataset")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.handlers.clear()
    logger.addHandler(h)
    return logger


# ---------- tokenizer save/load ----------
def _save_tokenizer(tok: REMI, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(tok, "save"):
        tok.save(str(path))
        return
    if hasattr(tok, "save_params"):
        tok.save_params(str(path))
        return
    raise RuntimeError("This MidiTok version does not expose save/save_params")


def _build_default_tokenizer(logger: logging.Logger, tokenizer_json: Path) -> REMI:
    TokenizerConfig = None
    try:
        from miditok import TokenizerConfig as _TC
        TokenizerConfig = _TC
    except Exception:
        try:
            from miditok.classes import TokenizerConfig as _TC
            TokenizerConfig = _TC
        except Exception:
            TokenizerConfig = None

    if TokenizerConfig is None:
        raise ImportError("Could not import TokenizerConfig. Update MidiTok.")

    try:
        cfg = TokenizerConfig(
            pitch_range=(21, 109),
            beat_res={(0, 4): 8},
            num_velocities=16,
            use_programs=True,
        )
    except TypeError:
        cfg = TokenizerConfig.from_dict({
            "pitch_range": (21, 109),
            "beat_res": {(0, 4): 8},
            "num_velocities": 16,
            "use_programs": True,
        })

    tok = REMI(tokenizer_config=cfg)
    _save_tokenizer(tok, tokenizer_json)
    logger.info(f"Created new tokenizer at: {tokenizer_json.resolve()}")
    return tok


def load_tokenizer(tokenizer_json: str, logger: logging.Logger) -> REMI:
    p = Path(tokenizer_json.strip().rstrip(","))
    if not p.exists():
        logger.warning(f"Tokenizer JSON not found -> creating new: {p.resolve()}")
        return _build_default_tokenizer(logger, p)
    try:
        if p.stat().st_size == 0:
            raise ValueError("Tokenizer JSON is empty")
        return REMI(params=str(p))
    except Exception as e:
        logger.warning(f"Failed to load tokenizer ({p.resolve()}): {e} -> rebuilding")
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return _build_default_tokenizer(logger, p)


# ---------- utils ----------
def find_midi_files(root_dir: Path) -> List[Path]:
    files = list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))
    return sorted(set(files))


def _flatten_ids(tok: REMI, out) -> List[int]:
    if hasattr(out, "ids"):
        return list(out.ids)
    if isinstance(out, list):
        if len(out) == 0:
            return []
        first = out[0]
        if hasattr(first, "ids"):
            flat: List[int] = []
            for seq in out:
                flat.extend(list(seq.ids))
            return flat
        if isinstance(first, int):
            return list(out)
        if isinstance(first, list):
            flat2: List[int] = []
            for seq in out:
                flat2.extend(seq)
            return flat2
    return list(out)


def _get_tok_pitch_range(tok: REMI) -> Tuple[int, int]:
    pr = None
    cfg = getattr(tok, "config", None)
    if cfg is not None and hasattr(cfg, "pitch_range"):
        pr = getattr(cfg, "pitch_range")
    if pr is None:
        cfg2 = getattr(tok, "tokenizer_config", None)
        if cfg2 is not None and hasattr(cfg2, "pitch_range"):
            pr = getattr(cfg2, "pitch_range")
    if pr is None:
        return (21, 109)
    try:
        return (int(pr[0]), int(pr[1]))
    except Exception:
        return (21, 109)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    if np.all(a == 0) or np.all(b == 0):
        return -1e9
    a, b = a - a.mean(), b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else -1e9


@dataclass
class KeyEstimate:
    tonic_pc: int
    mode: str
    score: float


def estimate_key_ks(pc_hist: np.ndarray) -> KeyEstimate:
    best = KeyEstimate(0, "major", -1e9)
    for tonic in range(12):
        maj_s = _corr(pc_hist, np.roll(KS_MAJOR, tonic))
        min_s = _corr(pc_hist, np.roll(KS_MINOR, tonic))
        if maj_s > best.score:
            best = KeyEstimate(tonic, "major", maj_s)
        if min_s > best.score:
            best = KeyEstimate(tonic, "minor", min_s)
    return best


def global_pitch_class_hist(midi: MidiFile, ignore_drums: bool = True) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float64)
    for inst in midi.instruments:
        if ignore_drums and getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            dur = max(1, int(n.end) - int(n.start))
            hist[int(n.pitch) % 12] += dur * (float(getattr(n, "velocity", 100)) / 127.0)
    return hist


def choose_transpose_semitones(src: KeyEstimate, target_major_pc: int, target_minor_pc: int) -> int:
    tgt = target_major_pc if src.mode == "major" else target_minor_pc
    delta = (tgt - src.tonic_pc) % 12
    return int(delta - 12 if delta > 6 else delta)


def remove_drums_inplace(midi: MidiFile) -> int:
    before = len(midi.instruments)
    midi.instruments = [i for i in midi.instruments if not getattr(i, "is_drum", False)]
    return before - len(midi.instruments)


def has_non_drum_notes(midi: MidiFile) -> bool:
    for inst in midi.instruments:
        if getattr(inst, "is_drum", False):
            continue
        if len(inst.notes) > 0:
            return True
    return False


def transpose_inplace(midi: MidiFile, semitones: int, skip_drums: bool = True) -> None:
    if semitones == 0:
        return
    for inst in midi.instruments:
        if skip_drums and getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            n.pitch = int(max(0, min(127, int(n.pitch) + semitones)))


def _midi_note_stats(midi: MidiFile) -> Tuple[int, int, int]:
    total_notes = 0
    max_end = 0
    inst_with_notes = 0
    for inst in midi.instruments:
        if len(inst.notes) == 0:
            continue
        inst_with_notes += 1
        total_notes += len(inst.notes)
        for n in inst.notes:
            if int(n.end) > max_end:
                max_end = int(n.end)
    return total_notes, max_end, inst_with_notes


def _midi_tempo_stats(midi: MidiFile) -> Tuple[float, float, int]:
    tempos = getattr(midi, "tempo_changes", None) or []
    if len(tempos) == 0:
        return 120.0, 120.0, 0
    vals = [float(t.tempo) for t in tempos if hasattr(t, "tempo")]
    if len(vals) == 0:
        return 120.0, 120.0, 0
    return float(min(vals)), float(max(vals)), int(len(vals))


def _bar_ticks_from_ts(midi: MidiFile) -> Optional[float]:
    tpb = int(getattr(midi, "ticks_per_beat", 480))
    tss = getattr(midi, "time_signature_changes", None) or []
    if len(tss) == 0:
        num, den = 4, 4
    else:
        ts0 = tss[0]
        num = int(getattr(ts0, "numerator", 4))
        den = int(getattr(ts0, "denominator", 4))
    if den <= 0:
        return None
    beats_per_bar = num * (4.0 / den)
    return float(tpb) * beats_per_bar


def is_midi_sane(midi: MidiFile, args) -> bool:
    if len(midi.instruments) == 0:
        return False
    total_notes, _max_end, inst_with_notes = _midi_note_stats(midi)
    if total_notes < int(args["min_notes"]):
        return False

    # Piano sanity: if only 1 track-with-notes, require piano unless allow_single_track_any
    if inst_with_notes == 1 and not bool(args.get("allow_single_track_any", False)):
        inst = None
        for i in midi.instruments:
            if len(i.notes) > 0:
                inst = i
                break
        if inst is None:
            return False
        if getattr(inst, "is_drum", False):
            return False
        if int(getattr(inst, "program", 0)) > 7:
            return False

    return True


def quality_check_or_raise(midi: MidiFile, tok: REMI, args) -> None:
    total_notes, max_end, inst_with_notes = _midi_note_stats(midi)

    if total_notes < args["min_notes"]:
        raise ValueError(f"too_few_notes({total_notes} < {args['min_notes']})")
    if args["max_notes"] is not None and total_notes > args["max_notes"]:
        raise ValueError(f"too_many_notes({total_notes} > {args['max_notes']})")
    if args["max_instruments"] is not None and inst_with_notes > args["max_instruments"]:
        raise ValueError(f"too_many_instruments({inst_with_notes} > {args['max_instruments']})")

    min_bpm, max_bpm, n_tempo = _midi_tempo_stats(midi)
    if (min_bpm < args["min_bpm"]) or (max_bpm > args["max_bpm"]):
        raise ValueError(f"tempo_out_of_range(min={min_bpm:.1f}, max={max_bpm:.1f})")
    if args["max_tempo_changes"] is not None and n_tempo > args["max_tempo_changes"]:
        raise ValueError(f"too_many_tempo_changes({n_tempo} > {args['max_tempo_changes']})")

    bar_ticks = _bar_ticks_from_ts(midi)
    if bar_ticks and bar_ticks > 0 and max_end > 0:
        bars = max_end / bar_ticks
        if bars < args["min_bars"]:
            raise ValueError(f"too_short({bars:.2f} bars < {args['min_bars']})")
        notes_per_bar = total_notes / max(1e-6, bars)
        if args["max_notes_per_bar"] is not None and notes_per_bar > args["max_notes_per_bar"]:
            raise ValueError(f"too_dense({notes_per_bar:.1f} notes/bar > {args['max_notes_per_bar']})")

    if args.get("drop_unsupported_time_sigs", False) and hasattr(tok, "has_midi_time_signatures_not_in_vocab"):
        if tok.has_midi_time_signatures_not_in_vocab(midi):
            raise ValueError("unsupported_time_signature")


def _parse_shift_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else [0]


def _get_pitch_minmax_non_drum(midi: MidiFile) -> Optional[Tuple[int, int]]:
    mn = 999
    mx = -999
    found = False
    for inst in midi.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            p = int(n.pitch)
            if p < mn:
                mn = p
            if p > mx:
                mx = p
            found = True
    if not found:
        return None
    return (mn, mx)


# ---------- worker encoding plumbing ----------
def _probe_tokenizer_io(tok: REMI, tmp_mid: Path) -> Tuple[str, str, bool]:
    """
    Return (fn_name, input_mode, tokens_to_ids_flag)
    input_mode: "obj" or "path"
    """
    probe_midi = MidiFile()
    probe_midi.ticks_per_beat = 480

    # Make a minimal valid piano note
    from miditoolkit import Instrument, Note  # type: ignore
    inst = Instrument(program=0, is_drum=False, name="probe_piano")
    inst.notes.append(Note(velocity=80, pitch=60, start=0, end=240))
    probe_midi.instruments.append(inst)

    # 1) Try MidiFile-object input
    for fn_name in ("encode", "midi_to_ids", "midi_to_tokens"):
        fn = getattr(tok, fn_name, None)
        if fn is None:
            continue
        try:
            out = fn(probe_midi)  # type: ignore
            tokens_to_ids = (fn_name == "midi_to_tokens" and hasattr(tok, "tokens_to_ids"))
            _ = _flatten_ids(tok, tok.tokens_to_ids(out) if tokens_to_ids else out)  # type: ignore
            return fn_name, "obj", tokens_to_ids
        except Exception:
            pass

    # 2) Try path input
    try:
        tmp_mid.parent.mkdir(parents=True, exist_ok=True)
        probe_midi.dump(str(tmp_mid))
    except Exception:
        # if dump fails, still try /tmp fallback later
        pass

    for fn_name in ("encode", "midi_to_ids", "midi_to_tokens"):
        fn = getattr(tok, fn_name, None)
        if fn is None:
            continue
        try:
            out = fn(str(tmp_mid))  # type: ignore
            tokens_to_ids = (fn_name == "midi_to_tokens" and hasattr(tok, "tokens_to_ids"))
            _ = _flatten_ids(tok, tok.tokens_to_ids(out) if tokens_to_ids else out)  # type: ignore
            return fn_name, "path", tokens_to_ids
        except Exception:
            pass

    raise RuntimeError("Could not find a working tokenizer IO method (obj or path).")


def _worker_init(tok_path: str, args_dict: dict) -> None:
    global _WORKER_TOK, _WORKER_ARGS
    global _WORKER_FN_NAME, _WORKER_INPUT_MODE, _WORKER_TOKENS_TO_IDS
    global _WORKER_PITCH_RANGE, _WORKER_TMP_DIR, _WORKER_TMP_MID

    warnings.filterwarnings("ignore")

    _WORKER_ARGS = args_dict
    _WORKER_TOK = REMI(params=tok_path)
    _WORKER_PITCH_RANGE = _get_tok_pitch_range(_WORKER_TOK)

    # Prefer /dev/shm for temp if exists
    base_tmp = "/dev/shm" if Path("/dev/shm").exists() else None
    td = tempfile.mkdtemp(prefix="midiprep_", dir=base_tmp)
    _WORKER_TMP_DIR = Path(td)
    _WORKER_TMP_MID = _WORKER_TMP_DIR / "tmp.mid"

    fn_name, mode, t2i = _probe_tokenizer_io(_WORKER_TOK, _WORKER_TMP_MID)
    _WORKER_FN_NAME = fn_name
    _WORKER_INPUT_MODE = mode
    _WORKER_TOKENS_TO_IDS = t2i


def _encode_midi_to_ids(midi: MidiFile) -> List[int]:
    global _WORKER_TOK, _WORKER_FN_NAME, _WORKER_INPUT_MODE, _WORKER_TOKENS_TO_IDS, _WORKER_TMP_MID
    if _WORKER_TOK is None or _WORKER_FN_NAME is None or _WORKER_INPUT_MODE is None:
        raise RuntimeError("Worker tokenizer not initialized")

    fn = getattr(_WORKER_TOK, _WORKER_FN_NAME, None)
    if fn is None:
        raise RuntimeError(f"Tokenizer missing method: {_WORKER_FN_NAME}")

    if _WORKER_INPUT_MODE == "obj":
        out = fn(midi)  # type: ignore
    else:
        # path mode fallback
        assert _WORKER_TMP_MID is not None
        midi.dump(str(_WORKER_TMP_MID))
        out = fn(str(_WORKER_TMP_MID))  # type: ignore

    if _WORKER_FN_NAME == "midi_to_tokens" and _WORKER_TOKENS_TO_IDS:
        out = _WORKER_TOK.tokens_to_ids(out)  # type: ignore

    return _flatten_ids(_WORKER_TOK, out)


# ---------- chunk worker ----------
def _process_chunk(file_paths: List[str], shifts: List[int]) -> Tuple[bool, array, int, int, Dict[str, int]]:
    """
    Returns:
      (success_any, ids_array, ok_files, skip_files, reason_counts)
    """
    global _WORKER_TOK, _WORKER_ARGS, _WORKER_PITCH_RANGE
    if _WORKER_TOK is None or _WORKER_ARGS is None:
        return (False, array("I"), 0, len(file_paths), {"worker_not_inited": len(file_paths)})

    args = _WORKER_ARGS
    pr_lo, pr_hi = _WORKER_PITCH_RANGE

    eos = getattr(_WORKER_TOK, "eos_token_id", None)

    out_ids = array("I")
    ok = 0
    skip = 0
    reasons: Dict[str, int] = {}

    for fp in file_paths:
        try:
            midi = MidiFile(fp)

            if args["sanitize"]:
                if not is_midi_sane(midi, args):
                    skip += 1
                    reasons["sanity_fail"] = reasons.get("sanity_fail", 0) + 1
                    continue

                if args["remove_drums"] and has_non_drum_notes(midi):
                    remove_drums_inplace(midi)

                quality_check_or_raise(midi, _WORKER_TOK, args)
            else:
                if args["remove_drums"] and has_non_drum_notes(midi):
                    remove_drums_inplace(midi)

            # normalize_keypair base shift
            base_shift = 0
            if args["normalize_keypair"]:
                key = estimate_key_ks(global_pitch_class_hist(midi, ignore_drums=True))
                tgt_maj = PITCH_CLASS_NAMES_SHARP.index(args["target_major"])
                tgt_min = PITCH_CLASS_NAMES_SHARP.index(args["target_minor"])
                base_shift = choose_transpose_semitones(key, tgt_maj, tgt_min)

            if base_shift != 0:
                transpose_inplace(midi, base_shift, skip_drums=True)

            # preprocess_midi once per file (if available)
            if args["sanitize"] and hasattr(_WORKER_TOK, "preprocess_midi"):
                _WORKER_TOK.preprocess_midi(midi)

            mm = _get_pitch_minmax_non_drum(midi)
            if mm is None:
                skip += 1
                reasons["no_non_drum_notes"] = reasons.get("no_non_drum_notes", 0) + 1
                continue

            pmin, pmax = mm

            got_any = False

            for sh in shifts:
                if (pmin + sh) < pr_lo or (pmax + sh) > pr_hi:
                    continue

                if sh != 0:
                    transpose_inplace(midi, sh, skip_drums=True)

                try:
                    ids = _encode_midi_to_ids(midi)
                    if ids:
                        out_ids.fromlist(ids)
                        if isinstance(eos, int):
                            out_ids.append(int(eos))
                        got_any = True
                except TypeError:
                    # This is what killed your previous run. If it happens here, count it.
                    reasons["encode_typeerror"] = reasons.get("encode_typeerror", 0) + 1
                except Exception as e:
                    key = f"encode_fail:{type(e).__name__}"
                    reasons[key] = reasons.get(key, 0) + 1
                finally:
                    if sh != 0:
                        transpose_inplace(midi, -sh, skip_drums=True)

            if got_any:
                ok += 1
            else:
                skip += 1
                reasons["all_shifts_failed"] = reasons.get("all_shifts_failed", 0) + 1

        except Exception as e:
            skip += 1
            key = f"file_failed:{type(e).__name__}"
            reasons[key] = reasons.get(key, 0) + 1

    return (len(out_ids) > 0, out_ids, ok, skip, reasons)


# ---------- packing ----------
def pack_to_blocks_from_array(stream: array, block_size: int) -> np.ndarray:
    n_tokens = len(stream)
    n_blocks = n_tokens // block_size
    if n_blocks <= 0:
        return np.zeros((0, block_size), dtype=np.int32)

    take = n_blocks * block_size
    buf = np.frombuffer(stream, dtype=np.uint32, count=take)
    blocks = buf.astype(np.int32, copy=False).reshape(n_blocks, block_size)
    return blocks


def chunk_list(xs: List[str], chunk_size: int) -> List[List[str]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


# ---------- main build_split ----------
def build_split(args_ns, split_name: str, midi_root_dir: Path, out_dir: Path, tok: REMI, logger: logging.Logger) -> None:
    if not midi_root_dir.exists():
        raise FileNotFoundError(f"{split_name} dir not found: {midi_root_dir.resolve()}")

    midi_files = find_midi_files(midi_root_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDIs found in: {midi_root_dir.resolve()}")

    if args_ns.shuffle_files:
        random.Random(args_ns.seed).shuffle(midi_files)
    if args_ns.max_files:
        midi_files = midi_files[:args_ns.max_files]

    logger.info(f"{split_name}: Fant {len(midi_files)} originalfiler.")

    tok_path = str(Path(args_ns.tokenizer_json).resolve())

    # Convert args to pure dict for worker pickling
    args_dict = {
        "sanitize": bool(args_ns.sanitize),
        "remove_drums": bool(args_ns.remove_drums),
        "normalize_keypair": bool(args_ns.normalize_keypair),
        "target_major": str(args_ns.target_major),
        "target_minor": str(args_ns.target_minor),

        "min_notes": int(args_ns.min_notes),
        "max_notes": int(args_ns.max_notes) if args_ns.max_notes is not None else None,
        "max_instruments": int(args_ns.max_instruments) if args_ns.max_instruments is not None else None,
        "min_bars": float(args_ns.min_bars),
        "max_notes_per_bar": float(args_ns.max_notes_per_bar) if args_ns.max_notes_per_bar is not None else None,
        "min_bpm": float(args_ns.min_bpm),
        "max_bpm": float(args_ns.max_bpm),
        "max_tempo_changes": int(args_ns.max_tempo_changes) if args_ns.max_tempo_changes is not None else None,
        "drop_unsupported_time_sigs": bool(args_ns.drop_unsupported_time_sigs),

        "allow_single_track_any": bool(args_ns.allow_single_track_any),
    }

    # Shifts
    shifts = _parse_shift_list(args_ns.augment_shifts) if (split_name == "train" and args_ns.augment_pitch) else [0]

    workers = int(args_ns.workers) if args_ns.workers is not None else (os.cpu_count() or 4)
    file_chunk_size = int(args_ns.file_chunk_size)

    logger.info(
        f"⚡ Hybrid tokenize: workers={workers}, file_chunk_size={file_chunk_size}, shifts={len(shifts)} "
        f"(augment={'ON' if (split_name=='train' and args_ns.augment_pitch) else 'OFF'}), sanitize={'ON' if args_ns.sanitize else 'OFF'}"
    )

    # Chunk file paths (strings) to reduce IPC overhead
    file_paths = [str(p) for p in midi_files]
    chunks = chunk_list(file_paths, file_chunk_size)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    all_ids = array("I")
    total_ok = 0
    total_skip = 0
    reason_counts: Dict[str, int] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(tok_path, args_dict),
    ) as ex:
        futures = [ex.submit(_process_chunk, ch, shifts) for ch in chunks]

        for fu in tqdm(as_completed(futures), total=len(futures), desc=f"Tokenize:{split_name}"):
            success_any, ids_arr, ok, skip, reasons = fu.result()
            if success_any and len(ids_arr) > 0:
                all_ids.extend(ids_arr)
            total_ok += ok
            total_skip += skip
            for k, v in reasons.items():
                reason_counts[k] = reason_counts.get(k, 0) + int(v)

    if total_skip > 0:
        logger.warning(f"{split_name}: Skipped {total_skip}/{len(midi_files)} files.")
        top = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
        for r, c in top:
            logger.warning(f"  {c:6d}  {r}")

    blocks = pack_to_blocks_from_array(all_ids, args_ns.block_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split_name}_input_ids.npy", blocks)

    meta = {
        "split": split_name,
        "num_files_scanned": len(midi_files),
        "files_passed": int(total_ok),
        "files_skipped": int(total_skip),
        "block_size": int(args_ns.block_size),
        "num_blocks": int(blocks.shape[0]),
        "total_token_ids": int(len(all_ids)),
        "vocab_size": int(len(tok)) if hasattr(tok, "__len__") else -1,
        "sanitize": bool(args_ns.sanitize),
        "augment_pitch": bool(split_name == "train" and args_ns.augment_pitch),
        "augment_shifts": shifts,
        "remove_drums": bool(args_ns.remove_drums),
        "normalize_keypair": bool(args_ns.normalize_keypair),
        "file_chunk_size": file_chunk_size,
        "workers": workers,
        "top_skip_reasons": dict(sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]),
    }
    (out_dir / f"{split_name}_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"✅ {split_name}: wrote {blocks.shape[0]} blocks -> {out_dir / f'{split_name}_input_ids.npy'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="data/midi/train")
    p.add_argument("--valid_dir", default="data/midi/valid")
    p.add_argument("--out_dir", default="data/packed")
    p.add_argument("--tokenizer_json", default="out/tokenizer.json")
    p.add_argument("--block_size", type=int, default=4096)

    # Augment
    p.add_argument("--augment_pitch", action="store_true")
    p.add_argument("--augment_shifts", default="-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6")

    # Options
    p.add_argument("--remove_drums", action="store_true")
    p.add_argument("--normalize_keypair", action="store_true")
    p.add_argument("--target_major", default="C")
    p.add_argument("--target_minor", default="A")

    # Sanitize / quality gates
    p.add_argument("--sanitize", action="store_true")
    p.add_argument("--min_notes", type=int, default=30)
    p.add_argument("--max_notes", type=int, default=30000)
    p.add_argument("--max_instruments", type=int, default=48)
    p.add_argument("--min_bars", type=float, default=2.0)
    p.add_argument("--max_notes_per_bar", type=float, default=500.0)
    p.add_argument("--min_bpm", type=float, default=40.0)
    p.add_argument("--max_bpm", type=float, default=260.0)
    p.add_argument("--max_tempo_changes", type=int, default=80)
    p.add_argument("--drop_unsupported_time_sigs", action="store_true")

    # single track rule
    p.add_argument("--allow_single_track_any", action="store_true")

    # runtime
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shuffle_files", action="store_true")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--file_chunk_size", type=int, default=64, help="Files per worker task (reduces IPC overhead)")

    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.log_level)

    tok = load_tokenizer(args.tokenizer_json, logger)
    out_dir = Path(args.out_dir)

    if Path(args.train_dir).exists():
        build_split(args, "train", Path(args.train_dir), out_dir, tok, logger)
    if Path(args.valid_dir).exists():
        build_split(args, "valid", Path(args.valid_dir), out_dir, tok, logger)

    logger.info("Dataset packing complete!")


if __name__ == "__main__":
    main()

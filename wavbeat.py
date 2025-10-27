#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from scipy import signal
import glob
import os
import argparse

# === global mix knobs ===
FILTER_COVERAGE = 2.0      # 1.0 = original scheduling; >1.0 = more of the song under LP/BP
FILTER_WET = 1.0           # 0..1, blend amount for filtered signal
FILTER_GLOBAL = None       # None | "lp" | "bp"  -> force a single LP/BP across ALL samples

# Defaults for the forced-all filter (used when FILTER_GLOBAL != None)
FILTER_LP_CUTOFF_HZ = 2900.0
FILTER_BP_LOW_HZ = 200.0
FILTER_BP_HIGH_HZ = 3900.0

# One-shot loudness
KICK_GAIN = 1.25
HAT_GAIN = 1.00
CLAP_GAIN = 1.15
CLAP_DEV_GAIN = 0.75

# =========================
# IO / Utility
# =========================

def get_next_filename(base_name: str) -> str:
    pattern = f"{base_name}_chopped_*.wav"
    existing = glob.glob(pattern)
    if not existing:
        return f"{base_name}_chopped_1.wav"
    nums = []
    for p in existing:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            nums.append(int(stem.rsplit("_", 1)[-1]))
        except Exception:
            pass
    n = max(nums) + 1 if nums else 1
    return f"{base_name}_chopped_{n}.wav"

def load_audio_mono(path: str):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    # trim digital silence
    nz = np.where(np.abs(x) > 1e-6)[0]
    if nz.size > 0:
        x = x[nz[0]:nz[-1]+1]
    return x, sr

def normalize(x: np.ndarray, peak: float = 0.98):
    m = float(np.max(np.abs(x)) + 1e-12)
    return (x / m) * peak

def resample_speed(x: np.ndarray, speed: float):
    speed = float(speed)
    if speed == 1.0:
        return x
    speed = float(np.clip(speed, 0.1, 2.0))
    up = 1000
    down = int(round(up * speed))  # speed>1 -> fewer samples (faster)
    y = signal.resample_poly(x, up, down)
    return y.astype(np.float32)

# =========================
# Reverb (VERY SUBTLE)
# =========================

def simple_reverb(x: np.ndarray, sr: int, mix=0.10, time_s=0.15, fb=0.28):
    """
    Plate-style reverb (Freeverb-ish) using an impulse-response built from:
      - parallel damped combs (density)
      - serial allpasses (diffusion)
      - pre-delay
    Kept same signature as before; 'time_s' ~ overall size, 'fb' ~ room/decay.
    """
    import numpy as np
    from scipy import signal

    # -------- knobs (internals so signature stays the same) --------
    room = float(np.clip(fb, 0.05, 0.95))          # feedback/decay
    damp = 0.35                                     # high-freq absorption in comb loops
    pre_delay_ms = 18.0                             # early reflection delay
    # scale classic comb/allpass times with 'time_s' (0.35 ≈ medium room)
    tscale = max(0.08, float(time_s)) / 0.35

    # -------- build a compact IR (fast) --------
    def _build_ir(sr: int) -> np.ndarray:
        # IR length ~ depends on "room" and "size"
        rt_tail = 0.6 + 2.2 * (tscale * room)       # seconds (rough heuristic)
        L = int(sr * min(4.0, max(0.5, rt_tail)))   # clamp for safety

        imp = np.zeros(L, dtype=np.float64)
        imp[0] = 1.0

        # comb delays in milliseconds (scaled)
        comb_ms = np.array([29.7, 37.1, 41.1, 43.7]) * tscale
        comb_d = np.maximum(8, (comb_ms * 1e-3 * sr).astype(int))

        # allpass stages (fixed-ish; tiny scale helps tiny rooms)
        ap_ms = (np.array([5.0, 1.7]) * (0.75 + 0.25 * tscale))
        ap_d  = np.maximum(4, (ap_ms * 1e-3 * sr).astype(int))
        ap_g  = [0.7, 0.5]

        # Parallel damped combs
        y = np.zeros(L, dtype=np.float64)
        for d in comb_d:
            c = np.zeros(L, dtype=np.float64)
            lp = 0.0  # one-pole lowpass in feedback loop
            g = room
            for n in range(d, L):
                # feedback sample
                w = c[n - d]
                lp += damp * (w - lp)              # lowpass in loop
                c[n] = imp[n] + g * lp
            y += c

        # Serial allpasses for diffusion
        for d, g in zip(ap_d, ap_g):
            z = np.zeros(L, dtype=np.float64)
            for n in range(d, L):
                z[n] = -g * y[n] + y[n - d] + g * z[n - d]
            y = z

        # pre-delay by padding IR head with zeros
        pd = int((pre_delay_ms * 1e-3) * sr)
        ir = np.pad(y, (pd, 0))

        # normalise IR to sane peak
        peak = float(np.max(np.abs(ir)) + 1e-12)
        ir = ir / peak * 0.9
        return ir

    x64 = x.astype(np.float64, copy=False)
    ir = _build_ir(sr)

    # FFT convolution (efficient for long tails)
    y = signal.fftconvolve(x64, ir, mode="full")[:len(x64)]

    # wet/dry blend WITHOUT tanh
    wet = float(np.clip(mix, 0.0, 1.0))
    out = (1.0 - wet) * x64 + wet * y

    return out.astype(np.float32)

def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def wider_glue_reverb(x: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
    """
    Fuller-band 'glue' reverb:
    - Uses the existing simple_reverb engine to avoid tinny HF-only tails.
    - Adds low-mid body to the wet via a broad band-pass blend.
    'strength' ≈ your --reverb knob (0..2).
    """
    s = float(np.clip(strength, 0.0, 2.0))
    mix   = np.clip(0.20 + 0.12 * s, 0.0, 0.48)   # wetter by default
    time_s = 0.40 + 0.15 * min(s, 1.5)            # slightly larger room
    fb    = 0.32 + 0.13 * min(s, 1.5)             # slightly longer decay

    # get a 100% wet tail, then shape it and blend ourselves
    wet_only = simple_reverb(x, sr, mix=1.0, time_s=time_s, fb=fb)

    # add body to the wet (broad low-mid emphasis) to avoid 'tinny' feel
    body = _butter_bp(wet_only, sr, low=120.0, high=3800.0, order=3)
    wet_shaped = 0.70 * wet_only + 0.30 * body

    y = (1.0 - mix) * x.astype(np.float64) + mix * wet_shaped.astype(np.float64)
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 1.0:
        y /= peak
    return y.astype(np.float32)

def wider_glue_reverb_wet(x: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
    """100% wet version of wider_glue_reverb (no dry), for bus sends."""
    s = float(np.clip(strength, 0.0, 2.0))
    time_s = 0.40 + 0.15 * min(s, 1.5)
    fb     = 0.32 + 0.13 * min(s, 1.5)

    wet_only = simple_reverb(x, sr, mix=1.0, time_s=time_s, fb=fb)
    body = _butter_bp(wet_only, sr, low=120.0, high=3800.0, order=3)
    wet_shaped = 0.65 * wet_only + 0.35 * body
    return wet_shaped.astype(np.float32)


def apply_bus_glue_reverb(mix: np.ndarray, sr: int, *, send=0.08, return_db=-12.0, strength=1.0) -> np.ndarray:
    """
    Small reverb send on the stereo bus.
    Increase 'send' a hair (e.g., 0.08 -> 0.10) for a little more verb.
    """
    dry = mix.astype(np.float32)
    aux_in = (dry * float(np.clip(send, 0.0, 1.0))).astype(np.float32)
    aux_wet = wider_glue_reverb_wet(aux_in, sr, strength=strength)

    y = dry.astype(np.float64) + _db_to_lin(return_db) * aux_wet.astype(np.float64)
    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > 1.0:
        y /= peak
    return y.astype(np.float32)

def apply_global_loop_reverb(chop_track: np.ndarray, sr: int, *,
                             wet_db: float = -6.0,
                             strength: float = 1.3,
                             tail_secs: float = 0.0) -> np.ndarray:
    """Global reverb for the whole backing loop, with optional end ring-out."""
    dry = chop_track.astype(np.float64)
    pad = np.zeros(int(max(0.0, tail_secs) * sr), dtype=np.float64)
    dry_padded = np.concatenate([dry, pad])

    wet = wider_glue_reverb_wet(dry_padded.astype(np.float32), sr, strength=strength).astype(np.float64)
    out = dry_padded + _db_to_lin(wet_db) * wet

    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)

# =========================
# Very light granular (texture only)
# =========================

def granular_synthesis(audio, sr, intensity=0.12):
    """
    Harmonic, smooth granular texture (keeps duration).
    - Coherent forward read-head with light jitter (no random jumps)
    - 50% overlap-add with Hann window (no "scramble"/gaps)
    - Consonant pitch offsets with mild micro-detune, then re-timed to grain_len
    """
    if len(audio) == 0:
        return audio

    rng = np.random.default_rng()
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # ~35..80 ms grains depending on intensity
    grain_ms = 0.035 + 0.045 + 0.025 * intensity
    grain_len = max(32, int(grain_ms * sr))
    hop = max(16, grain_len // 2)  # 50% overlap
    n_grains = int(np.ceil(len(audio) / hop)) + 2

    win = np.hanning(grain_len).astype(np.float64)
    out = np.zeros(len(audio) + grain_len, dtype=np.float64)

    # Consonant intervals (semitones) with heavy weight on unison
    allowed = np.array([0, 7, 12, -5], dtype=float)
    weights = np.array([0.78, 0.12, 0.07, 0.03], dtype=float)
    weights /= weights.sum()

    head = 0
    jitter = int(0.15 * grain_len)  # gentle wander
    for g in range(n_grains):
        # Coherent source window
        center = head + rng.integers(-jitter, jitter + 1)
        start = int(np.clip(center, 0, max(0, len(audio) - grain_len)))
        grain = audio[start:start + grain_len].astype(np.float64)
        grain = grain - grain.mean()  # tiny DC guard

        # Consonant pitch + micro-detune
        semi = float(rng.choice(allowed, p=weights))
        cents = float(rng.normal(0.0, 3.0))   # ± a few cents
        ratio = 2.0 ** ((semi + cents / 100.0) / 12.0)

        # Pitch by resampling, then re-time to original grain_len for SOLA-like OLA
        pitched = signal.resample(grain, max(4, int(round(grain_len * ratio))))
        if len(pitched) != grain_len:
            pitched = signal.resample(pitched, grain_len)

        pos = g * hop
        if pos >= len(out):
            break
        end = min(len(out), pos + grain_len)
        out[pos:end] += pitched[:end - pos] * win[:end - pos]

        head += hop
        if head >= len(audio):
            break

    # Soft normalisation + clip
    peak = float(np.max(np.abs(out)) + 1e-12)
    out = np.tanh(0.9 * out / peak)
    return out[:len(audio)].astype(np.float32)


# =========================
# One-shots (kick / hat)
# =========================

def _butter_hp(x, sr, cutoff=9000.0, order=5):
    # Clamp cutoff to be safely below Nyquist frequency (0.5 * sr)
    safe_cutoff = min(float(cutoff), 0.49 * sr)
    b, a = signal.butter(order, safe_cutoff / (0.5 * sr), btype='highpass')
    return signal.lfilter(b, a, x)

def _butter_lp(x, sr, cutoff=140.0, order=4):
    # Clamp cutoff to be safely below Nyquist frequency (0.5 * sr)
    safe_cutoff = min(float(cutoff), 0.49 * sr)
    b, a = signal.butter(order, safe_cutoff / (0.5 * sr), btype='lowpass')
    return signal.lfilter(b, a, x)

def make_kick_from_slice(x, sr):
    if len(x) == 0:
        return x
    dur = int(0.11 * sr)  # ~110 ms
    g = x[:dur].astype(np.float64)
    g = _butter_lp(g, sr, cutoff=np.random.uniform(90.0, 160.0))
    t = np.linspace(0.0, 1.0, len(g))
    env = np.exp(-6.5 * t)
    g *= env
    n = min(16, len(g))
    g[:n] += 0.4 * g[:n]
    g /= (np.max(np.abs(g)) + 1e-9)
    return g.astype(np.float32)

def make_hat_variants_from_slice(x, sr):
    """Return (closed_hat, open_hat) variants."""
    if len(x) == 0:
        return x, x
    # Closed
    c_dur = int(np.random.uniform(0.028, 0.050) * sr)
    c = x[:c_dur].astype(np.float64)
    c = _butter_hp(c, sr, cutoff=np.random.uniform(9500.0, 12000.0))
    t = np.linspace(0.0, 1.0, len(c))
    c *= np.exp(-16.0 * t)
    if len(c) > 8:
        k = np.random.randint(3, 5)
        c = signal.resample(c[::k], len(c))
    c /= (np.max(np.abs(c)) + 1e-9)

    # Open (longer, breathier)
    o_dur = int(np.random.uniform(0.075, 0.120) * sr)
    o = x[:o_dur].astype(np.float64)
    o = _butter_hp(o, sr, cutoff=np.random.uniform(8500.0, 11000.0))
    t = np.linspace(0.0, 1.0, len(o))
    o *= np.exp(-9.0 * t)
    if len(o) > 8:
        k = np.random.randint(2, 4)
        o = signal.resample(o[::k], len(o))
    o /= (np.max(np.abs(o)) + 1e-9)

    return c.astype(np.float32), o.astype(np.float32)

# =========================
# Events (kicks/hats) on triplet grid
# =========================

def generate_regular_kick_onsets(sr, bpm, length_samples):
    """4-on-the-floor: kicks at the start of every beat."""
    beat = 60.0 / bpm
    total_beats = int(np.ceil(length_samples / (beat * sr)))
    positions = [int(b * beat * sr) for b in range(total_beats)]
    return np.array(positions, dtype=int)

def generate_sparse_shuffled_hat_events(sr, bpm, length_samples,
                                        base_density=0.35, shuffle=0.18, rng=None):
    """
    Sparse hats on a shuffled triplet grid.
    Returns list of (position_samples, is_open_bool, gain_float).
    """
    if rng is None:
        rng = np.random.default_rng()

    beat = 60.0 / bpm
    tri = beat / 3.0
    total_beats = int(np.ceil(length_samples / (beat * sr)))
    events = []

    # per-triplet bias (prefer the last, swung one)
    hat_bias = [0.12, 0.22, 0.52]  # sum <= 1, scaled by base_density and bar_scale

    for b in range(total_beats):
        bar_idx = b // 4
        # bar-level sparsity variance
        bar_scale = 0.5 + 0.5 * rng.random()  # 0.5..1.0
        # occasional very sparse bars
        if rng.random() < 0.18:
            bar_scale *= 0.6

        offsets = [0.0, (1.0 + shuffle) * tri, (2.0 - shuffle) * tri]
        for i, off in enumerate(offsets):
            pos = int((b * beat + off) * sr)
            if pos >= length_samples:
                break
            p = base_density * hat_bias[i] * bar_scale
            if rng.random() < p:
                is_open = rng.random() < 0.18  # 18% open
                gain = (0.38 + 0.45 * rng.random()) * (1.2 if is_open else 1.0)
                events.append((pos, is_open, float(gain)))
    return events

def _tile_repeat(x: np.ndarray, total_len: int) -> np.ndarray:
    """Tile 1D array to at least total_len and crop."""
    reps = int(np.ceil(total_len / max(1, len(x))))
    return np.tile(x, reps)[:total_len].astype(np.float32)


def build_converging_looped_chops(
    slice_lib, sr, bpm, *, bars=8, subdiv=2, density=0.6,
    allow_double=True, granular_chance=0.01, xfade_ms=12,
    loop_bars=2, converge_bars=2
):
    """
    1) Build a 2-bar (configurable) MOTIF of legato chops.
    2) Tile that motif across the song.
    3) Optionally crossfade the first N bars from a 'free' random take into the loop.
    """
    assert subdiv >= 1
    beat_samps = int(sr * 60.0 / bpm)
    grid = beat_samps // subdiv

    # song + motif lengths
    bars = max(loop_bars, int(bars))
    out_len = 4 * bars * beat_samps
    motif_len = 4 * loop_bars * beat_samps

    rng = np.random.default_rng()
    xfade = max(0, int((xfade_ms / 1000.0) * sr))
    granular_used = 0

    # --- build motif ---
    motif = np.zeros(motif_len, dtype=np.float32)
    cursor = 0
    while cursor < motif_len:
        use_double = allow_double and (rng.random() > density)
        target_len = grid * (2 if use_double else 1)
        target_len = min(target_len, motif_len - max(0, cursor - xfade))

        seg = _stitch_to_len_legato(slice_lib, target_len, sr, xfade_ms=xfade_ms)

        if rng.random() < granular_chance and len(seg) > int(0.04 * sr):
            seg = granular_synthesis(seg, sr, intensity=0.12)
            granular_used += 1

        start = max(0, cursor - xfade)
        _overlap_add_legato(motif, seg, start, xfade)
        cursor = start + len(seg)

    # --- tile motif across song ---
    looped = _tile_repeat(motif, out_len)

    # --- random converge: crossfade first 'converge_bars' into the loop ---
    cb = int(max(0, converge_bars))
    if cb > 0:
        free, _, g2 = place_slices_legato_on_grid_constant_speed(
            slice_lib, sr, bpm, bars=bars, subdiv=subdiv, density=density,
            allow_double=allow_double, granular_chance=granular_chance,
            xfade_ms=xfade_ms, apply_reverb=False, reverb_amount=0.0
        )
        cf_len = min(out_len, 4 * cb * beat_samps)
        ramp = np.linspace(0.0, 1.0, cf_len, endpoint=True).astype(np.float64)
        y = looped.astype(np.float64)
        y[:cf_len] = free[:cf_len].astype(np.float64) * (1.0 - ramp) + y[:cf_len] * ramp
        looped = y.astype(np.float32)
        granular_used += g2

    return looped, out_len, granular_used



# =========================
# Slice library + rhythmic scheduler (constant-speed chops)
# =========================

def make_small_slices(x: np.ndarray, sr: int, count: int = 64, min_ms=70, max_ms=260):
    rng = np.random.default_rng()
    slices = []
    L = len(x)
    for _ in range(count):
        if L < sr // 10:
            break
        dur = rng.integers(int(min_ms/1000.0 * sr), int(max_ms/1000.0 * sr))
        start = rng.integers(0, max(1, L - dur))
        seg = x[start:start + dur].copy().astype(np.float32)
        # light edge fades
        n = len(seg)
        f = max(8, min(n // 6, int(0.01 * sr)))
        if f > 0 and n > 2*f:
            w = np.hanning(2*f)
            seg[:f] *= w[:f]
            seg[-f:] *= w[-f:]
        slices.append(seg)
    return slices

def _rate_slices(slice_lib, rate: float):
    """Apply playback-rate (pitch+speed) to backing-chop slices only."""
    r = float(np.clip(rate, 0.1, 2.0))
    if abs(r - 1.0) < 1e-9:
        return slice_lib
    return [resample_speed(seg, r) for seg in slice_lib]

def _fit_len_no_pitch(seg: np.ndarray, target_len: int) -> np.ndarray:
    """Crop/pad WITHOUT time-scaling -> keeps speed/pitch constant."""
    n = len(seg)
    if n == target_len:
        out = seg.copy()
    elif n > target_len:
        out = seg[:target_len].copy()
    else:
        out = np.pad(seg, (0, target_len - n))
    # micro fades to avoid clicks
    f = max(8, min(target_len // 12, 64))
    if target_len >= 2*f:
        w = np.hanning(2*f)
        out[:f] *= w[:f]
        out[-f:] *= w[-f:]
    return out.astype(np.float32)

def _envelope_abs_smooth(x: np.ndarray, sr: int, smooth_ms: float = 5.0) -> np.ndarray:
    """Fast rectified + moving-average envelope."""
    win = max(1, int((smooth_ms * 1e-3) * sr))
    if win <= 1:
        return np.abs(x).astype(np.float32)
    env = np.convolve(np.abs(x).astype(np.float64), np.ones(win) / win, mode="same")
    return env.astype(np.float32)

def _detect_transient_peaks(env: np.ndarray, sr: int,
                            min_dist_ms: float = 20.0,
                            height_q: float = 0.90) -> np.ndarray:
    """
    Robust transient picker on a rectified/smoothed envelope.
    - Smooths with short Hann
    - Subtracts a local median baseline (removes sustain)
    - Half-wave rectifies + light compression
    - Picks peaks with adaptive height + prominence + width constraints
    """
    e = env.astype(np.float64)

    # Short smoothing (~3 ms)
    win = max(9, int(0.003 * sr))
    if win % 2 == 0:
        win += 1
    h = np.hanning(win)
    h /= h.sum()
    e = np.convolve(e, h, mode="same")

    # Local baseline (median) to isolate onsets
    k = max(3, int(0.10 * sr))
    if k % 2 == 0:
        k += 1
    baseline = signal.medfilt(e, kernel_size=k)

    # Novelty: baseline-removed + half-wave + gentle comp
    nov = e - baseline
    nov[nov < 0.0] = 0.0
    nov = np.sqrt(nov + 1e-12)

    # Adaptive threshold + constraints
    dist = max(1, int((min_dist_ms * 1e-3) * sr))
    thr = float(np.quantile(nov, np.clip(height_q, 0.0, 1.0)))
    prom = 0.6 * thr
    wmin = max(1, int(0.002 * sr))   # ~2 ms
    wmax = max(wmin + 1, int(0.03 * sr))  # ~30 ms

    pk, _ = signal.find_peaks(nov, height=thr, prominence=prom,
                              width=(wmin, wmax), distance=dist)
    return pk.astype(int)

def make_small_slices_low_peak(x: np.ndarray, sr: int, count: int = 64,
                               min_ms: float = 300.0, max_ms: float = 600.0,
                               step_ms: float = 20.0,
                               height_q: float = 0.90, min_dist_ms: float = 20.0,
                               low_quantile: float = 0.25) -> list:
    """
    Build slices from windows that have the FEWEST transient peaks.
    - Only affects backing chops; do NOT use this for kicks/hats/claps.
    """
    if len(x) < int(0.1 * sr):
        return []

    rng = np.random.default_rng()

    # Envelope + peaks once for the whole file
    env = _envelope_abs_smooth(x, sr, smooth_ms=5.0)
    peaks = _detect_transient_peaks(env, sr, min_dist_ms=min_dist_ms, height_q=height_q)

    # Peak indicator & prefix sum for O(1) counts per window
    ispk = np.zeros(len(x), dtype=np.int32)
    ispk[np.clip(peaks, 0, len(ispk) - 1)] = 1
    csum = np.concatenate([[0], np.cumsum(ispk)])

    # Candidate windows on a grid
    step = max(1, int((step_ms * 1e-3) * sr))
    Lmin = max(8, int((min_ms * 1e-3) * sr))
    Lmax = max(Lmin + 1, int((max_ms * 1e-3) * sr))
    starts = np.arange(0, max(1, len(x) - Lmin), step, dtype=int)

    # Loudness floor to avoid near-silence
    env_floor = float(np.quantile(env, 0.25)) * 0.5

    cands = []
    for s in starts:
        dur = int(rng.integers(Lmin, Lmax))
        e = min(len(x), s + dur)
        if e - s < Lmin:
            continue
        # mean envelope gate
        m_env = float(env[s:e].mean())
        if m_env < env_floor:
            continue
        # peak count
        pk_cnt = int(csum[e] - csum[s])
        cands.append((pk_cnt, s, e))

    if not cands:
        # fallback to original random slicer
        return make_small_slices(x, sr, count=count, min_ms=min_ms, max_ms=max_ms)

    # Keep the lowest-peak subset, then sample from it
    cands.sort(key=lambda t: t[0])
    k = max(8, int(len(cands) * float(np.clip(low_quantile, 0.01, 1.0))))
    low_pk_zone = cands[:k]

    slices = []
    rng.shuffle(low_pk_zone)
    for pk_cnt, s, e in low_pk_zone[:count * 2]:  # a bit extra to survive fades
        seg = x[s:e].astype(np.float32).copy()
        # soft edge fades
        n = len(seg)
        f = max(8, min(n // 6, int(0.01 * sr)))
        if f > 0 and n > 2 * f:
            w = np.hanning(2 * f)
            seg[:f] *= w[:f]
            seg[-f:] *= w[-f:]
        slices.append(seg)
        if len(slices) >= count:
            break

    if not slices:  # rare fallback
        return make_small_slices(x, sr, count=count, min_ms=min_ms, max_ms=max_ms)
    return slices

def _overlap_add_legato(out: np.ndarray, seg: np.ndarray, start: int, xfade: int) -> None:
    """
    Equal-power crossfade with RMS matching at the seam.
    Reduces level dips/bumps vs linear ramps.
    """
    L = len(out)
    if start >= L:
        return
    seg = seg.astype(np.float64)
    end = min(L, start + len(seg))
    seg_len = end - start
    if seg_len <= 0:
        return

    cross = min(xfade, seg_len, start)
    if cross > 0:
        # equal-power ramps
        t = np.linspace(0.0, 1.0, cross, endpoint=True)
        fo = np.cos(0.5 * np.pi * t)   # fade-out old
        fi = np.sin(0.5 * np.pi * t)   # fade-in new

        a = out[start - cross:start].astype(np.float64)
        b = seg[:cross].astype(np.float64)

        # match RMS in overlap to minimize audibility
        ra = np.sqrt((a * a).mean() + 1e-12)
        rb = np.sqrt((b * b).mean() + 1e-12)
        if rb > 0.0:
            b = b * (ra / rb)

        out[start - cross:start] = a * fo + b * fi
        w_start = start
        s_off = cross
    else:
        w_start = start
        s_off = 0

    remain = seg_len - (w_start - start) - s_off
    if remain > 0:
        out[w_start:w_start + remain] += seg[s_off:s_off + remain]


def _stitch_to_len_legato(slice_lib, target_len, sr, xfade_ms=12):
    """
    Build a segment of EXACT length by chaining slices with crossfades.
    No resampling, no padding → truly legato inside each grid cell.
    """
    rng = np.random.default_rng()
    out = np.zeros(int(target_len), dtype=np.float32)
    xfade = max(0, int((xfade_ms / 1000.0) * sr))
    pos = 0
    while pos < target_len:
        remain = target_len - pos
        seg = slice_lib[rng.integers(0, len(slice_lib))]
        seg_use = seg[:remain].astype(np.float32)  # crop if longer; never pad
        _overlap_add_legato(out, seg_use, start=pos, xfade=xfade)
        pos += len(seg_use)
    return out

def place_slices_legato_on_grid_constant_speed(slice_lib, sr, bpm, bars=8, subdiv=2,
                                               density=0.6, allow_double=True,
                                               granular_chance=0.01, xfade_ms=12,
                                               apply_reverb=False, reverb_amount=1.0):
    """
    Legato version: slices are back-to-back with a small crossfade; no silent gaps.
    - Lengths are exact multiples of the grid (1x or 2x), so groove stays locked.
    - 'density' steers how often we choose 1x (dense) vs 2x (more sustained).
      (p_1x = density, p_2x = 1 - density)
    - Optional global reverb applied to entire output as post-process.
    """
    assert subdiv >= 1
    beat_samps = int(sr * 60.0 / bpm)
    grid = beat_samps // subdiv
    total_beats = 4 * bars
    out_len = total_beats * beat_samps
    out = np.zeros(out_len, dtype=np.float32)

    rng = np.random.default_rng()
    granular_used = 0
    xfade = max(0, int((xfade_ms / 1000.0) * sr))

    cursor = 0
    while cursor < out_len:
        # choose 1x or 2x grid; keep all starts on-grid by construction
        use_double = allow_double and (rng.random() > density)
        target_len = grid * (2 if use_double else 1)
        target_len = min(target_len, out_len - max(0, cursor - xfade))

        # Stitch multiple source slices to fill the cell EXACTLY, no padding ‚Üí true legato
        seg = _stitch_to_len_legato(slice_lib, target_len, sr, xfade_ms=xfade_ms)

        # rare, gentle granular on the stitched segment (length unchanged)
        if rng.random() < granular_chance and len(seg) > int(0.04 * sr):
            seg = granular_synthesis(seg, sr, intensity=0.12)
            granular_used += 1

        # Legato write: start at (cursor - xfade) to overlap smoothly
        start = max(0, cursor - xfade)
        _overlap_add_legato(out, seg, start, xfade)

        cursor = start + len(seg)

    # Apply global reverb if requested (before final normalize)
    if apply_reverb and reverb_amount > 0.0:
        # === replaced broken reverb with the script's working reverb ===
        out = wider_glue_reverb(out, sr, strength=reverb_amount)
    
    return out.astype(np.float32), out_len, granular_used

def place_slices_legato_on_grid_constant_speed(slice_lib, sr, bpm, bars=8, subdiv=2,
                                               density=0.6, allow_double=True,
                                               granular_chance=0.01, xfade_ms=12,
                                               apply_reverb=False, reverb_amount=1.0):
    """
    Legato version: slices are back-to-back with a small crossfade; no silent gaps.
    - Lengths are exact multiples of the grid (1x or 2x), so groove stays locked.
    - 'density' steers how often we choose 1x (dense) vs 2x (more sustained).
      (p_1x = density, p_2x = 1 - density)
    - Optional global reverb applied to entire output as post-process.
    """
    assert subdiv >= 1
    beat_samps = int(sr * 60.0 / bpm)
    grid = beat_samps // subdiv
    total_beats = 4 * bars
    out_len = total_beats * beat_samps
    out = np.zeros(out_len, dtype=np.float32)

    rng = np.random.default_rng()
    granular_used = 0
    xfade = max(0, int((xfade_ms / 1000.0) * sr))

    cursor = 0
    while cursor < out_len:
        # choose 1x or 2x grid; keep all starts on-grid by construction
        use_double = allow_double and (rng.random() > density)
        target_len = grid * (2 if use_double else 1)
        target_len = min(target_len, out_len - max(0, cursor - xfade))

        # Stitch multiple source slices to fill the cell EXACTLY, no padding → true legato
        seg = _stitch_to_len_legato(slice_lib, target_len, sr, xfade_ms=xfade_ms)

        # rare, gentle granular on the stitched segment (length unchanged)
        if rng.random() < granular_chance and len(seg) > int(0.04 * sr):
            seg = granular_synthesis(seg, sr, intensity=0.12)
            granular_used += 1

        # Legato write: start at (cursor - xfade) to overlap smoothly
        start = max(0, cursor - xfade)
        _overlap_add_legato(out, seg, start, xfade)

        cursor = start + len(seg)

    # Apply global reverb if requested (before final normalize)
    if apply_reverb and reverb_amount > 0.0:
        r = float(np.clip(reverb_amount, 0.0, 2.0))
        
        # Schroeder reverb (direct implementation, no IR)
        wet_amt = 0.25 * r
        rt60 = 1.25 * r  # decay time in seconds
        
        y = out.astype(np.float64).copy()
        
        # Comb filters (parallel)
        comb_delays = [int(0.0297 * sr), int(0.0371 * sr), int(0.0411 * sr), int(0.0437 * sr)]
        comb_out = np.zeros_like(y)
        for delay in comb_delays:
            g = 10 ** (-3 * delay / (rt60 * sr))  # decay coefficient
            buf = np.zeros(delay)
            temp = np.zeros_like(y)
            for i in range(len(y)):
                temp[i] = y[i] + g * buf[0]
                buf = np.roll(buf, -1)
                buf[-1] = temp[i]
            comb_out += temp
        comb_out /= len(comb_delays)
        
        # Allpass filters (serial)
        ap_delays = [int(0.005 * sr), int(0.0017 * sr)]
        ap_out = comb_out
        for delay in ap_delays:
            buf = np.zeros(delay)
            temp = np.zeros_like(ap_out)
            g = 0.7
            for i in range(len(ap_out)):
                temp[i] = -g * ap_out[i] + buf[0] + g * buf[0]
                buf = np.roll(buf, -1)
                buf[-1] = ap_out[i]
            ap_out = temp
        
        out = ((1.0 - wet_amt) * out + wet_amt * ap_out).astype(np.float32)
    
    return out.astype(np.float32), out_len, granular_used

def _butter_bp(x, sr, low=900.0, high=3500.0, order=4):
    low = max(40.0, float(low))
    high = min(0.49 * sr, float(high))
    if high <= low:
        high = low * 1.5
    b, a = signal.butter(order, [low / (0.5 * sr), high / (0.5 * sr)], btype='bandpass')
    return signal.lfilter(b, a, x)

def make_clap_from_slice(x, sr):
    """
    Snappy clap from the source sample: band-pass + short multi-flam + decay.
    """
    if len(x) == 0:
        return x
    rng = np.random.default_rng()
    dur = int(rng.uniform(0.065, 0.110) * sr)
    g = x[:dur].astype(np.float64)

    # Core tone - thinner, brighter
    lo = rng.uniform(1400.0, 1900.0)
    hi = rng.uniform(5500.0, 7500.0)
    g = _butter_bp(g, sr, low=lo, high=hi, order=5)

    # Faster decay + harder drive
    t = np.linspace(0.0, 1.0, len(g))
    g *= np.exp(-12.0 * t)
    g = np.tanh(2.1 * g)

    # Tighter flam
    out = g.copy()
    for w, ms in [(0.42, rng.uniform(2.0, 4.0)),
                  (0.28, rng.uniform(5.0, 8.0))]:
        d = int(ms * 1e-3 * sr)
        if d < len(g):
            tmp = np.zeros_like(g)
            tmp[d:] = g[:-d]
            out += w * tmp

    out /= (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)

def generate_clap_positions_from_kicks(kick_positions):
    """
    Every 2nd kick → backbeat (2, 4, ...).
    """
    if len(kick_positions) == 0:
        return np.array([], dtype=int)
    return np.asarray(kick_positions[1::2], dtype=int)

def generate_deviant_claps(base_positions, sr, prob=0.08, max_ms=22.0, rng=None):
    """
    Bonus claps that rarely deviate in timing (+/- jitter).
    Returns list of positions (ints).
    """
    if rng is None:
        rng = np.random.default_rng()
    dev = []
    max_jit = int(abs(max_ms) * 1e-3 * sr)
    if max_jit <= 0:
        return dev
    for p in base_positions:
        if rng.random() < prob:
            dev.append(int(p + rng.integers(-max_jit, max_jit + 1)))
    return dev

def apply_random_filter_schedules(x, sr, bpm, bars, subdiv, rng=None):
    """
    LP/BP over the chop track.
    - If FILTER_GLOBAL is "lp" or "bp": apply that filter across the ENTIRE track with FILTER_WET.
    - Otherwise, schedule several LP/BP events; FILTER_COVERAGE increases time under filters,
      FILTER_WET increases their blend amount.
    """
    if rng is None:
        rng = np.random.default_rng()

    y = x.astype(np.float64).copy()

    # --- Global (force all) path ---
    if FILTER_GLOBAL in ("lp", "bp"):
        if FILTER_GLOBAL == "lp":
            cutoff = float(FILTER_LP_CUTOFF_HZ)
            b, a = signal.butter(4, cutoff / (0.5 * sr), btype='lowpass')
        else:
            lo = float(FILTER_BP_LOW_HZ)
            hi = float(min(FILTER_BP_HIGH_HZ, 0.49 * sr))
            if hi <= lo:
                hi = lo * 1.5
            b, a = signal.butter(4, [lo / (0.5 * sr), hi / (0.5 * sr)], btype='bandpass')

        wet = float(np.clip(FILTER_WET, 0.0, 1.0))
        yf = signal.lfilter(b, a, y)
        out = (1.0 - wet) * y + wet * yf
        peak = np.max(np.abs(out))
        if peak > 1.0:
            out /= peak
        return out.astype(np.float32)

    # --- Scheduled events path ---
    beat = 60.0 / bpm
    beat_samps = int(sr * beat)
    total_beats = 4 * bars

    # more coverage => more/longer events
    cov = float(max(0.0, FILTER_COVERAGE))
    n_events = int(np.clip(rng.integers(2, 6) * (bars / 8.0) * cov, 1, max(1, int(7 * cov))))
    fades_ms = (rng.uniform(18.0, 35.0), rng.uniform(18.0, 35.0))

    for _ in range(n_events):
        dur_beats = float(rng.uniform(0.5, 2.5 * cov))
        start_beat = float(rng.uniform(0.0, max(0.0, total_beats - dur_beats)))
        end_beat = start_beat + dur_beats

        s = int(start_beat * beat_samps)
        e = int(end_beat * beat_samps)
        if e - s < int(0.15 * sr):
            continue

        seg = y[s:e].copy()

        # random LP or BP
        if rng.random() < 0.5:
            cutoff = float(rng.uniform(1400.0, 4800.0))
            b, a = signal.butter(4, cutoff / (0.5 * sr), btype='lowpass')
        else:
            lo = float(rng.uniform(350.0, 1400.0))
            hi = float(rng.uniform(lo * 1.8, min(lo * 3.6, 0.49 * sr)))
            b, a = signal.butter(4, [lo / (0.5 * sr), hi / (0.5 * sr)], btype='bandpass')

        seg_f = signal.lfilter(b, a, seg)

        fin_ms, fout_ms = fades_ms
        fin = max(8, min(int(fin_ms * 1e-3 * sr), len(seg_f) // 3))
        fout = max(8, min(int(fout_ms * 1e-3 * sr), len(seg_f) // 3))

        env = np.ones(len(seg_f), dtype=np.float64)
        env[:fin] *= np.linspace(0.0, 1.0, fin)
        env[-fout:] *= np.linspace(1.0, 0.0, fout)
        env *= float(np.clip(FILTER_WET, 0.0, 1.0))

        seg_out = seg * (1.0 - env) + seg_f * env
        y[s:e] = seg_out

    peak = np.max(np.abs(y))
    if peak > 1.0:
        y /= peak
    return y.astype(np.float32)


# =========================
# Mixing helpers
# =========================

def mix_one_shots(base, one_shot, positions, gain=1.0):
    out = base.astype(np.float64).copy()
    one = (one_shot.astype(np.float64) * gain)
    L = len(out)
    for p in positions:
        if p >= L:
            continue
        end = min(L, p + len(one))
        seg = end - p
        if seg > 0:
            out[p:end] += one[:seg]
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)

def mix_hat_events(base, hat_closed, hat_open, events, global_gain=1.0):
    """
    events: list of (pos, is_open, gain); global_gain scales all hats.
    """
    out = base.astype(np.float64).copy()
    L = len(out)
    gg = float(global_gain)
    for pos, is_open, g in events:
        hat = hat_open if is_open else hat_closed
        end = min(L, pos + len(hat))
        seg = end - pos
        if seg > 0:
            out[pos:end] += (hat[:seg].astype(np.float64) * g * gg)
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out.astype(np.float32)

def build_strict_looped_chops(
    slice_lib, sr, bpm, *,
    bars=8, subdiv=2, density=0.6,
    loop_bars=2, converge_bars=2,
    allow_double=True, granular_chance=0.0,
    xfade_ms=12, rng=None
):
    """
    Build ONE 2-bar motif and hard-tile it. Optional 'converge_bars' crossfades
    only the FIRST bars from a free take into the loop; after that it's identical.
    Returns (looped, out_len, granular_used).
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    beat_samps = int(sr * 60.0 / bpm)
    grid = max(1, beat_samps // max(1, subdiv))
    bars = max(loop_bars, int(bars))
    out_len = 4 * bars * beat_samps
    motif_len = 4 * loop_bars * beat_samps
    xfade = max(0, int((xfade_ms / 1000.0) * sr))

    granular_used = 0

    # --- make the motif (fixed content) ---
    motif = np.zeros(motif_len, dtype=np.float32)
    cursor = 0
    while cursor < motif_len:
        use_double = allow_double and (rng.random() > density)
        target_len = min(grid * (2 if use_double else 1), motif_len - max(0, cursor - xfade))
        seg = _stitch_to_len_legato(slice_lib, target_len, sr, xfade_ms=xfade_ms)
        if granular_chance and rng.random() < granular_chance and len(seg) > int(0.04 * sr):
            seg = granular_synthesis(seg, sr, intensity=0.12)
            granular_used += 1
        start = max(0, cursor - xfade)
        _overlap_add_legato(motif, seg, start, xfade)
        cursor = start + len(seg)

    # --- hard-tile motif across song length ---
    looped = _tile_repeat(motif, out_len)

    # --- optional converge into the loop only at the very start ---
    cb = max(0, int(converge_bars))
    if cb > 0:
        free, _, g2 = place_slices_legato_on_grid_constant_speed(
            slice_lib, sr, bpm,
            bars=bars, subdiv=subdiv, density=density,
            allow_double=allow_double, granular_chance=granular_chance,
            xfade_ms=xfade_ms, apply_reverb=False, reverb_amount=0.0
        )
        granular_used += g2
        cf_len = min(out_len, 4 * cb * beat_samps)
        ramp = np.linspace(0.0, 1.0, cf_len, endpoint=True).astype(np.float64)
        y = looped.astype(np.float64)
        y[:cf_len] = free[:cf_len].astype(np.float64) * (1.0 - ramp) + y[:cf_len] * ramp
        looped = y.astype(np.float32)

    return looped, out_len, granular_used

# =========================
# Main process
# =========================

def process(input_file: str, bpm: int = 120, speed: float = 1.0,
            bars: int = None, subdiv: int = 2,
            base_density: float = 0.6, reverb: float = 1.0,
            hat_density: float = 0.35,
            clap: bool = False,              # snare-ish
            clap_dev_prob: float = 0.08,
            clap_dev_ms: float = 22.0,
            rate: float = 1.0):              # rate now targets backing chops ONLY
    # Load
    audio, sr = load_audio_mono(input_file)

    # Derive default bars from file length if not set
    if bars is None:
        total_beats_est = int((len(audio) / sr) / (60.0 / bpm))
        bars = int(max(4, min(16, total_beats_est // 4)))  # clamp 4..16

    # Slice library from source
    chop_slice_lib = make_small_slices_low_peak(audio, sr, count=64, min_ms=300, max_ms=600)
    perc_slice_lib = make_small_slices(audio, sr, count=64, min_ms=300, max_ms=600)

    # >>> Apply 'rate' (kick/hat/clap are unaffected)
    chop_slice_lib = _rate_slices(chop_slice_lib, rate)

    # 2-bar looped chops
    chop_track, out_len, granular_used = build_strict_looped_chops(
        chop_slice_lib, sr, bpm,
        bars=bars, subdiv=subdiv, density=base_density,
        loop_bars=2, converge_bars=2,   # set to 0 for strict from bar 1
        allow_double=True, granular_chance=0.01, xfade_ms=30
    )

    # Fuller-band glue verb on looped chops (static; won't break the loop identity)
    chop_track = apply_global_loop_reverb(chop_track, sr, wet_db=-3.0, strength=reverb, tail_secs=1.0)

    # keep loop identical; disable time-varying filter schedules on chops
    # chop_track = apply_random_filter_schedules(chop_track, sr, bpm, bars, subdiv)

    # One-shot percs from source (NOT rate-scaled)
    rng = np.random.default_rng()
    kick_src = (perc_slice_lib[rng.integers(0, len(perc_slice_lib))] 
                if perc_slice_lib else audio[:int(0.12 * sr)])
    hat_src  = (perc_slice_lib[rng.integers(0, len(perc_slice_lib))] 
                if perc_slice_lib else audio[:int(0.14 * sr)])

    kick = make_kick_from_slice(kick_src, sr)
    hat_closed, hat_open = make_hat_variants_from_slice(hat_src, sr)

    # Kicks: strict regular pattern
    k_on = generate_regular_kick_onsets(sr, bpm, out_len)

    # Hats: sparse, with per-hit variance and occasional open hats
    shuffle_used = float(np.random.uniform(0.12, 0.22))
    hat_events = generate_sparse_shuffled_hat_events(
        sr, bpm, out_len, base_density=hat_density, shuffle=shuffle_used, rng=rng
    )

    # Layer percussion (kick + hat) on top of reverbed chops
    layered = mix_one_shots(chop_track, kick, k_on, gain=KICK_GAIN)
    layered = mix_hat_events(layered, hat_closed, hat_open, hat_events, global_gain=HAT_GAIN)

    # Optional claps (snare-ish)
    claps_main = 0
    claps_dev = 0
    if clap:
        clap_src = (perc_slice_lib[rng.integers(0, len(perc_slice_lib))] 
                    if perc_slice_lib else audio[:int(0.16 * sr)])
        clap_one = make_clap_from_slice(clap_src, sr)

        c_on = generate_clap_positions_from_kicks(k_on)  # backbeat
        layered = mix_one_shots(layered, clap_one, c_on, gain=CLAP_GAIN)
        claps_main = int(len(c_on))

        c_dev = generate_deviant_claps(c_on, sr, prob=clap_dev_prob, max_ms=clap_dev_ms, rng=rng)
        if len(c_dev):
            layered = mix_one_shots(layered, clap_one, c_dev, gain=CLAP_DEV_GAIN * CLAP_GAIN)
            claps_dev = int(len(c_dev))

    # >>> tiny mix-bus glue reverb (post layering, pre speed/normalize)
    layered = apply_bus_glue_reverb(layered, sr, send=0.90, return_db=-6.0, strength=1.5)

    # Global speed still affects the whole mix (tempo/pitch)
    final = resample_speed(layered, speed)
    final = apply_bus_glue_reverb(final, sr, send=0.90, return_db=-11.0, strength=1.5)
    final = normalize(np.tanh(final * 1.04), 0.98)

    meta = {
        "bpm": bpm,
        "speed": speed,
        "bars": bars,
        "subdiv": subdiv,
        "shuffle": round(shuffle_used, 3),
        "granular_used": int(granular_used),
        "kicks": int(len(k_on)),
        "hats": int(len(hat_events)),
        "claps": claps_main,
        "claps_deviant": claps_dev,
        "sr": sr,
        "length_sec": round(len(final)/sr, 3),
        "chop_rate": float(rate),
        "reverb": float(reverb),
    }
    return final, sr, meta


def main(input_file: str, bpm: int = 120, speed: float = 1.0,
         rate: float = 1.0,
         bars: int = None, subdiv: int = 2, hat_density: float = 0.35,
         clap: bool = False, clap_dev_prob: float = 0.08, clap_dev_ms: float = 22.0,
         reverb: float = 1.0):
    # Build with 'rate' applied only to backing chops
    data, sr, meta = process(
        input_file, bpm=bpm, speed=speed, bars=bars, subdiv=subdiv,
        base_density=0.6, reverb=reverb, hat_density=hat_density,
        clap=clap, clap_dev_prob=clap_dev_prob, clap_dev_ms=clap_dev_ms,
        rate=rate
    )

    # Output samplerate no longer changes with 'rate' (percs must stay put)
    sr_out = sr

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = get_next_filename(base_name)
    sf.write(output_file, data, sr_out)

    length_sec_out = round(len(data) / sr_out, 3)

    print(
        f"saved: {output_file} | "
        f"bpm {meta['bpm']} | bars {meta['bars']} | subdiv {meta['subdiv']} | "
        f"shuffle {meta['shuffle']} | speed {meta['speed']} | rate(chops) {meta['chop_rate']} | "
        f"reverb {meta['reverb']} | granular {meta['granular_used']} | "
        f"k:{meta['kicks']} h:{meta['hats']} c:{meta['claps']}(+{meta['claps_deviant']}) | "
        f"len {length_sec_out}s | sr {sr_out}Hz"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constant-speed chopper + kicks + sparse hats + optional clap + scheduled LP/BP on chops."
    )
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo (default 120)")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback rate (0.1..2.0) for BACKING CHOPS ONLY; kick/hat/clap unaffected.")
    parser.add_argument("--reverb", type=float, default=1.0, help="Glue reverb amount 0..2 (0 disables; 1 ≈ previous default).")
    parser.add_argument("--speed", type=float, default=1.0, help="Global speed factor (default 1.0)")
    parser.add_argument("--bars", type=int, default=None, help="Exact bars to render (default: auto 4..16)")
    parser.add_argument("--subdiv", type=int, default=1, help="Chop grid per beat (1=quarters, 2=eighths, 4=sixteenths)")
    parser.add_argument("--hat_density", type=float, default=0.60, help="Overall hat density (lower = sparser)")
    parser.add_argument("--clap", action="store_true", help="Add claps on backbeats (every 2nd kick).")
    parser.add_argument("--clap_dev_prob", type=float, default=0.10, help="Prob of bonus deviating clap per backbeat (0..1).")
    parser.add_argument("--clap_dev_ms", type=float, default=22.0, help="Max abs deviation (ms) for bonus clap.")

    args = parser.parse_args()
    main(args.input_file, args.bpm, args.speed, args.rate, args.bars, args.subdiv,
         args.hat_density, args.clap, args.clap_dev_prob, args.clap_dev_ms)
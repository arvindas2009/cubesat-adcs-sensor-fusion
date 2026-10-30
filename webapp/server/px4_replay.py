"""PX4 log replay: the precomputed, cached document behind the "PX4 replay" page (spec: docs/specs/px4_replay.md).

A visual aid only. Every series comes from the existing Project A code (``run_filter``, the ``metrics``
helpers, ``interp_to``, ``to_euler_zyx``); nothing is estimated here. **EKF2 is a reference, not truth.**
Quaternions are scalar-first ``[w, x, y, z]``, body (FRD) to NED; SI units; degrees only in fields named
``*_deg``. Times are seconds on the log's own clock (the filter's IMU time base).

This module imports no web framework (``app.py`` calls it); heavy imports (``sensor_fusion.run_flight``,
which pulls in the ULog loader) are lazy.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import chi2

from common import quaternion as quat
from sensor_fusion import metrics
from sensor_fusion.download_logs import DATA_DIR, PINNED_LOGS
from sensor_fusion.ekf_drone import IDX_P, IDX_TH, run_filter, wrap_angle
from sensor_fusion.interp import interp_to

__all__ = ["CACHE_VERSION", "DEFAULT_RATE_HZ", "DEFAULT_CACHE_DIR", "DOWNLOAD_HINT", "CAVEAT", "UnknownLog",
           "LogMissing", "build_replay", "replay_to_json", "ReplayStore"]

CACHE_VERSION = 1
DEFAULT_RATE_HZ = 50.0
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"      # <repo>/webapp/.cache (gitignored)
DOWNLOAD_HINT = "python -m sensor_fusion.download_logs"
CAVEAT = "EKF2 is a reference, not truth"
SCHEMA_VERSION = 1
MIN_TOKEN_PREFIX = 4         # shortest accepted id prefix


class UnknownLog(LookupError):
    """The token matches no pinned log (or more than one)."""


class LogMissing(FileNotFoundError):
    """The pinned log exists but its ``.ulg`` file is not in the data directory."""


# ---------------------------------------------------------------------------
# build_replay
# ---------------------------------------------------------------------------
def _span_mask(t: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """True where grid time ``t`` [s] lies inside the reference's own span ``[t_ref[0], t_ref[-1]]``."""
    t_ref = np.asarray(t_ref, float)
    if t_ref.size == 0:
        return np.zeros(t.shape, bool)
    return (t >= t_ref[0]) & (t <= t_ref[-1])


def _interp_masked(t: np.ndarray, mask: np.ndarray, t_ref, x, *, quaternion: bool = False) -> np.ndarray:
    """``interp_to`` onto the grid samples where ``mask`` holds; NaN elsewhere. Shape ``(n,) + x.shape[1:]``."""
    x = np.asarray(x, float)
    out = np.full((t.size,) + x.shape[1:], np.nan)
    if mask.any():
        out[mask] = interp_to(t[mask], t_ref, x, quaternion=quaternion)
    return out


def _euler_deg(q: np.ndarray) -> np.ndarray:
    """(n, 3) ``[roll, pitch, yaw]`` [deg] of body-to-NED quaternions ``q`` (n, 4); NaN rows give NaN."""
    out = np.full((q.shape[0], 3), np.nan)
    ok = np.all(np.isfinite(q), axis=1)
    if ok.any():
        out[ok] = np.rad2deg(np.stack(quat.to_euler_zyx(q[ok]), axis=1))
    return out


def _nis_entries(innovations: dict, t0: float, t1: float, rate_hz: float) -> dict:
    """Decimated NIS series per sensor (spec 1.1): finite samples, every ``stride``-th if too many."""
    budget = rate_hz * (t1 - t0)
    out = {}
    for name in metrics.SENSOR_ORDER:
        inn = innovations.get(name)
        if inn is None or np.size(inn.nis) < 1:
            continue
        t = np.asarray(inn.t, float)
        nis = np.asarray(inn.nis, float)
        acc = np.asarray(inn.accepted, bool)
        dof = np.asarray(inn.dof, int)
        keep = np.isfinite(t) & np.isfinite(nis)
        t, nis, acc, dof = t[keep], nis[keep], acc[keep], dof[keep]
        if t.size < 1:
            continue
        if t.size > budget:
            stride = int(math.ceil(t.size / budget)) if budget > 0 else t.size
            t, nis, acc, dof = t[::stride], nis[::stride], acc[::stride], dof[::stride]
        d = int(Counter(dof.tolist()).most_common(1)[0][0])
        out[name] = {"t": t, "nis": nis, "accepted": acc, "dof": d,
                     "band95": np.asarray(chi2.ppf([0.025, 0.975], d), float)}
    return out


def build_replay(streams, ref, cfg, *, log_meta: dict, rate_hz: float = DEFAULT_RATE_HZ) -> dict:
    """Run the filter once and resample ours, EKF2 and their differences onto a uniform grid.

    ``streams``: ``SensorStreams`` (SI, NED, FRD). ``ref``: ``ekf2_reference``-shaped dict (``t_pos`` s,
    ``pos_ned`` m, ``t_att`` s, ``q`` body-to-NED, ``reset_times`` s). ``cfg``: ``EkfConfig``. Returns the
    unrounded dict of spec section 1.1 (arrays for the series; ``replay_to_json`` serialises it). EKF2 and
    difference entries are NaN outside the reference's own time span. Differences are ours minus EKF2.
    """
    res = run_filter(streams, cfg)
    t0, t1 = float(res.t[0]), float(res.t[-1])
    n = int(math.floor((t1 - t0) * rate_hz + 1e-9)) + 1
    t = t0 + np.arange(n) / rate_hz

    # ours
    p = interp_to(t, res.t, res.p)
    q = interp_to(t, res.t, res.q, quaternion=True)
    std = interp_to(t, res.t, res.std)
    ours = {"p_ned_m": p, "q": q, "euler_deg": _euler_deg(q),
            "sigma3_att_deg": 3.0 * np.rad2deg(std[:, IDX_TH]),
            "sigma3_pos_m": 3.0 * std[:, IDX_P]}

    # EKF2 (NaN outside its own span, separately for position and attitude)
    m_pos = _span_mask(t, ref["t_pos"])
    m_att = _span_mask(t, ref["t_att"])
    p2 = _interp_masked(t, m_pos, ref["t_pos"], ref["pos_ned"])
    q2 = _interp_masked(t, m_att, ref["t_att"], ref["q"], quaternion=True)
    ekf2 = {"p_ned_m": p2, "q": q2, "euler_deg": _euler_deg(q2)}

    # differences, ours - EKF2, with the metrics helpers
    dp = p - p2
    tilt = np.full(n, np.nan)
    yaw = np.full(n, np.nan)
    if m_att.any():
        tilt[m_att] = np.rad2deg(metrics.tilt_angle(q[m_att], q2[m_att]))
        yaw[m_att] = np.rad2deg(wrap_angle(metrics.yaw_of(q[m_att]) - metrics.yaw_of(q2[m_att])))
    diff = {"pos_h_m": np.linalg.norm(dp[:, :2], axis=1), "pos_d_m": dp[:, 2].copy(),
            "tilt_deg": tilt, "yaw_deg": yaw}

    resets = np.asarray(ref.get("reset_times", []), float).ravel()
    resets = np.sort(resets[np.isfinite(resets) & (resets >= t0) & (resets <= t1)])

    return {
        "schema_version": SCHEMA_VERSION, "caveat": CAVEAT, "rate_hz": float(rate_hz),
        "log": log_meta,
        "filter": {"increment": cfg.increment, "runtime_s": res.meta["runtime_s"], "n_imu": res.meta["n_imu"],
                   "n_updates": res.meta["n_updates"]},
        "t": t, "ours": ours, "ekf2": ekf2, "diff": diff,
        "nis": _nis_entries(res.innovations, t0, t1, rate_hz),
        "events": {"ekf2_resets_s": resets, "skip_init_s": metrics.SKIP_INIT_S},
    }


# ---------------------------------------------------------------------------
# replay_to_json
# ---------------------------------------------------------------------------
def _numbers(flat: list, fmt: str, n: int) -> str:
    """``n`` numbers of ``flat`` formatted with the %-template ``fmt`` (one giant ``%`` is fastest in CPython)."""
    return (",".join([fmt] * n)) % tuple(flat) if n else ""


def _enc_array(a: np.ndarray) -> str:
    """JSON text of a numpy array (nested lists); floats with 7 significant digits, NaN/inf -> null."""
    if a.ndim == 0:
        return _enc(a.item())
    if a.dtype == bool:
        flat = ["true" if v else "false" for v in a.ravel().tolist()]
        text_fmt = "%s"
    elif a.dtype.kind in "iu":
        flat = a.ravel().tolist()
        text_fmt = "%d"
    else:
        flat = a.astype(float, copy=False).ravel().tolist()
        text_fmt = "%.7g"
    if a.size == 0:
        return "[]"
    if a.ndim == 1:
        s = "[" + _numbers(flat, text_fmt, a.size) + "]"
    elif a.ndim == 2:
        k = a.shape[1]
        row = "[" + ",".join([text_fmt] * k) + "]"
        s = "[" + ",".join([row] * a.shape[0]) % tuple(flat) + "]"
    else:
        return "[" + ",".join(_enc_array(sub) for sub in a) + "]"
    if a.dtype.kind == "f":
        s = s.replace("-inf", "null").replace("inf", "null").replace("nan", "null")
    return s


def _enc(o) -> str:
    """JSON text of a (nested) replay object: dict, list/tuple, ndarray, numpy scalar, str, bool, int, float, None."""
    if o is None:
        return "null"
    if isinstance(o, dict):
        return "{" + ",".join(json.dumps(str(k)) + ":" + _enc(v) for k, v in o.items()) + "}"
    if isinstance(o, np.ndarray):
        return _enc_array(o)
    if isinstance(o, np.generic):
        return _enc(o.item())
    if isinstance(o, (bool, np.bool_)):
        return "true" if o else "false"
    if isinstance(o, int):
        return str(o)
    if isinstance(o, float):
        return f"{o:.7g}" if math.isfinite(o) else "null"
    if isinstance(o, str):
        return json.dumps(o)
    if isinstance(o, (list, tuple)):
        if o and all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in o):
            return _enc_array(np.asarray(o))
        return "[" + ",".join(_enc(v) for v in o) + "]"
    return json.dumps(str(o))


def replay_to_json(rep: dict) -> str:
    """Serialise a ``build_replay`` dict to JSON text.

    Floats are written with 7 significant digits (``%.7g``), NaN/+-inf as ``null``, booleans as JSON
    booleans, ints as ints; (n, k) arrays become lists of lists. Vectorised: one ``%`` formatting call per
    array (about 0.3 s for the 34 k-sample grid of the 675 s log).
    """
    return _enc(rep)


# ---------------------------------------------------------------------------
# ReplayStore
# ---------------------------------------------------------------------------
def _flatten_arrays(rep: dict) -> dict:
    """Numeric leaves of a replay dict as ``{"ours.p_ned_m": array, ...}`` (for the offline ``.npz``)."""
    out = {}

    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(o, np.ndarray) and o.dtype.kind in "biuf":
            out[prefix] = o
        elif isinstance(o, (bool, int, float, np.number)) and prefix:
            out[prefix] = np.asarray(o)

    walk("", {k: v for k, v in rep.items() if k not in ("log", "caveat")})
    return out


def _json_default(o):
    return str(o)


class ReplayStore:
    """Resolves pinned logs, builds their replay once and caches the JSON on disk (spec section 1.3).

    ``context_loader(entry) -> (streams, ref)`` and ``cfg_factory() -> EkfConfig`` default to
    ``sensor_fusion.run_flight`` (``load_context``, ``build_config(3)``), imported lazily. Thread safe:
    at most one build per log id at a time; a second caller waits and then reads the cache.
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, logs=PINNED_LOGS, data_dir=DATA_DIR,
                 context_loader: Callable | None = None, cfg_factory: Callable | None = None,
                 rate_hz: float = DEFAULT_RATE_HZ) -> None:
        self.cache_dir = Path(cache_dir)
        self.logs = tuple(logs)
        self.data_dir = Path(data_dir)
        self._context_loader = context_loader
        self._cfg_factory = cfg_factory
        self.rate_hz = float(rate_hz)
        self.n_computed = 0
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # -- lookup ---------------------------------------------------------
    def path(self, entry: dict) -> Path:
        """Location of the entry's ``.ulg`` file."""
        return self.data_dir / f"{entry['id']}.ulg"

    def resolve(self, token) -> dict:
        """Entry for a role, a full id or an id prefix of >= 4 characters matching exactly one entry."""
        tok = "" if token is None else str(token)
        hits = [e for e in self.logs if e["role"] == tok or e["id"] == tok]
        if not hits and len(tok) >= MIN_TOKEN_PREFIX:
            hits = [e for e in self.logs if e["id"].startswith(tok)]
        if len(hits) != 1:
            raise UnknownLog(f"unknown log '{token}'")
        return hits[0]

    def _cfg(self):
        if self._cfg_factory is not None:
            return self._cfg_factory()
        from sensor_fusion.run_flight import build_config
        return build_config(3)

    def _load(self, entry: dict):
        if self._context_loader is not None:
            return self._context_loader(entry)
        from sensor_fusion.run_flight import load_context
        ctx = load_context(entry)
        return ctx["S"], ctx["ref"]

    # -- cache ----------------------------------------------------------
    def _key(self, entry: dict, cfg) -> str:
        st = self.path(entry).stat()
        blob = json.dumps({"cache_version": CACHE_VERSION, "id": entry["id"], "rate_hz": self.rate_hz,
                           "config": dataclasses.asdict(cfg), "ulg_size": st.st_size,
                           "ulg_mtime_ns": st.st_mtime_ns}, sort_keys=True, default=_json_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _files(self, entry: dict, key: str) -> tuple[Path, Path]:
        stem = f"px4_{entry['id'][:8]}_{key}"
        return self.cache_dir / f"{stem}.json", self.cache_dir / f"{stem}.npz"

    def list_logs(self) -> list[dict]:
        """Per pinned log ``{"id", "role", "notes", "present", "cached"}``."""
        cfg = None
        out = []
        for e in self.logs:
            present = self.path(e).is_file()
            cached = False
            if present:
                try:
                    cfg = cfg if cfg is not None else self._cfg()
                    cached = self._files(e, self._key(e, cfg))[0].is_file()
                except OSError:
                    cached = False
            out.append({"id": e["id"], "role": e["role"], "notes": e["notes"], "present": present,
                        "cached": cached})
        return out

    def _id_lock(self, log_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(log_id, threading.Lock())

    @staticmethod
    def _atomic_write(path: Path, writer: Callable) -> None:
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp, "wb") as f:
                writer(f)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # -- main entry -----------------------------------------------------
    def get_json(self, token) -> str:
        """The replay JSON text for ``token`` (cached on disk).

        Raises ``UnknownLog`` (bad token) or ``LogMissing`` (``.ulg`` not downloaded; message contains
        ``DOWNLOAD_HINT``); any loader/filter exception propagates.
        """
        entry = self.resolve(token)
        if not self.path(entry).is_file():
            raise LogMissing(f"log {entry['id']} ({entry['role']}) is not downloaded; run: {DOWNLOAD_HINT}")
        cfg = self._cfg()
        key = self._key(entry, cfg)
        jpath, npath = self._files(entry, key)
        if jpath.is_file():
            return jpath.read_text(encoding="utf-8")
        with self._id_lock(entry["id"]):
            if jpath.is_file():                                   # built by another thread meanwhile
                return jpath.read_text(encoding="utf-8")
            streams, ref = self._load(entry)
            rep = build_replay(streams, ref, cfg, log_meta=self._log_meta(entry, streams), rate_hz=self.rate_hz)
            text = replay_to_json(rep)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            arrays = _flatten_arrays(rep)
            self._atomic_write(npath, lambda f: np.savez_compressed(f, **arrays))
            self._atomic_write(jpath, lambda f: f.write(text.encode("utf-8")))   # the hit marker goes last
            self.n_computed += 1
            return text

    @staticmethod
    def _log_meta(entry: dict, streams) -> dict:
        """Header of the replay: ids, notes, IMU span [s] and the origin / declination / gravity of the streams."""
        def num(x):
            try:
                x = float(x)
            except (TypeError, ValueError):
                return None
            return x if math.isfinite(x) else None

        lla = getattr(streams, "origin_lla", None)
        lat = lon = None
        if lla is not None and len(lla) >= 2:
            lat, lon = num(math.degrees(float(lla[0]))), num(math.degrees(float(lla[1])))
        imu_t = np.asarray(streams.imu_t, float)
        decl = num(streams.declination)
        meta = getattr(streams, "meta", None) or {}
        utc = meta.get("gps_utc_start")
        return {"id": entry["id"], "role": entry["role"], "notes": entry["notes"],
                "duration_s": num(imu_t[-1] - imu_t[0]) if imu_t.size else None,
                "origin_lat_deg": lat, "origin_lon_deg": lon,
                "gps_utc_start": None if utc is None else str(utc),
                "declination_deg": None if decl is None else math.degrees(decl),
                "g_m_s2": num(streams.g)}

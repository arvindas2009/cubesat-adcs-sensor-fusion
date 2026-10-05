"""Reproducible download of the PX4 flight logs used by Project A.

Run ``.venv/Scripts/python -m sensor_fusion.download_logs`` to fetch the pinned logs
into ``sensor_fusion/data/`` (``*.ulg`` is gitignored).  Nothing here queries a
"latest" listing at run time: the three log IDs are hard-coded in ``PINNED_LOGS``.

Screening criteria (applied once, by hand + ``--screen``, to pick the pinned IDs)
-------------------------------------------------------------------------------
* multirotor airframe (Flight Review "Quadrotor"), real hardware (not ``PX4_SITL``);
* GPS 3D fix for the bulk of the flight;
* duration 2-15 min;
* contains topics ``sensor_combined`` (or ``vehicle_imu``), ``vehicle_magnetometer``
  (or ``sensor_mag``), ``vehicle_air_data``, ``vehicle_gps_position``,
  ``vehicle_local_position``, ``vehicle_attitude``;
* file size < ~40 MB; IMU (``sensor_combined``) rate >= 200 Hz preferred;
* at least one hover-like (gentle position-hold) and one aggressive-manoeuvre
  (manual / acro / stabilized, large body rates) flight.

``--screen`` lists candidate rows from the Flight Review browse endpoint (one
request) and is never needed at run time.  ``--check <id> ...`` downloads the
given logs and prints the topics / rates / angular-rate statistics used to verify a
candidate with pyulog.

Server politeness: one session, explicit timeouts, <= 3 retries with exponential
backoff, existing files are never re-downloaded.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://review.px4.io"
ULOG_MAGIC = b"ULog\x01\x12\x35"
USER_AGENT = "ubeSat-sensor-fusion-student-project/0.1 (research; contact via GitHub)"
TIMEOUT_S = (15, 120)  # (connect, read)
MAX_RETRIES = 3

# Picked once by the screening step described above, then hard-coded.
# See sensor_fusion/data/NOTES.md for the per-log verification.
PINNED_LOGS: tuple[dict, ...] = (
    {"id": "f4e81624-7ccc-4859-bb45-d02fa0246b77", "role": "hover",
     "notes": "PX4_FMU_V6C, PX4 v1.17.0, Position mode, 281 s, gentle flight (max 2.2 m/s, p99 |gyro| 26 deg/s), "
              "|B| = 0.482 +/- 0.008 G; sensor_combined ~221 Hz"},
    {"id": "7cd0a812-d6d6-4c67-9da8-1f3a9cc460b0", "role": "aggressive",
     "notes": "Holybro S500 (4015), PX4_FMU_V6C, PX4 v1.17.0, Stabilized (manual) flight, 170 s, "
              "max |gyro| 386 deg/s (p99 115 deg/s), 4 m/s horizontal / 2.2 m/s vertical; sensor_combined ~221 Hz"},
    {"id": "78808b7c-d9e9-4350-b311-3bb510f29a7b", "role": "other",
     "notes": "Generic quad (4002), PX4_FMU_V5, PX4 v1.14.0, Position mode, 675 s, up to 9 m/s and 24 m height, "
              "RTK GPS (fix 6) at 2 Hz in the OLD integer-degree GPS message format; sensor_combined ~197 Hz"},
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get_with_retries(session, url, *, params=None, stream=False):
    """GET with timeout and <= MAX_RETRIES retries (exponential backoff 2, 4, 8 s)."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT_S, stream=stream)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 401, 403, 404):  # not retryable
                r.raise_for_status()
            last_exc = requests.HTTPError(f"HTTP {r.status_code} for {url}")
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        if attempt < MAX_RETRIES:
            time.sleep(2.0 ** (attempt + 1))
    raise RuntimeError(f"download failed after {MAX_RETRIES + 1} attempts: {url}") from last_exc


def _has_ulog_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(len(ULOG_MAGIC)) == ULOG_MAGIC
    except OSError:
        return False


def download_log(log_id: str, dest_dir: Path | str = DATA_DIR, session=None) -> Path:
    """Download ``review.px4.io/download?log=<id>`` to ``dest_dir/<id>.ulg``.

    Skipped if the file exists and is non-empty.  The ULog magic bytes are verified;
    a file failing the check is deleted and ``ValueError`` raised.
    """
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", str(log_id)):
        raise ValueError(f"not a Flight Review log UUID: {log_id!r}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{log_id}.ulg"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    session = session or _session()
    tmp = dest.with_suffix(".ulg.part")
    r = _get_with_retries(session, f"{BASE_URL}/download", params={"log": log_id}, stream=True)
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
    if not _has_ulog_magic(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{log_id}: downloaded file is not a ULog (bad magic bytes)")
    tmp.replace(dest)
    return dest


def download_pinned(dest_dir: Path | str = DATA_DIR) -> list[Path]:
    """Download every entry of ``PINNED_LOGS`` (sequentially, one session)."""
    session = _session()
    paths = []
    for entry in PINNED_LOGS:
        paths.append(download_log(entry["id"], dest_dir, session))
        time.sleep(1.0)  # be polite between files
    return paths


# --------------------------------------------------------------------------- screening
def _duration_s(text: str) -> int:
    return sum(int(a) * {"h": 3600, "m": 60, "s": 1}[b] for a, b in re.findall(r"(\d+)([hms])", text))


def screen(n_rows: int = 500, min_s: int = 120, max_s: int = 900, session=None) -> list[dict]:
    """One request to the Flight Review browse endpoint; returns candidate rows.

    Filters: Quadrotor, not SITL, duration in [min_s, max_s].  Only used for the
    one-off screening; the result is not used at run time.
    """
    params = {"draw": 1, "start": 0, "length": n_rows, "search[value]": "",
              "order[0][column]": 1, "order[0][dir]": "desc"}
    for i in range(10):
        params[f"columns[{i}][data]"] = i
        params[f"columns[{i}][searchable]"] = "true"
        params[f"columns[{i}][orderable]"] = "true"
    r = _get_with_retries(session or _session(), f"{BASE_URL}/browse_data_retrieval", params=params)
    out = []
    for row in r.json()["data"]:
        m = re.search(r"log=([0-9a-f-]+)", row[1])
        if not m or row[3] != "Quadrotor" or row[5] == "PX4_SITL":
            continue
        dur = _duration_s(row[7])
        if min_s <= dur <= max_s:
            out.append({"id": m.group(1), "airframe": row[4], "hardware": row[5],
                        "software": row[6], "duration": row[7], "start": row[8], "modes": row[9]})
    return out


def check_candidates(ids, dest_dir: Path | str = DATA_DIR) -> None:
    """Download candidate logs and print what pyulog finds (topics, rates, rate stats)."""
    import numpy as np
    from pyulog import ULog

    need = ["sensor_combined", "vehicle_imu", "vehicle_magnetometer", "sensor_mag", "vehicle_air_data",
            "vehicle_gps_position", "vehicle_local_position", "vehicle_attitude", "vehicle_status"]
    session = _session()
    for lid in ids:
        path = download_log(lid, dest_dir, session)
        ulog = ULog(str(path), message_name_filter_list=need)
        print(f"== {lid}  {path.stat().st_size / 1e6:.1f} MB")
        for name in need:
            ds = [d for d in ulog.data_list if d.name == name]
            if not ds:
                print(f"   {name:24s} MISSING")
                continue
            for d in ds:
                t = d.data["timestamp"] * 1e-6
                rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else float("nan")
                print(f"   {name:24s} multi_id={d.multi_id} n={len(t):7d} rate~{rate:7.1f} Hz")
        for d in ulog.data_list:
            if d.name == "sensor_combined":
                g = np.hypot.reduce([d.data[f"gyro_rad[{i}]"] for i in range(3)])
                print(f"   max |gyro| = {np.degrees(g.max()):.0f} deg/s, p99 = {np.degrees(np.percentile(g, 99)):.0f} deg/s")
        time.sleep(1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--screen", action="store_true", help="list candidate logs from Flight Review (1 request)")
    ap.add_argument("--check", nargs="+", metavar="ID", help="download + inspect candidate log IDs")
    ap.add_argument("--dest", default=str(DATA_DIR))
    args = ap.parse_args(argv)
    if args.screen:
        for row in screen():
            print(row)
        return 0
    if args.check:
        check_candidates(args.check, args.dest)
        return 0
    for p in download_pinned(args.dest):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

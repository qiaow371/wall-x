#!/usr/bin/env python3
"""Training-sidecar hardware sampler (NVIDIA / 江算 PPU / 昇腾 NPU).

Does not import torch and does not touch the training process. Polls vendor
CLIs, writes JSONL next to the train log, and on exit writes a summary used
to later retune local batch / accum.

Usage (start scripts already do this):

  python3 hw_train_monitor.py --out /path/train.log.hw.jsonl --pid $$ --devices 0,1,2,3

Attach to an already-running job (does not restart it):

  python3 hw_train_monitor.py --attach
  python3 hw_train_monitor.py --out /path/hw.jsonl --watch-regex 'train_from_yaml.py --config configs/foo.yaml' --devices 8,9,10,11,12,13,14,15

Summarize later:

  python3 hw_train_monitor.py --summary /path/hw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

PPU_SMI_BIN = "/usr/local/PPU_SDK/ppu-smi/bin"

_STOP = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace("MiB", "").replace("W", "")
    if not s or s in ("[N/A]", "N/A", "-", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _intish(x: Any) -> Optional[int]:
    n = _num(x)
    return int(n) if n is not None else None


def _which(cmd: str) -> Optional[str]:
    path = os.environ.get("PATH", "")
    extra = [PPU_SMI_BIN, "/usr/local/sbin", "/usr/local/bin", "/usr/bin"]
    os.environ["PATH"] = ":".join(extra + ([path] if path else []))
    return shutil.which(cmd)


def detect_backend() -> str:
    # 江算机同时有 ppu-smi 和 nvidia-smi 兼容壳；以 PPU 为准。
    if _which("ppu-smi"):
        return "ppu"
    if _which("npu-smi"):
        return "npu"
    if _which("nvidia-smi"):
        return "nvidia"
    return "none"


def _run(cmd: List[str], timeout: float = 8.0) -> str:
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False, text=True,
        )
        return p.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_smi_csv(raw: str) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip().replace("\r", "")
        if not line or not re.match(r"^\d+", line):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        rec: Dict[str, Any] = {
            "id": _intish(parts[0]),
            "util": _num(parts[1]),
            "mem_used_mib": _num(parts[2]),
            "mem_total_mib": _num(parts[3]),
            "temp_c": _num(parts[4]) if len(parts) > 4 else None,
            "power_w": _num(parts[5]) if len(parts) > 5 else None,
        }
        if rec["id"] is None:
            continue
        cards.append(rec)
    return cards


def cards_from_npu(raw: str) -> List[Dict[str, Any]]:
    """Port of fleet_monitor.cards_from_npu: Phy-ID is the torch_npu device id."""
    out: Dict[int, Dict[str, Any]] = {}
    pend: Dict[str, Any] = {}
    for line in raw.splitlines():
        c = [x.strip() for x in line.split("|")]
        if len(c) < 4 or not re.match(r"^\d+\b", c[1]):
            continue
        nums = re.findall(r"\d+", c[1])
        if ":" in c[2]:
            rec = dict(pend)
            body = " ".join(c[3:])
            mt = re.match(r"(\d+)\b", body)
            if mt:
                rec["util"] = _num(mt.group(1))
            mem = None
            for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", body):
                if int(b) > 0:
                    mem = (float(a), float(b))
            if mem:
                rec["mem_used_mib"], rec["mem_total_mib"] = mem
            cid = int(nums[1] if len(nums) > 1 else nums[0])
            cur = out.setdefault(cid, {"id": cid})
            cur.update({k: v for k, v in rec.items() if v is not None})
            pend = {}
            continue
        body = c[3] if c[2] in ("OK", "Warning", "Critical") else c[2]
        hw = re.match(r"([\d.]+|-)\s+(\d+)\b", body)
        pend = {}
        if hw:
            if hw.group(1) != "-":
                pend["power_w"] = _num(hw.group(1))
            pend["temp_c"] = _num(hw.group(2))
    return [out[k] for k in sorted(out)]


def sample_cards(backend: str) -> List[Dict[str, Any]]:
    if backend == "ppu":
        raw = _run([
            "ppu-smi",
            "--query-ppu=index,utilization.ppu,memory.used,memory.total,temperature.ppu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        return parse_smi_csv(raw)
    if backend == "nvidia":
        raw = _run([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        return parse_smi_csv(raw)
    if backend == "npu":
        return cards_from_npu(_run(["npu-smi", "info"], timeout=20.0))
    return []


def host_stats() -> Dict[str, Any]:
    load1 = load5 = load15 = None
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as f:
            a, b, c = f.read().split()[:3]
            load1, load5, load15 = float(a), float(b), float(c)
    except (OSError, ValueError):
        pass
    mem_pct = None
    try:
        info: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                k, rest = line.split(":", 1)
                info[k] = int(rest.strip().split()[0])
        tot, avail = info.get("MemTotal") or 0, info.get("MemAvailable") or 0
        if tot:
            mem_pct = round((tot - avail) * 100.0 / tot, 1)
    except (OSError, ValueError):
        pass
    cores = os.cpu_count()
    return {"load1": load1, "load5": load5, "load15": load15, "mem_pct": mem_pct, "cores": cores}


def filter_devices(cards: List[Dict[str, Any]], devices: Optional[List[int]]) -> List[Dict[str, Any]]:
    if not devices:
        return cards
    want = set(devices)
    return [c for c in cards if c.get("id") in want]


def parse_devices(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        s = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or ""
    s = s.strip()
    if not s or s.lower() == "all":
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or None


def aggregate(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    utils = [c["util"] for c in cards if c.get("util") is not None]
    mem_u = [c["mem_used_mib"] for c in cards if c.get("mem_used_mib") is not None]
    mem_t = [c["mem_total_mib"] for c in cards if c.get("mem_total_mib") is not None]
    pwr = [c["power_w"] for c in cards if c.get("power_w") is not None]
    fracs = []
    for c in cards:
        u, t = c.get("mem_used_mib"), c.get("mem_total_mib")
        if u is not None and t:
            fracs.append(u / t)
    n_busy = sum(1 for u in utils if u >= 50)
    return {
        "n_cards": len(cards),
        "n_busy": n_busy,
        "util_avg": round(sum(utils) / len(utils), 2) if utils else None,
        "mem_used_mib_avg": round(sum(mem_u) / len(mem_u), 1) if mem_u else None,
        "mem_total_mib": round(sum(mem_t) / len(mem_t), 1) if mem_t else None,
        "mem_frac_avg": round(sum(fracs) / len(fracs), 4) if fracs else None,
        "power_w_avg": round(sum(pwr) / len(pwr), 1) if pwr else None,
    }


def one_sample(backend: str, devices: Optional[List[int]]) -> Dict[str, Any]:
    cards = filter_devices(sample_cards(backend), devices)
    rec: Dict[str, Any] = {
        "ts": _now_iso(),
        "backend": backend,
        "devices": devices,
        "cards": cards,
    }
    rec.update(aggregate(cards))
    rec["host"] = host_stats()
    return rec


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    frac = k - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def hint(util_avg: Optional[float], mem_frac: Optional[float], util_p05: Optional[float]) -> str:
    if util_avg is None:
        return "no samples"
    if util_avg < 20 or (util_p05 is not None and util_p05 < 5 and util_avg < 40):
        return "stalled or idle — check dataloader / NCCL hang / AICore 0%"
    if util_avg < 70:
        return "compute underused — data pipeline or batch too small"
    if mem_frac is not None and mem_frac < 0.70 and util_avg >= 90:
        return "compute-bound with HBM headroom — try larger local batch"
    if mem_frac is not None and mem_frac >= 0.90:
        return "near memory cap — do not raise batch"
    return "saturated compute; memory ok"


def summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    utils, fracs, mems, pwrs = [], [], [], []
    for s in samples:
        if s.get("util_avg") is not None:
            utils.append(float(s["util_avg"]))
        if s.get("mem_frac_avg") is not None:
            fracs.append(float(s["mem_frac_avg"]))
        if s.get("mem_used_mib_avg") is not None:
            mems.append(float(s["mem_used_mib_avg"]))
        if s.get("power_w_avg") is not None:
            pwrs.append(float(s["power_w_avg"]))
    util_avg = round(sum(utils) / len(utils), 2) if utils else None
    mem_frac = round(sum(fracs) / len(fracs), 4) if fracs else None
    util_p05 = round(percentile(utils, 0.05), 2) if utils else None
    first_ts, last_ts = (samples[0].get("ts"), samples[-1].get("ts")) if samples else (None, None)
    out = {
        "n_samples": len(samples),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "backend": samples[-1].get("backend") if samples else None,
        "devices": samples[-1].get("devices") if samples else None,
        "util_avg": util_avg,
        "util_p05": util_p05,
        "util_p50": round(percentile(utils, 0.50), 2) if utils else None,
        "util_p95": round(percentile(utils, 0.95), 2) if utils else None,
        "mem_frac_avg": mem_frac,
        "mem_frac_p95": round(percentile(fracs, 0.95), 4) if fracs else None,
        "mem_used_mib_avg": round(sum(mems) / len(mems), 1) if mems else None,
        "mem_total_mib": samples[-1].get("mem_total_mib") if samples else None,
        "power_w_avg": round(sum(pwrs) / len(pwrs), 1) if pwrs else None,
        "n_cards": samples[-1].get("n_cards") if samples else None,
    }
    out["hint"] = hint(util_avg, mem_frac, util_p05)
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def regex_alive(pattern: str) -> bool:
    raw = _run(["ps", "-eo", "args"], timeout=5.0)
    rx = re.compile(pattern)
    for line in raw.splitlines():
        if "hw_train_monitor" in line:
            continue
        if rx.search(line):
            return True
    return False


def _handle_stop(signum: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def loop(args: argparse.Namespace) -> int:
    backend = detect_backend()
    devices = parse_devices(args.devices)
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    latest = out + ".latest.json"
    summary_path = out + ".summary.json"
    if out.endswith(".jsonl"):
        latest = out[:-6] + ".latest.json"
        summary_path = out[:-6] + ".summary.json"

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    try:
        signal.signal(signal.SIGHUP, _handle_stop)
    except (OSError, ValueError):
        pass

    samples: List[Dict[str, Any]] = []
    deadline = time.time() + max(0, args.wait_start)
    saw_target = False
    t0 = time.time()

    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": _now_iso(), "event": "start", "backend": backend,
            "devices": devices, "pid": args.pid, "watch_regex": args.watch_regex,
            "interval_s": args.interval, "host": os.uname().nodename if hasattr(os, "uname") else "",
        }, ensure_ascii=False) + "\n")
        fh.flush()

        while not _STOP:
            rec = one_sample(backend, devices)
            rec["uptime_s"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            samples.append(rec)
            write_json(latest, rec)

            if args.once:
                break

            alive = True
            if args.pid is not None:
                alive = pid_alive(args.pid)
            elif args.watch_regex:
                alive = regex_alive(args.watch_regex)
            if alive:
                saw_target = True
            if args.pid is not None and not alive and saw_target:
                break
            if args.watch_regex and not alive:
                if saw_target or time.time() >= deadline:
                    if saw_target:
                        break
                    if time.time() >= deadline:
                        rec_end = {"ts": _now_iso(), "event": "watch_timeout", "watch_regex": args.watch_regex}
                        fh.write(json.dumps(rec_end, ensure_ascii=False) + "\n")
                        break
            if args.pid is not None and not alive and not saw_target:
                if time.time() >= deadline:
                    break

            time.sleep(max(1.0, args.interval))

        summary = summarize([s for s in samples if "cards" in s])
        summary["event"] = "summary"
        summary["ts"] = _now_iso()
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        fh.flush()

    write_json(summary_path, summary)
    print("[hw-monitor] summary -> %s  util_avg=%s mem_frac=%s  %s" % (
        summary_path, summary.get("util_avg"), summary.get("mem_frac_avg"), summary.get("hint"),
    ), file=sys.stderr)
    return 0


NPU_FIXTURE = """
| 0     Ascend910           | OK            | 570.0       48                0    / 0             |
| 0     1                   | 0000:01:00.0  | 97          0    / 0          37508/ 65536         |
| 1     Ascend910           | OK            | 568.2       47                0    / 0             |
| 1     2                   | 0000:02:00.0  | 96          0    / 0          38000/ 65536         |
"""


def self_test() -> int:
    csv = "8, 100 %, 55020 MiB, 98304 MiB, 62, 320.5\n9, 99, 55100, 98304, 61, 318\n"
    cards = parse_smi_csv(csv)
    assert cards[0]["id"] == 8 and cards[0]["util"] == 100.0
    assert cards[0]["mem_used_mib"] == 55020
    npu = cards_from_npu(NPU_FIXTURE)
    assert [c["id"] for c in npu] == [1, 2], npu
    assert npu[0]["util"] == 97 and npu[0]["mem_total_mib"] == 65536
    agg = aggregate(cards)
    assert agg["n_cards"] == 2 and agg["util_avg"] == 99.5
    xs = [float(i) for i in range(11)]
    assert abs(percentile(xs, 0.5) - 5.0) < 1e-6
    h = hint(99.0, 0.56, 95.0)
    assert "headroom" in h
    print("self-test ok")
    return 0


def print_summary(path: str) -> int:
    rows = load_jsonl(path)
    samples = [r for r in rows if "cards" in r]
    if not samples:
        # maybe it's already a summary json
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(summarize(samples), ensure_ascii=False, indent=2))
    return 0


TRAIN_MARKERS = (
    "train_from_yaml.py",
    "run_sft.py",
    "/train.py",
    "scripts/train.py",
)
SKIP_MARKERS = (
    "hw_train_monitor",
    "attach_hw_monitor",
    "fleet_probe",
    "occupy",
    "nvidia-smi",
    "ppu-smi",
    "npu-smi",
)


def _read_proc(pid: int) -> Tuple[str, str]:
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return "", ""
    env = ""
    try:
        with open("/proc/%d/environ" % pid, "rb") as f:
            env = f.read().replace(b"\x00", b"\n").decode("utf-8", "replace")
    except OSError:
        pass
    return cmd, env


def _env_val(env: str, key: str) -> str:
    for line in env.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return ""


def _log_from_fds(pid: int) -> Optional[str]:
    fd_dir = "/proc/%d/fd" % pid
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return None
    hits: List[str] = []
    for name in names:
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if target.endswith((".log", ".out", ".txt")) and "hw.jsonl" not in target:
            hits.append(target)
    for path in hits:
        if "train" in os.path.basename(path).lower() or path.endswith("nohup.out"):
            return path
    return hits[0] if hits else None


def discover_jobs() -> List[Dict[str, Any]]:
    jobs: Dict[str, Dict[str, Any]] = {}
    try:
        pids = [int(x) for x in os.listdir("/proc") if x.isdigit()]
    except OSError:
        return []
    for pid in pids:
        cmd, env = _read_proc(pid)
        if not cmd:
            continue
        if any(s in cmd for s in SKIP_MARKERS):
            continue
        if cmd.startswith("bash ") or cmd.startswith("sh ") or "bash -lc" in cmd or cmd.startswith("docker "):
            continue
        if not any(m in cmd for m in TRAIN_MARKERS):
            continue
        key = cmd
        mt = re.search(r"configs/\S+\.ya?ml", cmd) or re.search(r"--config\s+(\S+)", cmd)
        if mt:
            key = mt.group(0) if mt.lastindex is None else (mt.group(1) or mt.group(0))
        devices = _env_val(env, "CUDA_VISIBLE_DEVICES") or _env_val(env, "ASCEND_RT_VISIBLE_DEVICES")
        log = _log_from_fds(pid)
        rec = jobs.get(key)
        if rec is None or pid < rec["pid"]:
            jobs[key] = {
                "pid": pid,
                "key": key,
                "devices": devices,
                "log": log,
                "cmd": cmd[:240],
            }
    out: List[Dict[str, Any]] = []
    for rec in jobs.values():
        log = rec.get("log")
        if log:
            rec["out"] = log + ".hw.jsonl"
        else:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", rec["key"])[-80:]
            rec["out"] = "/tmp/hw_monitor_%s.jsonl" % slug
        out.append(rec)
    return sorted(out, key=lambda r: r["pid"])


def already_monitoring(pid: int, out_path: str) -> bool:
    raw = _run(["ps", "-eo", "args"], timeout=5.0)
    for line in raw.splitlines():
        if "hw_train_monitor.py" not in line:
            continue
        if ("--pid %d" % pid) in line or out_path in line:
            return True
    return False


def attach_jobs(interval: float) -> int:
    jobs = discover_jobs()
    if not jobs:
        print("[hw-monitor] no running train jobs")
        return 0
    n = 0
    for job in jobs:
        if already_monitoring(int(job["pid"]), job["out"]):
            print("[hw-monitor] skip already watching pid=%s out=%s" % (job["pid"], job["out"]))
            continue
        os.makedirs(os.path.dirname(os.path.abspath(job["out"])) or ".", exist_ok=True)
        err = job["out"][:-6] + ".err" if job["out"].endswith(".jsonl") else job["out"] + ".err"
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--out", job["out"],
            "--interval", str(interval),
            "--pid", str(job["pid"]),
            "--devices", job.get("devices") or "",
            "--wait-start", "30",
        ]
        with open(err, "a", encoding="utf-8") as ef:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
        print("[hw-monitor] attach pid=%s devices=%s out=%s cmd=%s" % (
            job["pid"], job.get("devices") or "all", job["out"], job["cmd"],
        ))
        n += 1
    print("[hw-monitor] attached %d job(s)" % n)
    return 0


def start_from_trainer(out_path: Any, devices: Optional[str] = None) -> None:
    """Rank-0 helper: spawn a sidecar that watches this training process.

    No-op on other ranks and when HW_MONITOR=0. Does not import torch.
    """
    if os.environ.get("HW_MONITOR", "1") == "0":
        return
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0") or 0))
    except ValueError:
        rank = 0
    if rank != 0:
        return
    out = str(out_path)
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    err = out[:-6] + ".err" if out.endswith(".jsonl") else out + ".err"
    vis = devices
    if vis is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or ""
    argv = [
        sys.executable, os.path.abspath(__file__),
        "--out", out,
        "--interval", os.environ.get("HW_MONITOR_INTERVAL", "30"),
        "--pid", str(os.getpid()),
        "--devices", vis,
        "--wait-start", "120",
    ]
    with open(err, "a", encoding="utf-8") as ef:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=ef, start_new_session=True)
    print("[hw-monitor] sidecar -> %s watch_pid=%s devices=%s" % (out, os.getpid(), vis or "all"), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="JSONL path")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between samples (default 30)")
    ap.add_argument("--pid", type=int, default=None, help="stop when this pid exits (start-script $$ before exec)")
    ap.add_argument("--watch-regex", default=None, help="stop when no ps args match this regex")
    ap.add_argument("--wait-start", type=float, default=90.0, help="seconds to wait for --pid/--watch-regex to appear")
    ap.add_argument("--devices", default=None, help="physical ids, comma-separated; default CUDA/ASCEND visible")
    ap.add_argument("--once", action="store_true", help="one sample then exit")
    ap.add_argument("--summary", metavar="FILE", help="print summary of an existing jsonl")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--discover", action="store_true", help="print running train jobs as JSONL")
    ap.add_argument("--attach", action="store_true", help="start sidecars for running trains (does not restart them)")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.summary:
        return print_summary(args.summary)
    if args.discover:
        for job in discover_jobs():
            print(json.dumps(job, ensure_ascii=False))
        return 0
    if args.attach:
        return attach_jobs(args.interval)
    if args.once and not args.out:
        rec = one_sample(detect_backend(), parse_devices(args.devices))
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 0
    if not args.out:
        ap.error("--out is required unless --once/--summary/--self-test/--discover/--attach")
    if args.pid is None and not args.watch_regex and not args.once:
        args.pid = os.getppid()
    return loop(args)


if __name__ == "__main__":
    sys.exit(main())

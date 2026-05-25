#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_VERSION = 1
REPO = Path(__file__).resolve().parents[1]
REPO_PRESETS = REPO / "presets" / "moe"
MIB = 1024 ** 2

BASE_ENV = {
    "MLA": "3",
    "CPU_MOE": "1",
    "IGNORE_EOS": "1",
    "GGML_CUDA_NO_PINNED": "1",
    "GGML_MOE_STREAM": "1",
    "GGML_MOE_STREAM_BATCH_ONLY": "1",
    "GGML_MOE_STREAM_DEFER": "1",
    "GGML_MOE_STREAM_CPU_OPS": "1",
    "GGML_MOE_PARALLEL_EXPERTS": "1",
    "GGML_MOE_STREAM_FUSED_UP_GATE": "1",
    "GGML_MOE_VRAM_CACHE_SPLIT": "1",
    "GGML_MOE_STREAM_ONE_CACHE_MIB": "0",
    "GGML_MOE_VRAM_CACHE_POLICY": "lfu_lru",
    "GGML_MOE_STAGE_PINNED_SLOTS": "16",
    "GGML_MOE_PREFETCH_DOWN": "0",
    "GGML_MOE_PREFETCH_DOWN_DEPTH": "8",
}

CHAT_DEFAULT_ARGS = [
    ("-b", "2048"),
    ("-n", "96"),
    ("--temp", "0"),
    ("--top-p", "1.0"),
    ("--min-p", "0.0"),
    ("--repeat-last-n", "512"),
    ("--repeat-penalty", "1.20"),
    ("--presence-penalty", "0.1"),
    ("--frequency-penalty", "0.2"),
]

PACK_HEADER = struct.Struct("<16sIIQQ")
PACK_ENTRY = struct.Struct("<128siIQQ")
PACK_MAGIC = b"GGMLMOEPACKv1\0\0\0"


@dataclass(frozen=True)
class GpuInfo:
    name: str
    total_mib: int
    free_mib: int
    bus_id: str
    driver: str


@dataclass(frozen=True)
class MemInfo:
    total_mib: int
    available_mib: int


@dataclass(frozen=True)
class CpuInfo:
    model_name: str
    cpu_count: int
    threads_per_core: int
    cores_per_socket: int
    sockets: int
    numa_nodes: int


@dataclass(frozen=True)
class StorageInfo:
    source: str
    fstype: str
    options: str
    disk: str
    transport: str
    rotational: int
    scheduler: str
    model: str
    size: str


@dataclass(frozen=True)
class PackStats:
    entries: int
    payload_bytes: int
    max_entry_bytes: int
    n_expert: int
    upgate_bytes: int
    down_bytes: int


@dataclass(frozen=True)
class ModelStats:
    name: str
    arch: str
    n_layer: int
    n_expert: int | None
    n_expert_used: int | None
    non_expert_fixed_bytes: int
    non_expert_layer_bytes: list[int]
    tensor_count: int


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_helper(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stat_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def sanitize_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = value.strip(".-_")
    return value[:80] or "model"


def model_shards(first_shard: Path) -> list[Path]:
    match = re.match(r"(.+)-\d{5}-of-(\d{5})\.gguf$", first_shard.name)
    if not match:
        return [first_shard]
    prefix, total = match.groups()
    shards = sorted(first_shard.parent.glob(f"{prefix}-*-of-{total}.gguf"))
    return [p.resolve() for p in shards] if shards else [first_shard]


def field_value(field: Any) -> Any:
    if field is None or not getattr(field, "types", None):
        return None
    type_name = field.types[0].name
    if type_name == "ARRAY":
        return None
    if type_name == "STRING":
        return bytes(field.parts[-1]).decode("utf-8", errors="replace")
    return field.parts[-1].tolist()[0]


def layer_from_tensor(name: str) -> int | None:
    match = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def is_expert_tensor(name: str) -> bool:
    return any(
        marker in name
        for marker in (
            ".ffn_up_exps.",
            ".ffn_gate_exps.",
            ".ffn_down_exps.",
            ".ffn_gate_up_exps.",
        )
    )


def read_model_stats(models: list[Path]) -> ModelStats:
    sys.path.insert(0, str(REPO / "gguf-py"))
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as exc:
        raise SystemExit("cannot import gguf-py; run from the ik_llama checkout or install gguf") from exc

    reader = GGUFReader(models[0], "r")
    arch = field_value(reader.get_field("general.architecture")) or ""
    name = field_value(reader.get_field("general.name")) or models[0].stem

    def meta_int(suffix: str) -> int | None:
        if not arch:
            return None
        value = field_value(reader.get_field(f"{arch}.{suffix}"))
        return int(value) if value is not None else None

    n_layer = meta_int("block_count")
    n_expert = meta_int("expert_count")
    n_expert_used = meta_int("expert_used_count")

    by_layer: dict[int, int] = {}
    fixed_bytes = 0
    max_layer = -1
    tensor_count = 0
    for path in models:
        shard_reader = reader if path == models[0] else GGUFReader(path, "r")
        tensor_count += len(shard_reader.tensors)
        for tensor in shard_reader.tensors:
            layer = layer_from_tensor(tensor.name)
            if layer is not None:
                max_layer = max(max_layer, layer)
            if is_expert_tensor(tensor.name):
                continue
            if layer is None:
                fixed_bytes += int(tensor.n_bytes)
            else:
                by_layer[layer] = by_layer.get(layer, 0) + int(tensor.n_bytes)

    if n_layer is None:
        n_layer = max_layer + 1 if max_layer >= 0 else 0
    layer_bytes = [by_layer.get(i, 0) for i in range(n_layer)]
    return ModelStats(
        name=str(name),
        arch=str(arch),
        n_layer=int(n_layer),
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        non_expert_fixed_bytes=fixed_bytes,
        non_expert_layer_bytes=layer_bytes,
        tensor_count=tensor_count,
    )


def read_pack_stats(path: Path) -> PackStats:
    with path.open("rb") as f:
        header = f.read(PACK_HEADER.size)
        if len(header) != PACK_HEADER.size:
            raise SystemExit(f"expert pack is truncated: {path}")
        magic, version, _header_size, n_entries, _data_start = PACK_HEADER.unpack(header)
        if magic != PACK_MAGIC or version != 1:
            raise SystemExit(f"not a v1 MoE expert pack: {path}")

        payload = 0
        max_entry = 0
        max_expert = -1
        upgate = 0
        down = 0
        for _ in range(n_entries):
            raw = f.read(PACK_ENTRY.size)
            if len(raw) != PACK_ENTRY.size:
                raise SystemExit(f"expert pack index is truncated: {path}")
            name_raw, expert_idx, _reserved, _offset, nbytes = PACK_ENTRY.unpack(raw)
            name = name_raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            payload += int(nbytes)
            max_entry = max(max_entry, int(nbytes))
            max_expert = max(max_expert, int(expert_idx))
            if ".ffn_down_exps." in name:
                down += int(nbytes)
            elif is_expert_tensor(name) or name.endswith(":gate") or name.endswith(":up"):
                upgate += int(nbytes)
        return PackStats(
            entries=int(n_entries),
            payload_bytes=payload,
            max_entry_bytes=max_entry,
            n_expert=max_expert + 1,
            upgate_bytes=upgate,
            down_bytes=down,
        )


def read_gpu_info() -> GpuInfo | None:
    if shutil.which("nvidia-smi") is None:
        return None
    query = "name,memory.total,memory.free,pci.bus_id,driver_version"
    out = run_text(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 5:
        return None
    try:
        return GpuInfo(
            name=parts[0],
            total_mib=int(float(parts[1])),
            free_mib=int(float(parts[2])),
            bus_id=parts[3],
            driver=parts[4],
        )
    except ValueError:
        return None


def read_mem_info() -> MemInfo:
    if Path("/proc/meminfo").is_file():
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[0].rstrip(":")] = int(parts[1]) // 1024
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", total)
        return MemInfo(total_mib=total, available_mib=available)

    total = 0
    if hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // MIB
        except (OSError, ValueError):
            total = 0
    return MemInfo(total_mib=int(total), available_mib=int(total))


def read_cpu_info() -> CpuInfo:
    values: dict[str, str] = {}
    out = run_text(["lscpu"])
    if out:
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()

    def int_value(key: str) -> int:
        match = re.search(r"\d+", values.get(key, ""))
        return int(match.group(0)) if match else 0

    return CpuInfo(
        model_name=values.get("Model name", platform.processor()),
        cpu_count=int_value("CPU(s)") or (os.cpu_count() or 0),
        threads_per_core=int_value("Thread(s) per core"),
        cores_per_socket=int_value("Core(s) per socket"),
        sockets=int_value("Socket(s)"),
        numa_nodes=int_value("NUMA node(s)"),
    )


def read_storage_info(path: Path) -> StorageInfo:
    source = ""
    fstype = ""
    options = ""
    out = run_text(["findmnt", "-T", str(path), "-no", "SOURCE,FSTYPE,OPTIONS"])
    if out:
        parts = out.split(maxsplit=2)
        source = parts[0] if len(parts) >= 1 else ""
        fstype = parts[1] if len(parts) >= 2 else ""
        options = parts[2] if len(parts) >= 3 else ""

    disk = source
    if source.startswith("/dev/"):
        parent = run_text(["lsblk", "-ndo", "PKNAME", source])
        disk_name = parent.splitlines()[0].strip() if parent else Path(source).name
        disk = f"/dev/{disk_name}" if disk_name else source

    transport = ""
    rotational = 0
    scheduler = ""
    model = ""
    size = ""
    if disk:
        lsblk = run_text(["lsblk", "-ndo", "TRAN,ROTA,SCHED,MODEL,SIZE", disk])
        if lsblk:
            parts = lsblk.split()
            transport = parts[0] if len(parts) >= 1 else ""
            rotational = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
            scheduler = parts[2] if len(parts) >= 3 else ""
            if len(parts) >= 4:
                size = parts[-1]
                model = " ".join(parts[3:-1])

    return StorageInfo(
        source=source,
        fstype=fstype,
        options=options,
        disk=disk,
        transport=transport,
        rotational=rotational,
        scheduler=scheduler,
        model=model,
        size=size,
    )


def llama_cache_dir() -> Path:
    if os.environ.get("LLAMA_CACHE"):
        return Path(os.environ["LLAMA_CACHE"]).expanduser()
    if sys.platform.startswith("linux"):
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "llama.cpp"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "llama.cpp"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


def existing_file(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_file() else None


def first_existing(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
        files = [p.resolve() for p in matches if p.is_file()]
        if files:
            return files[0]
    return None


def discover_model(explicit: str | None) -> Path:
    if explicit:
        model = existing_file(explicit)
        if not model:
            raise SystemExit(f"missing model: {explicit}")
        return model
    model = existing_file(os.environ.get("MODEL"))
    if model:
        return model

    candidates = [
        "/home/wici/models/glm-5.1/UD-IQ3_XXS/*-00001-of-*.gguf",
        "/home/wici/models/glm-5.1/**/*.gguf",
        "/home/wici/models/**/*.gguf",
    ]
    found = first_existing(candidates)
    if found:
        return found
    raise SystemExit("missing model: set MODEL=/path/model.gguf or pass --model")


def discover_pack(model: Path, explicit: str | None) -> Path:
    if explicit:
        pack = existing_file(explicit)
        if not pack:
            raise SystemExit(f"missing expert pack: {explicit}")
        return pack
    pack = existing_file(os.environ.get("GGML_MOE_EXPERT_PACK"))
    if pack:
        return pack

    search_roots = [model.parent, model.parent.parent, Path("/home/wici/models/glm-5.1"), Path("/home/wici/models")]
    for root in search_roots:
        if not root.is_dir():
            continue
        files = sorted(root.glob("*.expert-pack"))
        if files:
            return files[0].resolve()
    raise SystemExit("missing expert pack: set GGML_MOE_EXPERT_PACK=/path/pack or pass --expert-pack")


def discover_profile(explicit: str | None) -> Path | None:
    if explicit:
        profile = existing_file(explicit)
        if not profile:
            raise SystemExit(f"missing route profile: {explicit}")
        return profile
    profile = existing_file(os.environ.get("GGML_MOE_VRAM_PROFILE"))
    if profile:
        return profile

    preferred = REPO / "bench" / "wici-glm51-moe" / "codex-route32-t8.route.csv"
    if preferred.is_file():
        return preferred.resolve()

    bench_dir = REPO / "bench" / "wici-glm51-moe"
    if bench_dir.is_dir():
        routes = sorted(bench_dir.glob("*.route.csv"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        if routes:
            return routes[0].resolve()
    return None


def is_nvme_path(path: Path) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    source = run_text(["findmnt", "-T", str(path), "-no", "SOURCE"])
    if not source:
        return False
    device = source.splitlines()[0].strip()
    if "nvme" in device:
        return True
    out = run_text(["lsblk", "-ndo", "NAME,TRAN,ROTA", device])
    if not out and device.startswith("/dev/"):
        out = run_text(["lsblk", "-ndo", "NAME,TRAN,ROTA", device.removeprefix("/dev/")])
    if not out:
        return False
    text = out.lower()
    return "nvme" in text and not text.rstrip().endswith(" 1")


def floor_to(value: int, quantum: int) -> int:
    return max(0, (value // quantum) * quantum)


def repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def compute_n_gpu_layers(model: ModelStats, gpu: GpuInfo | None, ctx_size: int, cache_mib_floor: int) -> tuple[int, int]:
    if gpu is None or model.n_layer <= 0:
        return 0, 0

    fixed_mib = math.ceil(model.non_expert_fixed_bytes / MIB)
    layer_mib = [math.ceil(v / MIB) for v in model.non_expert_layer_bytes]
    total_mib = gpu.total_mib
    ctx_factor = max(1.0, ctx_size / 2048.0)
    runtime_reserve_mib = int(max(1024, total_mib * 0.06) * ctx_factor)
    available = max(0, total_mib - runtime_reserve_mib - cache_mib_floor)

    used = fixed_mib
    n_layers = 0
    for bytes_mib in layer_mib:
        if bytes_mib <= 0:
            n_layers += 1
            continue
        if used + bytes_mib > available:
            break
        used += bytes_mib
        n_layers += 1
    return min(n_layers, model.n_layer), used


def compute_vram_cache_mib(gpu: GpuInfo | None, gpu_model_mib: int, pack: PackStats, ctx_size: int) -> int:
    if gpu is None or pack.max_entry_bytes <= 0:
        return 0
    live_reserve_mib = max(1024, int(gpu.total_mib * 0.05))
    live_headroom = gpu.free_mib - live_reserve_mib
    useful_cap = max(512, int(pack.payload_bytes / MIB * 0.04))
    vram_fraction_cap = max(512, int(gpu.total_mib * 0.13))
    budget = min(live_headroom, useful_cap, vram_fraction_cap)
    slot_mib = max(1, math.ceil(pack.max_entry_bytes / MIB))
    return floor_to(max(0, int(budget)), max(128, slot_mib))


def read_env_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            values[key] = value
    return values


def measured_upgate_pct(profile: Path, cache_mib: int) -> int | None:
    bench_dir = profile.parent
    if not bench_dir.is_dir():
        return None
    route_sim = load_helper("moe_route_cache_sim_for_runner", REPO / "scripts" / "moe-route-cache-sim.py")
    best: tuple[float, int, Path] | None = None
    for stderr_path in bench_dir.glob("*.stderr.txt"):
        env = read_env_log(route_sim.measured_env_path(stderr_path))
        if env.get("GGML_MOE_VRAM_CACHE_MIB") != str(cache_mib):
            continue
        if env.get("GGML_MOE_VRAM_PROFILE_PROTECT") != "1":
            continue
        env_profile = env.get("GGML_MOE_VRAM_PROFILE")
        if env_profile and Path(env_profile).name != profile.name:
            continue
        try:
            pct = route_sim.parse_measured_pct(stderr_path)
            report = route_sim.parse_runtime_stderr(stderr_path)
        except Exception:
            continue
        if pct is None or report.timing is None:
            continue
        candidate = (float(report.timing.eval_ms), int(pct), stderr_path)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best is not None else None


def compute_upgate_pct(pack: PackStats, profile: Path | None, cache_mib: int) -> int:
    if profile is not None and cache_mib > 0:
        try:
            measured = measured_upgate_pct(profile, cache_mib)
            if measured is not None:
                return measured
            route_sim = load_helper("moe_route_cache_sim_for_runner", REPO / "scripts" / "moe-route-cache-sim.py")
            entries = route_sim.load_entries(profile)
            estimates = [
                route_sim.split_estimate(entries, cache_mib, pct, 20, True, 1.0, 1.0, "protected")
                for pct in range(35, 76, 5)
            ]
            best = min(estimates, key=lambda est: (float(est["weighted_miss_gib"]), float(est["miss_gib"]), int(est["pct"])))
            return int(best["pct"])
        except Exception as exc:
            print(f"[moe-run] warning: route profile split estimate failed: {exc}", file=sys.stderr)
    total = pack.upgate_bytes + pack.down_bytes
    if total <= 0:
        return 50
    return max(35, min(75, round(100 * pack.upgate_bytes / total / 5) * 5))


def compute_ram_tier_mib(mem: MemInfo, pack: PackStats) -> int:
    if mem.available_mib <= 0 or pack.max_entry_bytes <= 0:
        return 0
    safety = max(2048, int(mem.total_mib * 0.10))
    available = max(0, mem.available_mib - safety)
    useful_cap = max(0, int(pack.payload_bytes / MIB * 0.02))
    budget = min(int(available * 0.36), useful_cap)
    return floor_to(budget, 1024)


def compute_env(
    model_path: Path,
    pack_path: Path,
    profile_path: Path | None,
    model: ModelStats,
    pack: PackStats,
    gpu: GpuInfo | None,
    mem: MemInfo,
    ctx_size: int,
) -> dict[str, str]:
    cache_mib = compute_vram_cache_mib(gpu, 0, pack, ctx_size)
    n_gpu_layers, _gpu_model_mib = compute_n_gpu_layers(model, gpu, ctx_size, cache_mib_floor=cache_mib)

    gpu_enabled = gpu is not None and n_gpu_layers > 0
    direct_io = is_nvme_path(pack_path)
    ram_tier_mib = compute_ram_tier_mib(mem, pack)
    avg_slot_mib = max(1.0, pack.payload_bytes / max(pack.entries, 1) / MIB)
    ram_skip = round(cache_mib / avg_slot_mib) if avg_slot_mib > 0 else 0

    env = {
        "MODEL": str(model_path),
        "CTX_SIZE": str(ctx_size),
        "N_GPU_LAYERS": str(n_gpu_layers),
        "GGML_MOE_BATCH_PROFILE": "0",
        "GGML_MOE_GPU_HANDOFF": "1" if gpu_enabled else "0",
        "GGML_MOE_VRAM_CACHE_MIB": str(cache_mib),
        "GGML_MOE_VRAM_CACHE_UPGATE_PCT": str(compute_upgate_pct(pack, profile_path, cache_mib)),
        "GGML_MOE_STAGE_PINNED": "1" if gpu_enabled and direct_io else "0",
        "GGML_MOE_STREAM_UP_GATE_PARALLEL": "1" if gpu_enabled else "0",
        "GGML_MOE_STREAM_UP_GATE_PARALLEL_STAGE": "1" if gpu_enabled and direct_io else "0",
        "GGML_MOE_STREAM_UP_GATE_STAGE_SPLIT": "1" if gpu_enabled and direct_io else "0",
        "GGML_MOE_DOWN_PARALLEL_STAGE": "1" if gpu_enabled and direct_io else "0",
        "GGML_MOE_EXPERT_PACK": str(pack_path),
        "GGML_MOE_IO_BACKEND": "direct" if direct_io else "mmap",
        "GGML_MOE_RAM_TIER_MIB": str(ram_tier_mib),
        "GGML_MOE_RAM_TIER_SKIP": str(ram_skip),
    }
    if profile_path is not None:
        env["GGML_MOE_VRAM_PROFILE"] = repo_relative_or_absolute(profile_path)
        env["GGML_MOE_VRAM_PROFILE_PROTECT"] = "1"
    else:
        env["GGML_MOE_VRAM_PROFILE_PROTECT"] = "0"
    return env


def build_fingerprint(
    model_paths: list[Path],
    pack_path: Path,
    profile_path: Path | None,
    model: ModelStats,
    pack: PackStats,
    gpu: GpuInfo | None,
    cpu: CpuInfo,
    mem: MemInfo,
    storage: StorageInfo,
    ctx_size: int,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ctx_size": ctx_size,
        "model_files": [stat_fingerprint(path) for path in model_paths],
        "pack_file": stat_fingerprint(pack_path),
        "profile_file": stat_fingerprint(profile_path) if profile_path else None,
        "model": model.__dict__,
        "pack": pack.__dict__,
        "gpu": gpu.__dict__ if gpu else None,
        "cpu": cpu.__dict__,
        "mem": mem.__dict__,
        "storage": storage.__dict__,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), payload


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_preset(json_path: Path, env_path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(json_path, json.dumps(data, indent=2, sort_keys=True) + "\n")
    env_lines = [
        "# Generated by scripts/moe-run.py; includes base defaults plus computed preset env.",
        f"# preset_json={json_path}",
    ]
    for key, value in effective_env(data).items():
        env_lines.append(f"export {key}={shlex.quote(str(value))}")
    write_text_atomic(env_path, "\n".join(env_lines) + "\n")


def effective_env(data: dict[str, Any]) -> dict[str, str]:
    env = {key: str(value) for key, value in BASE_ENV.items()}
    env.update({key: str(value) for key, value in data["env"].items()})
    return env


CHEAP_GROUNDTRUTH_MATCH_KEYS = {
    "platform",
    "host",
    "model_path",
    "model_basename",
    "expert_pack_path",
    "expert_pack_basename",
    "profile_path",
    "profile_basename",
    "ctx_size",
    "gpu_total_mib_min",
    "gpu_total_mib_max",
    "gpu_name",
    "gpu_total_mib",
    "gpu_bus_id",
    "gpu_driver",
    "cpu_model",
    "cpu_count",
    "threads_per_core",
    "cores_per_socket",
    "sockets",
    "numa_nodes",
    "mem_total_mib",
    "storage_source",
    "storage_fstype",
    "storage_options",
    "storage_disk",
    "storage_transport",
    "storage_rotational",
    "storage_scheduler",
    "storage_model",
    "storage_size",
}


def match_one(actual: str, expected: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, str):
        return actual == expected
    if isinstance(expected, Iterable):
        return actual in expected
    return False


def groundtruth_common_matches(
    match: dict[str, Any],
    model_path: Path,
    pack_path: Path,
    profile_path: Path | None,
    gpu: GpuInfo | None,
    cpu: CpuInfo,
    mem: MemInfo,
    storage: StorageInfo,
    ctx_size: int,
) -> bool:
    if "platform" in match and platform.platform() != match["platform"]:
        return False
    if not match_one(socket.gethostname(), match.get("host")):
        return False
    if not match_one(str(model_path), match.get("model_path")):
        return False
    if not match_one(model_path.name, match.get("model_basename")):
        return False
    if not match_one(str(pack_path), match.get("expert_pack_path")):
        return False
    if not match_one(pack_path.name, match.get("expert_pack_basename")):
        return False
    if "profile_path" in match and str(profile_path) != str(match["profile_path"]):
        return False
    if "profile_basename" in match and (profile_path is None or profile_path.name != match["profile_basename"]):
        return False
    if "ctx_size" in match and int(match["ctx_size"]) != ctx_size:
        return False
    if "gpu_total_mib_min" in match and (gpu is None or gpu.total_mib < int(match["gpu_total_mib_min"])):
        return False
    if "gpu_total_mib_max" in match and (gpu is None or gpu.total_mib > int(match["gpu_total_mib_max"])):
        return False
    if "gpu_name" in match and (gpu is None or gpu.name != match["gpu_name"]):
        return False
    if "gpu_total_mib" in match and (gpu is None or gpu.total_mib != int(match["gpu_total_mib"])):
        return False
    if "gpu_bus_id" in match and (gpu is None or gpu.bus_id != match["gpu_bus_id"]):
        return False
    if "gpu_driver" in match and (gpu is None or gpu.driver != match["gpu_driver"]):
        return False
    if "cpu_model" in match and cpu.model_name != match["cpu_model"]:
        return False
    if "cpu_count" in match and cpu.cpu_count != int(match["cpu_count"]):
        return False
    if "threads_per_core" in match and cpu.threads_per_core != int(match["threads_per_core"]):
        return False
    if "cores_per_socket" in match and cpu.cores_per_socket != int(match["cores_per_socket"]):
        return False
    if "sockets" in match and cpu.sockets != int(match["sockets"]):
        return False
    if "numa_nodes" in match and cpu.numa_nodes != int(match["numa_nodes"]):
        return False
    if "mem_total_mib" in match and mem.total_mib != int(match["mem_total_mib"]):
        return False
    if "storage_source" in match and storage.source != match["storage_source"]:
        return False
    if "storage_fstype" in match and storage.fstype != match["storage_fstype"]:
        return False
    if "storage_options" in match and storage.options != match["storage_options"]:
        return False
    if "storage_disk" in match and storage.disk != match["storage_disk"]:
        return False
    if "storage_transport" in match and storage.transport != match["storage_transport"]:
        return False
    if "storage_rotational" in match and storage.rotational != int(match["storage_rotational"]):
        return False
    if "storage_scheduler" in match and storage.scheduler != match["storage_scheduler"]:
        return False
    if "storage_model" in match and storage.model != match["storage_model"]:
        return False
    if "storage_size" in match and storage.size != match["storage_size"]:
        return False
    return True


def groundtruth_matches(
    data: dict[str, Any],
    model_path: Path,
    pack_path: Path,
    profile_path: Path | None,
    model: ModelStats,
    gpu: GpuInfo | None,
    cpu: CpuInfo,
    mem: MemInfo,
    storage: StorageInfo,
    ctx_size: int,
) -> bool:
    match = data.get("match", {})
    if not groundtruth_common_matches(match, model_path, pack_path, profile_path, gpu, cpu, mem, storage, ctx_size):
        return False
    if "arch" in match and model.arch != match["arch"]:
        return False
    if "n_layer" in match and model.n_layer != int(match["n_layer"]):
        return False
    if "n_expert" in match and model.n_expert != int(match["n_expert"]):
        return False
    if "n_expert_used" in match and model.n_expert_used != int(match["n_expert_used"]):
        return False
    return True


def groundtruth_matches_cheap(
    data: dict[str, Any],
    model_path: Path,
    pack_path: Path,
    profile_path: Path | None,
    gpu: GpuInfo | None,
    cpu: CpuInfo,
    mem: MemInfo,
    storage: StorageInfo,
    ctx_size: int,
) -> bool:
    match = data.get("match", {})
    if any(key not in CHEAP_GROUNDTRUTH_MATCH_KEYS for key in match):
        return False
    return groundtruth_common_matches(match, model_path, pack_path, profile_path, gpu, cpu, mem, storage, ctx_size)


def load_groundtruth_preset_fast(
    model_path: Path,
    pack_path: Path,
    profile_path: Path | None,
    gpu: GpuInfo | None,
    cpu: CpuInfo,
    mem: MemInfo,
    storage: StorageInfo,
    ctx_size: int,
) -> tuple[dict[str, Any], Path, Path] | None:
    groundtruth_dir = REPO_PRESETS / "groundtruth"
    if not groundtruth_dir.is_dir():
        return None
    for path in sorted(groundtruth_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if groundtruth_matches_cheap(data, model_path, pack_path, profile_path, gpu, cpu, mem, storage, ctx_size):
            env_path = path.with_suffix(".env")
            return data, path, env_path
    return None


def load_groundtruth_preset(
    model_path: Path,
    pack_path: Path,
    profile_path: Path | None,
    model: ModelStats,
    gpu: GpuInfo | None,
    cpu: CpuInfo,
    mem: MemInfo,
    storage: StorageInfo,
    ctx_size: int,
) -> tuple[dict[str, Any], Path, Path] | None:
    groundtruth_dir = REPO_PRESETS / "groundtruth"
    if not groundtruth_dir.is_dir():
        return None
    for path in sorted(groundtruth_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if groundtruth_matches(data, model_path, pack_path, profile_path, model, gpu, cpu, mem, storage, ctx_size):
            env_path = path.with_suffix(".env")
            return data, path, env_path
    return None


def read_matching_preset(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != fingerprint:
        print(f"[moe-run] warning: ignoring preset with mismatched fingerprint: {path}", file=sys.stderr)
        return None
    return data


def load_or_create_preset(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    model_path = discover_model(args.model)
    model_paths = model_shards(model_path)
    model_path = model_paths[0]
    pack_path = discover_pack(model_path, args.expert_pack)
    profile_path = discover_profile(args.profile)

    ctx_size = args.ctx_size or int(os.environ.get("CTX_SIZE", "2048"))
    gpu = read_gpu_info()
    cpu = read_cpu_info()
    mem = read_mem_info()
    storage = read_storage_info(pack_path)

    if not args.force:
        groundtruth = load_groundtruth_preset_fast(model_path, pack_path, profile_path, gpu, cpu, mem, storage, ctx_size)
        if groundtruth is not None:
            return groundtruth

    model = read_model_stats(model_paths)
    pack = read_pack_stats(pack_path)

    if not args.force:
        groundtruth = load_groundtruth_preset(model_path, pack_path, profile_path, model, gpu, cpu, mem, storage, ctx_size)
        if groundtruth is not None:
            return groundtruth

    fingerprint, inputs = build_fingerprint(model_paths, pack_path, profile_path, model, pack, gpu, cpu, mem, storage, ctx_size)

    slug = sanitize_slug(model.name or model_path.stem)
    preset_name = f"{fingerprint[:16]}"
    repo_json_path = REPO_PRESETS / slug / f"{preset_name}.json"
    repo_env_path = REPO_PRESETS / slug / f"{preset_name}.env"
    cache_dir = llama_cache_dir() / "moe-presets" / slug
    json_path = cache_dir / f"{preset_name}.json"
    env_path = cache_dir / f"{preset_name}.env"

    if not args.force:
        repo_preset = read_matching_preset(repo_json_path, fingerprint)
        if repo_preset is not None:
            return repo_preset, repo_json_path, repo_env_path
        cache_preset = read_matching_preset(json_path, fingerprint)
        if cache_preset is not None:
            return cache_preset, json_path, env_path

    env = compute_env(model_path, pack_path, profile_path, model, pack, gpu, mem, ctx_size)
    data = {
        "fingerprint": fingerprint,
        "created_by": "scripts/moe-run.py",
        "inputs": inputs,
        "env": env,
    }
    write_preset(json_path, env_path, data)
    if args.save_repo_preset or os.environ.get("MOE_RUN_SAVE_REPO_PRESET") == "1":
        write_preset(repo_json_path, repo_env_path, data)
        return data, repo_json_path, repo_env_path
    return data, json_path, env_path


def normalize_extra(extra: list[str]) -> list[str]:
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def has_thread_arg(extra: list[str]) -> bool:
    return any(arg == "-t" or arg == "--threads" or arg.startswith("--threads=") for arg in extra)


def has_threads_batch_arg(extra: list[str]) -> bool:
    return any(arg == "-tb" or arg == "--threads-batch" or arg.startswith("--threads-batch=") for arg in extra)


def has_any_arg(extra: list[str], names: set[str]) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names if name.startswith("--")) for arg in extra)


def default_chat_args(extra: list[str]) -> list[str]:
    defaults: list[str] = []
    for key, value in CHAT_DEFAULT_ARGS:
        names = {key}
        if key == "-b":
            names.add("--batch-size")
        if key == "-n":
            names.add("--predict")
        if not has_any_arg(extra, names):
            defaults.extend([key, value])
    return defaults


def chat_load_args(args: argparse.Namespace, extra: list[str]) -> list[str]:
    if args.chat_load_mode != "fast-prompt":
        return []
    defaults: list[str] = []
    if not has_any_arg(extra, {"--defer-experts"}):
        defaults.append("--defer-experts")
    if not has_any_arg(extra, {"--no-warmup"}):
        defaults.append("--no-warmup")
    return defaults


def thread_count(args: argparse.Namespace, data: dict[str, Any]) -> int:
    inputs = data.get("inputs") or {}
    model = inputs.get("model") or {}
    cpu = inputs.get("cpu") or {}
    n_expert_used = model.get("n_expert_used")
    cpu_count = int(cpu.get("cpu_count") or 0)
    if args.threads:
        return args.threads
    if n_expert_used:
        threads = int(n_expert_used)
        return max(1, min(threads, cpu_count or threads))
    return default_threads_from_inputs()


def chat_gpu_layers(args: argparse.Namespace, data: dict[str, Any]) -> int:
    if args.chat_gpu_layers is not None:
        return args.chat_gpu_layers
    env_value = os.environ.get("CHAT_N_GPU_LAYERS")
    if env_value:
        return int(env_value)
    env = effective_env(data)
    return max(int(env["N_GPU_LAYERS"]), 70)


def build_command(args: argparse.Namespace, data: dict[str, Any], extra: list[str]) -> list[str]:
    if args.chat:
        env = effective_env(data)
        binary = os.environ.get("BIN", str(REPO / "build-cuda" / "bin" / "llama-cli"))
        cmd = [
            binary,
            "-m", str(env["MODEL"]),
            "-c", str(env["CTX_SIZE"]),
            "-ngl", str(chat_gpu_layers(args, data)),
            "-fa", "on",
            "-mla", str(env["MLA"]),
            "-cmoe",
            "--conversation",
        ]
        cmd.extend(chat_load_args(args, extra))
        cmd.extend(default_chat_args(extra))
        cmd.append("--color")
        if not has_thread_arg(extra):
            cmd.extend(["-t", str(thread_count(args, data))])
        if not has_threads_batch_arg(extra):
            cmd.extend(["-tb", str(args.chat_threads_batch)])
        cmd.extend(extra)
        return cmd

    bench = REPO / "scripts" / "bench-wici-glm51-moe.sh"
    cmd = ["bash", str(bench)]
    if not has_thread_arg(extra):
        cmd.extend(["-t", str(thread_count(args, data))])
    cmd.extend(extra)
    return cmd


def default_threads_from_inputs() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(cpus, max(4, cpus // 2)))


def chat_stderr_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return llama_cache_dir() / "moe-chat" / f"{timestamp}-{os.getpid()}.stderr.log"


def run_quiet_chat(cmd: list[str], env: dict[str, str], preset_path: Path) -> int:
    stderr_path = chat_stderr_log_path()
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[moe-run] preset={preset_path}", file=sys.stderr)
    print(f"[moe-run] stderr_log={stderr_path}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os.chdir(REPO)
    with stderr_path.open("wb") as stderr_file:
        return subprocess.call(cmd, env=env, stderr=stderr_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute or reuse a host/model MoE preset, then run the GLM MoE harness."
    )
    parser.add_argument("--model", help="GGUF model shard. Defaults to MODEL or known local model paths.")
    parser.add_argument("--expert-pack", help="MoE expert pack. Defaults to GGML_MOE_EXPERT_PACK or nearby *.expert-pack.")
    parser.add_argument("--profile", help="Route profile CSV. Defaults to GGML_MOE_VRAM_PROFILE or latest bench route profile.")
    parser.add_argument("--ctx-size", type=int, help="Context size used for memory budgeting. Defaults to CTX_SIZE or 2048.")
    parser.add_argument("--tokens", type=int, help="Tokens for the benchmark harness. Defaults to TOKENS or the harness default.")
    parser.add_argument("--threads", type=int, help="CPU threads passed to llama-cli as -t unless already supplied after --.")
    parser.add_argument("--chat", action="store_true", help="Run llama-cli in interactive chat mode instead of the benchmark harness.")
    parser.add_argument("--chat-load-mode", choices=("fast-prompt", "eager"), default="fast-prompt", help="Chat load policy. fast-prompt defers expert residency and skips warmup; eager keeps llama-cli's preload/warmup behavior.")
    parser.add_argument("--chat-verbose", action="store_true", help="Keep llama-cli chat startup logs visible instead of redirecting stderr to LLAMA_CACHE/moe-chat.")
    parser.add_argument("--chat-gpu-layers", type=int, help="GPU layers for chat mode. Defaults to CHAT_N_GPU_LAYERS or the measured wici fast-prompt value.")
    parser.add_argument("--chat-threads-batch", type=int, default=int(os.environ.get("CHAT_THREADS_BATCH", "24")), help="Batch/prompt processing threads for chat mode unless -tb/--threads-batch is passed after --.")
    parser.add_argument("--chat-active-prewarm", action="store_true", help="Run an optional prewarm decode before the first chat prompt.")
    parser.add_argument("--chat-no-active-prewarm", action="store_true", help="Deprecated compatibility flag; active prewarm is off unless --chat-active-prewarm is set.")
    parser.add_argument("--force", action="store_true", help="Recompute the preset even if the fingerprint already exists.")
    parser.add_argument("--save-repo-preset", action="store_true", help="Also write the generated matching preset under presets/moe for reuse.")
    parser.add_argument("--dry-run", action="store_true", help="Print the preset and command without executing.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments after -- are passed to llama-cli.")
    args = parser.parse_args()

    if args.ctx_size is not None and args.ctx_size <= 0:
        raise SystemExit("--ctx-size must be positive")
    if args.tokens is not None and args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    if args.threads is not None and args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if args.chat_gpu_layers is not None and args.chat_gpu_layers < 0:
        raise SystemExit("--chat-gpu-layers must be non-negative")
    if args.chat_threads_batch <= 0:
        raise SystemExit("--chat-threads-batch must be positive")
    if args.chat_verbose and not args.chat:
        raise SystemExit("--chat-verbose requires --chat")
    if args.chat_active_prewarm and not args.chat:
        raise SystemExit("--chat-active-prewarm requires --chat")
    if args.chat_no_active_prewarm and not args.chat:
        raise SystemExit("--chat-no-active-prewarm requires --chat")
    if args.chat_active_prewarm and args.chat_no_active_prewarm:
        raise SystemExit("--chat-active-prewarm conflicts with --chat-no-active-prewarm")
    if args.chat and not args.dry_run and not sys.stdin.isatty():
        raise SystemExit(
            "--chat needs an interactive terminal. Run it from a shell on wici, "
            "or use: ssh -t wici 'cd /home/wici/venti/ik_llama && python3 scripts/moe-run.py --chat'"
        )

    data, json_path, env_path = load_or_create_preset(args)
    extra = normalize_extra(args.extra)
    cmd = build_command(args, data, extra)

    env = os.environ.copy()
    env.update(effective_env(data))
    if args.tokens is not None:
        env["TOKENS"] = str(args.tokens)
    if args.chat and args.chat_active_prewarm and args.chat_load_mode == "fast-prompt":
        env["LLAMA_CHAT_ACTIVE_PREWARM"] = "1"

    if args.dry_run:
        print(f"preset={json_path}")
        print(f"env_file={env_path}")
        print(f"cwd={REPO}")
        if args.chat and not args.chat_verbose:
            print(f"chat_stderr_log={chat_stderr_log_path()}")
        print("command=" + " ".join(shlex.quote(part) for part in cmd))
        return 0

    if args.chat and not args.chat_verbose:
        return run_quiet_chat(cmd, env, json_path)

    print(f"[moe-run] preset={json_path}", file=sys.stderr)
    os.chdir(REPO)
    os.execvpe(cmd[0], cmd, env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

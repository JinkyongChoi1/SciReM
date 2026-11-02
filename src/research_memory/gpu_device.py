"""One locally chosen shared GPU inside a reviewed one-GPU SGE allocation."""
import csv
import io
import os
from pathlib import Path
import re
import stat
import subprocess


def _command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=20,
                            stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError("GPU metadata command failed: " + Path(arguments[0]).name)
    return result.stdout


def _resources(text):
    for line in text.splitlines():
        if line.startswith("hard resource_list:"):
            return dict(item.strip().split("=", 1) for item in line.split(":", 1)[1].split(",") if "=" in item)
    return {}


def _hardware(text):
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 6:
            raise RuntimeError("Malformed GPU hardware metadata")
        index, uuid, pci, total, free, utilization = (cell.strip() for cell in row)
        try:
            item = {"index": int(index), "uuid": uuid, "pci_bus_id": pci,
                    "total_memory_mib": int(total), "free_memory_mib": int(free),
                    "utilization_percent": int(utilization)}
        except ValueError as error:
            raise RuntimeError("GPU hardware memory/utilization metadata unavailable") from error
        if (item["index"] < 0 or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid) or
            not 0 <= item["free_memory_mib"] <= item["total_memory_mib"] or
            not 0 <= item["utilization_percent"] <= 100):
            raise RuntimeError("Invalid GPU hardware metadata")
        devices.append(item)
    if not devices or len({d["uuid"] for d in devices}) != len(devices) or len({d["index"] for d in devices}) != len(devices):
        raise RuntimeError("GPU hardware enumeration is empty or ambiguous")
    return devices


def _select(devices, selector, minimum):
    if selector is not None:
        selector = selector.strip()
        if not selector or "," in selector:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must identify exactly one GPU")
        matching = [d for d in devices if d["uuid"] == selector or selector.isdigit() and d["index"] == int(selector)]
        if len(matching) != 1:
            raise RuntimeError("Provided CUDA_VISIBLE_DEVICES is not an unambiguous enumerated GPU")
        selected = matching[0]
        reason = "provided single CUDA_VISIBLE_DEVICES selector respected; not proof of exclusive assignment"
    else:
        selected = min(devices, key=lambda d: (-d["free_memory_mib"], d["utilization_percent"], d["index"]))
        reason = "local shared-device choice, not exclusive assignment"
    if selected["free_memory_mib"] < minimum:
        raise RuntimeError("Selected GPU lacks the reviewed free-VRAM headroom")
    return selected, reason


def _driver_device(uuid):
    matches = []
    for path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        fields = {}
        for line in path.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        if fields.get("GPU UUID") != uuid:
            continue
        try:
            minor = int(fields["Device Minor"])
        except (KeyError, ValueError) as error:
            raise RuntimeError("Driver device minor mapping unavailable") from error
        device = Path("/dev/nvidia" + str(minor))
        metadata = device.stat()
        if minor < 0 or not stat.S_ISCHR(metadata.st_mode) or os.minor(metadata.st_rdev) != minor:
            raise RuntimeError("Driver UUID mapping does not match a GPU character device")
        matches.append({"minor_number": minor, "device_major": os.major(metadata.st_rdev),
                        "device_path": str(device), "driver_information_source": str(path)})
    if len(matches) != 1:
        raise RuntimeError("Driver UUID-to-device mapping is missing or ambiguous")
    return matches[0]


def _mount_paths(device_path):
    paths = [device_path, "/dev/nvidiactl"]
    paths += [name for name in ("/dev/nvidia-uvm", "/dev/nvidia-uvm-tools") if Path(name).exists()]
    if any(not stat.S_ISCHR(Path(name).stat().st_mode) for name in paths):
        raise RuntimeError("GPU namespace mounts require verified character devices")
    return paths


def choose_gpu(min_free_mib=38912):
    """No locks or exclusivity claim: another user can allocate VRAM after this query."""
    job = os.environ.get("JOB_ID", "")
    if not job.isdigit() or not isinstance(min_free_mib, int) or min_free_mib <= 0:
        raise RuntimeError("Reviewed finite SGE allocation and positive VRAM headroom required")
    resources = _resources(_command(["qstat", "-j", job]))
    if resources.get("gpu") not in ("1", "1.0"):
        raise RuntimeError("Exactly one GPU must be reserved for this SGE job")
    devices = _hardware(_command(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.free,utilization.gpu",
                                  "--format=csv,noheader,nounits"]))
    selected, reason = _select(devices, os.environ.get("CUDA_VISIBLE_DEVICES"), min_free_mib)
    mapped = _driver_device(selected["uuid"])
    return {"job_id": job, "requested_gpu_count": 1, "selected_uuid": selected["uuid"],
        "selected_index": selected["index"], "pci_bus_id": selected["pci_bus_id"],
        "free_memory_mib": selected["free_memory_mib"], "total_memory_mib": selected["total_memory_mib"],
        "utilization_percent": selected["utilization_percent"], **mapped,
        "mount_paths": _mount_paths(mapped["device_path"]), "cuda_visible_devices": selected["uuid"],
        "minimum_free_memory_mib": min_free_mib, "selection_reason": reason,
        "scheduler_assigned_device": False, "exclusive": False, "race_possible": True,
        "intended_worker_visible_gpu_count": 1, "worker_visibility_verification": "pending private worker",
        "sharing_caveat": "Other users may share this GPU; free-VRAM observations are not reservations."}

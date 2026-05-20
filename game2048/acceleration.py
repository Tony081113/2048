from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class AccelerationBackend:
    name: str
    available: bool
    active: bool
    detail: str

    @property
    def summary(self) -> str:
        status = "啟用" if self.active else "未啟用"
        return f"{self.name} ({status}) - {self.detail}"


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_module(module_name: str) -> Optional[object]:
    if not _has_module(module_name):
        return None
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _configure_nvidia_dll_paths() -> None:
    base_spec = importlib.util.find_spec("nvidia")
    if base_spec is None or not base_spec.submodule_search_locations:
        return

    base_dir = Path(next(iter(base_spec.submodule_search_locations)))
    candidate_dirs = [
        base_dir / "cuda_nvrtc" / "bin",
        base_dir / "cuda_runtime" / "bin",
    ]

    existing_paths = os.environ.get("PATH", "")
    path_entries = existing_paths.split(os.pathsep) if existing_paths else []
    for dll_dir in candidate_dirs:
        if not dll_dir.is_dir():
            continue
        dll_path = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_path)
        if dll_path not in path_entries:
            path_entries.insert(0, dll_path)

    os.environ["PATH"] = os.pathsep.join(path_entries)


def probe_cupy() -> Tuple[Optional[object], int, str]:
    _configure_nvidia_dll_paths()
    cupy = _load_module("cupy")
    if cupy is None:
        return None, 0, ""
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count <= 0:
            return None, 0, ""
        device_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
        return cupy, device_count, device_name
    except Exception:
        return None, 0, ""


def load_cupy(prefer_gpu: bool = True) -> Optional[object]:
    if not prefer_gpu:
        return None

    cupy, _device_count, _device_name = probe_cupy()
    if cupy is None:
        return None
    try:
        cupy.zeros((1,), dtype=cupy.float32)
        return cupy
    except Exception:
        return None


def detect_acceleration_backend(prefer_gpu: bool = True) -> AccelerationBackend:
    """
    偵測是否存在可用的加速後端。

    目前專案的 Expectimax 搜尋仍是以 Python 控制流程為主，
    GPU 主要用在盤面啟發式評估上。
    """
    cupy, device_count, device_name = probe_cupy()
    if cupy is not None and prefer_gpu:
        active_cupy = load_cupy(prefer_gpu=True)
        if active_cupy is not None:
            return AccelerationBackend(
                name="CuPy",
                available=True,
                active=True,
                detail=f"使用 {device_name} 進行 GPU 盤面評估，共 {device_count} 個 CUDA 裝置",
            )

        return AccelerationBackend(
            name="CPU",
            available=True,
            active=True,
            detail=f"偵測到 {device_name}，但 GPU 初始化失敗，已退回 CPU",
        )

    if cupy is not None and not prefer_gpu:
        return AccelerationBackend(
            name="CPU",
            available=True,
            active=True,
            detail=f"GPU 可用 ({device_name})，但目前已停用",
        )

    torch = _load_module("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return AccelerationBackend(
                    name="CUDA",
                    available=True,
                    active=False,
                    detail=f"偵測到 {gpu_name}；目前未選用 PyTorch GPU 後端",
                )
        except Exception:
            pass

        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        if mps is not None:
            try:
                if mps.is_available():
                    return AccelerationBackend(
                        name="MPS",
                        available=True,
                        active=False,
                        detail="偵測到 Apple GPU 後端；目前演算法仍使用 CPU 搜尋",
                    )
            except Exception:
                pass

    return AccelerationBackend(
        name="CPU",
        available=True,
        active=True,
        detail="目前專案未啟用 GPU 後端",
    )
import importlib.util
import os
import subprocess
import sys


def ensure_matplotlib_installed() -> None:
    """Install matplotlib if missing, then restart current Python process."""
    if importlib.util.find_spec("matplotlib") is not None:
        return

    print("[setup] matplotlib not found. Installing...")
    cmd = [sys.executable, "-m", "pip", "install", "matplotlib"]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install matplotlib automatically. "
            "Please run: python -m pip install matplotlib"
        )

    print("[setup] matplotlib installed. Restarting...")
    os.execv(sys.executable, [sys.executable, *sys.argv])

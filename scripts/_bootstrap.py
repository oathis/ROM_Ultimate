from pathlib import Path
import os
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _candidate_base_pythons():
    candidates = []

    env_override = os.environ.get("ROM_BASE_PYTHON")
    if env_override:
        candidates.append(Path(env_override))

    env_file = Path.home() / ".conda" / "environments.txt"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            env_root = line.strip()
            if not env_root:
                continue
            candidates.append(Path(env_root) / "python.exe")

    candidates.extend(
        [
            Path("C:/Anaconda3/python.exe"),
            Path.home() / "anaconda3" / "python.exe",
            Path("C:/ProgramData/anaconda3/python.exe"),
        ]
    )
    return candidates


def _resolve_base_python():
    for path in _candidate_base_pythons():
        if path.exists():
            return path.resolve()
    return None


def ensure_base_python():
    # Allow bypass when explicitly needed.
    if os.environ.get("ROM_USE_BASE", "1") != "1":
        return
    if os.environ.get("ROM_SKIP_BASE_REEXEC") == "1":
        return
    if not sys.argv or not str(sys.argv[0]).lower().endswith(".py"):
        return

    target = _resolve_base_python()
    if target is None:
        return

    try:
        current = Path(sys.executable).resolve()
    except Exception:
        current = Path(sys.executable)

    same = False
    try:
        same = current.samefile(target)
    except Exception:
        same = str(current).lower() == str(target).lower()

    if same:
        return

    env = os.environ.copy()
    env["ROM_SKIP_BASE_REEXEC"] = "1"
    cmd = [str(target), *sys.argv]
    raise SystemExit(subprocess.call(cmd, env=env))


ensure_base_python()

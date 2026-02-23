from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    slug = slug.strip("-_")
    return slug.lower() or "session"


def sessions_root(root_dir: Path) -> Path:
    return Path(root_dir) / "artifacts" / "sessions"


@dataclass
class SessionPaths:
    session_id: str
    root_dir: Path
    session_dir: Path
    workspace_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    predictions_dir: Path
    runs_dir: Path
    logs_dir: Path
    metadata_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "root_dir": str(self.root_dir),
            "session_dir": str(self.session_dir),
            "workspace_dir": str(self.workspace_dir),
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "models_dir": str(self.models_dir),
            "predictions_dir": str(self.predictions_dir),
            "runs_dir": str(self.runs_dir),
            "logs_dir": str(self.logs_dir),
            "metadata_path": str(self.metadata_path),
        }


def _default_raw_dir(root_dir: Path) -> Path:
    return Path(root_dir) / "data" / "raw" / "Dataset"


def ensure_session(root_dir: Path, session_id: str, raw_dir: Path | None = None) -> SessionPaths:
    root_dir = Path(root_dir)
    session_id = _slugify(session_id)
    base_dir = sessions_root(root_dir)
    session_dir = base_dir / session_id
    workspace_dir = session_dir / "workspace"
    processed_dir = workspace_dir / "data" / "processed"
    models_dir = workspace_dir / "models"
    predictions_dir = workspace_dir / "predictions"
    runs_dir = session_dir / "runs"
    logs_dir = session_dir / "logs"
    metadata_path = session_dir / "session.json"

    for path in (processed_dir, models_dir, predictions_dir, runs_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    previous: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    resolved_raw = Path(raw_dir) if raw_dir is not None else Path(previous.get("raw_dir", _default_raw_dir(root_dir)))
    created_at = previous.get("created_at", _now_iso())
    updated_at = _now_iso()

    metadata = {
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "raw_dir": str(resolved_raw),
        "workspace": {
            "processed_dir": str(processed_dir),
            "models_dir": str(models_dir),
            "predictions_dir": str(predictions_dir),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return SessionPaths(
        session_id=session_id,
        root_dir=root_dir,
        session_dir=session_dir,
        workspace_dir=workspace_dir,
        raw_dir=resolved_raw,
        processed_dir=processed_dir,
        models_dir=models_dir,
        predictions_dir=predictions_dir,
        runs_dir=runs_dir,
        logs_dir=logs_dir,
        metadata_path=metadata_path,
    )


def create_session(root_dir: Path, session_name: str | None = None, raw_dir: Path | None = None) -> SessionPaths:
    root_dir = Path(root_dir)
    label = _slugify(session_name or "session")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = f"{label}-{timestamp}"

    candidate = session_id
    suffix = 1
    while (sessions_root(root_dir) / candidate).exists():
        suffix += 1
        candidate = f"{session_id}-{suffix}"

    return ensure_session(root_dir=root_dir, session_id=candidate, raw_dir=raw_dir)


def list_sessions(root_dir: Path) -> list[dict[str, Any]]:
    root_dir = Path(root_dir)
    base_dir = sessions_root(root_dir)
    if not base_dir.exists():
        return []

    items: list[dict[str, Any]] = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        metadata_path = entry / "session.json"
        payload: dict[str, Any] = {"session_id": entry.name, "session_dir": str(entry)}
        if metadata_path.exists():
            try:
                payload.update(json.loads(metadata_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                payload["metadata_error"] = "invalid_json"
        items.append(payload)

    items.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return items


def _json_safe(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(value) for value in data]
    return data


def record_run(
    session: SessionPaths,
    stage: str,
    status: str,
    request: dict[str, Any],
    summary: dict[str, Any] | None = None,
    log_text: str = "",
    error: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    stage_slug = _slugify(stage)
    json_path = session.runs_dir / f"{run_id}_{stage_slug}.json"
    log_path = session.logs_dir / f"{run_id}_{stage_slug}.log"

    payload = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "started_at": started_at or _now_iso(),
        "finished_at": finished_at or _now_iso(),
        "request": _json_safe(request),
        "summary": _json_safe(summary or {}),
        "error": error,
        "log_path": str(log_path),
        "record_path": str(json_path),
    }

    log_path.write_text(log_text or "", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def recent_runs(session: SessionPaths, limit: int = 20) -> list[dict[str, Any]]:
    if not session.runs_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(session.runs_dir.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
        if len(records) >= limit:
            break
    return records


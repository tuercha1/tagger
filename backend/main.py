"""打标后端。"""
from __future__ import annotations

import json
import sqlite3
import string
import threading
import urllib.error
import urllib.request
import uuid
import zlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from contextlib import asynccontextmanager

from tagger import Tagger

# 配置
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
CONFIG_PATH = BASE_DIR / "config.json"
TRANSLATIONS_DB = BASE_DIR / "translations.db"
TRANSLATIONS_BACKUP_DIR = BASE_DIR / "backups"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIG_LOCK = threading.Lock()
BATCH_TRANSLATE_SIZE = 100
DB_LOOKUP_BATCH_SIZE = 500


def _load_config() -> dict:
    """读配置。"""
    with CONFIG_LOCK:
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cfg = {}
        else:
            cfg = {}
    cfg.setdefault("models", [])
    cfg.setdefault("active", "")
    return cfg


def _save_config(cfg: dict) -> None:
    data = json.dumps(cfg, ensure_ascii=False, indent=2)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_LOCK:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=CONFIG_PATH.parent,
            delete=False,
            prefix=f".{CONFIG_PATH.name}.",
            suffix=".tmp",
        ) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        tmp_path.replace(CONFIG_PATH)


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRANSLATIONS_DB), timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _init_translation_db() -> None:
    """初始化词库。"""
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                en TEXT PRIMARY KEY,
                zh TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(translations)").fetchall()
        }
        if "source" not in columns:
            conn.execute("ALTER TABLE translations ADD COLUMN source TEXT NOT NULL DEFAULT 'ai'")
        if "hit_count" not in columns:
            conn.execute("ALTER TABLE translations ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0")
        if "created_at" not in columns:
            conn.execute("ALTER TABLE translations ADD COLUMN created_at TEXT")
        conn.execute(
            "UPDATE translations SET created_at = COALESCE(created_at, updated_at, datetime('now'))"
        )
        conn.execute(
            "UPDATE translations SET updated_at = COALESCE(updated_at, datetime('now'))"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_translations_updated_at "
            "ON translations(updated_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_translations_source "
            "ON translations(source)"
        )


_init_translation_db()


# 模型
_tagger: Tagger | None = None
_lock = threading.Lock()
_loading = False
_load_error: str | None = None
_active_id: str = ""
_active_name: str = ""
_active_dir: str = ""
_load_generation = 0


def _load_model(model_id: str, name: str, model_dir: str) -> None:
    global _tagger, _loading, _load_error, _active_id, _active_name, _active_dir, _load_generation
    with _lock:
        _load_generation += 1
        generation = _load_generation
        _loading = True
        _tagger = None
        _load_error = None
        _active_id = model_id
        _active_name = name
        _active_dir = model_dir
    try:
        t = Tagger(model_dir, use_cuda=False)
        with _lock:
            if generation == _load_generation:
                _tagger = t
                _loading = False
    except Exception as e:  # noqa: BLE001
        with _lock:
            if generation == _load_generation:
                _load_error = str(e)
                _loading = False


def get_tagger() -> Tagger:
    with _lock:
        tagger = _tagger
        loading = _loading
        load_error = _load_error
    if tagger is not None:
        return tagger
    if loading:
        raise HTTPException(status_code=503, detail="模型加载中，请稍候…")
    if load_error:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {load_error}")
    raise HTTPException(status_code=503, detail="模型未加载")


def _detect_model_dirs(root: Path) -> list[str]:
    """搜模型目录。"""
    found: list[str] = []
    if (root / "model.onnx").exists() and (root / "selected_tags.csv").exists():
        found.append(str(root))
    try:
        for p in root.rglob("model.onnx"):
            d = p.parent
            if (d / "selected_tags.csv").exists():
                s = str(d)
                if s not in found:
                    found.append(s)
    except (PermissionError, OSError):
        pass
    return found


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动加载模型
    cfg = _load_config()
    active_id = cfg.get("active", "")
    for m in cfg.get("models", []):
        if m.get("id") == active_id:
            threading.Thread(
                target=_load_model,
                args=(m["id"], m.get("name", ""), m.get("dir", "")),
                daemon=True,
            ).start()
            break
    yield


app = FastAPI(title="打标工具", lifespan=lifespan)


# 工具
def _list_images(folder: Path, recursive: bool = False) -> list[Path]:
    globber = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        [p for p in globber if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def _parse_tag_text(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _read_tags_file(txt: Path) -> list[str]:
    if not txt.exists():
        return []
    try:
        return _parse_tag_text(txt.read_text(encoding="utf-8").strip())
    except OSError:
        return []


def _dedupe(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for t in tags:
        k = t.strip()
        if k and k not in seen:
            seen.add(k)
            result.append(k)
    return result


# 模型接口
def _models_public(cfg: dict) -> list[dict]:
    return [
        {"id": m.get("id", ""), "name": m.get("name", ""), "dir": m.get("dir", "")}
        for m in cfg.get("models", [])
        if m.get("id") and m.get("dir")
    ]


@app.get("/api/model/status")
def model_status():
    cfg = _load_config()
    with _lock:
        loaded = _tagger is not None
        loading = _loading
        load_error = _load_error
        active_id = _active_id
        active_name = _active_name
        active_dir = _active_dir
        providers = _tagger.providers if _tagger is not None else []
    return {
        "loaded": loaded,
        "loading": loading,
        "error": load_error,
        "active_id": active_id,
        "active_name": active_name,
        "active_dir": active_dir,
        "models": _models_public(cfg),
        "providers": providers,
    }


@app.get("/api/model/detect")
def model_detect(root: str = Query(...)):
    """搜模型目录。"""
    r = Path(root)
    if not r.is_dir():
        raise HTTPException(status_code=404, detail="路径不存在")
    dirs = _detect_model_dirs(r)
    return {"dirs": dirs, "count": len(dirs)}


class AddModelReq(BaseModel):
    dir: str
    name: str = ""


@app.post("/api/model/add")
def model_add(req: AddModelReq):
    """添加模型。"""
    md = Path(req.dir)
    if not (md / "model.onnx").exists() or not (md / "selected_tags.csv").exists():
        raise HTTPException(
            status_code=400,
            detail="该文件夹下未找到 model.onnx 或 selected_tags.csv，请确认是正确的模型目录",
        )
    cfg = _load_config()
    # 复用同路径
    for m in cfg["models"]:
        if Path(m.get("dir", "")) == md:
            mid = m["id"]
            name = m.get("name", md.name)
            break
    else:
        mid = uuid.uuid4().hex[:8]
        name = req.name.strip() or md.name
        cfg["models"].append({"id": mid, "name": name, "dir": str(md)})
    cfg["active"] = mid
    _save_config(cfg)
    threading.Thread(target=_load_model, args=(mid, name, str(md)), daemon=True).start()
    return {"ok": True, "model": {"id": mid, "name": name, "dir": str(md)}, "loading": True}


class SwitchModelReq(BaseModel):
    id: str


@app.post("/api/model/switch")
def model_switch(req: SwitchModelReq):
    """切换模型。"""
    cfg = _load_config()
    target = next((m for m in cfg["models"] if m.get("id") == req.id), None)
    if not target:
        raise HTTPException(status_code=404, detail="模型不存在")
    cfg["active"] = target["id"]
    _save_config(cfg)
    threading.Thread(
        target=_load_model, args=(target["id"], target["name"], target["dir"]), daemon=True
    ).start()
    return {"ok": True, "loading": True}


@app.delete("/api/model/{model_id}")
def model_delete(model_id: str):
    """删除模型。"""
    cfg = _load_config()
    cfg["models"] = [m for m in cfg["models"] if m["id"] != model_id]
    if cfg["active"] == model_id:
        cfg["active"] = ""
    _save_config(cfg)
    global _tagger, _loading, _load_error, _active_id, _active_name, _active_dir, _load_generation
    if _active_id == model_id:
        with _lock:
            _load_generation += 1
            _tagger = None
            _loading = False
            _load_error = None
            _active_id = ""
            _active_name = ""
            _active_dir = ""
    return {"ok": True, "models": _models_public(cfg)}


# 文件夹接口
@app.get("/api/browse")
def browse(path: str = Query("")):
    """浏览文件夹。"""
    if not path:
        drives = [f"{c}:\\" for c in string.ascii_uppercase if Path(f"{c}:\\").exists()]
        return {"path": "", "parent": "", "dirs": [{"name": d, "path": d} for d in drives]}
    p = Path(path)
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="路径不存在")
    dirs = []
    try:
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if child.is_dir():
                dirs.append({"name": child.name, "path": str(child)})
    except PermissionError:
        pass
    parent = str(p.parent) if str(p.parent) != str(p) else ""
    return {"path": str(p), "parent": parent, "dirs": dirs}


# 工作目录接口
@app.get("/api/workspace")
def get_workspace():
    """取工作目录。"""
    cfg = _load_config()
    return {"path": cfg.get("default_workspace", "")}


class WorkspaceReq(BaseModel):
    path: str


@app.post("/api/workspace")
def set_workspace(req: WorkspaceReq):
    """设工作目录。"""
    cfg = _load_config()
    cfg["default_workspace"] = req.path
    _save_config(cfg)
    return {"ok": True}


@app.get("/api/workspace/folders")
def workspace_folders(path: str = Query(None)):
    """列子文件夹。"""
    if path:
        d = Path(path)
        if not d.is_dir():
            return {"folders": [], "workspace": ""}
    else:
        cfg = _load_config()
        ws = cfg.get("default_workspace", "")
        if not ws:
            return {"folders": [], "workspace": ""}
        d = Path(ws)
        if not d.is_dir():
            return {"folders": [], "workspace": ws}
    folders = []
    # 子文件夹统计
    try:
        for child in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            if child.is_dir():
                try:
                    count = sum(
                        1 for f in child.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
                    )
                except (PermissionError, OSError):
                    count = 0
                if count > 0:
                    folders.append({"name": child.name, "path": str(child), "count": count})
    except PermissionError:
        pass
    return {"folders": folders, "workspace": str(d)}


# 扫描接口
@app.get("/api/scan")
def scan(folder: str = Query(...), recursive: bool = Query(False)):
    d = Path(folder)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="文件夹不存在")
    images = _list_images(d, recursive)
    items = []
    for p in images:
        txt = p.with_suffix(".txt")
        tag_count = len(_read_tags_file(txt))
        items.append({
            "name": p.name,
            "path": str(p),
            "has_tags": txt.exists(),
            "tag_count": tag_count,
        })
    return {"folder": str(d), "count": len(items), "images": items}


# 图片接口
@app.get("/api/thumb")
def thumb(path: str = Query(...), size: int = Query(320)):
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    size = min(max(size, 32), 2048)
    try:
        from PIL import Image as PILImage
        with PILImage.open(p) as src:
            img = src.convert("RGB")
        img.thumbnail((size, size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image")
def serve_image(path: str = Query(...)):
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不是支持的图片文件")
    return FileResponse(str(p))


# 标签接口
@app.get("/api/tags")
def get_tags(path: str = Query(...)):
    p = Path(path)
    txt = p.with_suffix(".txt")
    if not txt.exists():
        return {"tags": [], "raw": ""}
    try:
        raw = txt.read_text(encoding="utf-8").strip()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"读取标签失败: {e}")
    tags = _parse_tag_text(raw)
    return {"tags": tags, "raw": raw}


class SaveTagsReq(BaseModel):
    path: str
    tags: list[str]


@app.post("/api/tags")
def save_tags(req: SaveTagsReq):
    p = Path(req.path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    txt = p.with_suffix(".txt")
    cleaned = _dedupe(req.tags)
    if not cleaned:
        # 空标签删文件
        if txt.exists():
            txt.unlink()
        return {"ok": True, "tags": [], "cleared": True}
    txt.write_text(", ".join(cleaned), encoding="utf-8")
    return {"ok": True, "tags": cleaned, "cleared": False}


# 标签统计
class TagStatsReq(BaseModel):
    paths: list[str]


@app.post("/api/tags/stats")
def tag_stats(req: TagStatsReq):
    """统计标签。"""
    from collections import Counter

    counter: Counter = Counter()
    image_tags: dict[str, list[str]] = {}
    for path_str in req.paths:
        p = Path(path_str)
        txt = p.with_suffix(".txt")
        tags = _read_tags_file(txt)
        image_tags[path_str] = tags
        counter.update(tags)
    result = [{"name": name, "count": count} for name, count in counter.most_common()]
    return {"tags": result, "total_images": len(req.paths), "image_tags": image_tags}


# 单张推理
@app.get("/api/tag/single")
def tag_single(path: str = Query(...), top_n: int = Query(60)):
    tagger = get_tagger()
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    top_n = min(max(top_n, 1), 500)
    candidates = tagger.tag_candidates(p, top_n=top_n)
    return {"candidates": candidates}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# 选图打标
class TagSelectedReq(BaseModel):
    paths: list[str]
    trigger: str = ""
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    include_character: bool = False
    normalize: bool = True
    overwrite: bool = False
    prefix: str = ""


@app.post("/api/tag/selected")
def tag_selected(req: TagSelectedReq):
    """选图打标。"""
    tagger = get_tagger()
    prefix_tags = [x.strip() for x in req.prefix.split(",") if x.strip()] if req.prefix else []
    trigger = req.trigger.strip()

    def event_stream() -> Iterator[str]:
        total = len(req.paths)
        done = 0
        for i, path_str in enumerate(req.paths):
            p = Path(path_str)
            try:
                if not p.is_file():
                    yield _sse({"type": "error", "index": i + 1, "total": total,
                                "name": p.name, "error": "文件不存在"})
                    continue
                txt = p.with_suffix(".txt")
                if txt.exists() and not req.overwrite:
                    existing = _read_tags_file(txt)
                    done += 1
                    yield _sse({"type": "progress", "index": i + 1, "total": total,
                                "path": path_str, "name": p.name,
                                "tags": existing, "skipped": True})
                    continue
                model_tags = tagger.tag_image(
                    p,
                    general_threshold=req.general_threshold,
                    character_threshold=req.character_threshold,
                    include_character=req.include_character,
                    normalize=req.normalize,
                )
                # 标签置信度
                tag_probs: dict[str, float] = {
                    t["name"]: round(t["prob"], 4) for t in model_tags
                }
                output: list[str] = []
                if trigger:
                    output.append(trigger)
                output.extend(prefix_tags)
                output.extend([t["name"] for t in model_tags])
                final = _dedupe(output)
                done += 1
                yield _sse({"type": "progress", "index": i + 1, "total": total,
                            "path": path_str, "name": p.name,
                            "tags": final, "tag_probs": tag_probs, "skipped": False})
            except Exception as e:  # noqa: BLE001
                yield _sse({"type": "error", "index": i + 1, "total": total,
                            "name": p.name, "error": str(e)})
        yield _sse({"type": "done", "total": total, "done": done})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# 翻译
PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "openai-response": {
        "endpoint": "https://api.openai.com/v1/responses",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
    },
    "anthropic": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-20250514",
    },
    "azure": {
        "endpoint": "https://你的资源名.openai.azure.com/openai/deployments/你的部署名/chat/completions?api-version=2024-02-15-preview",
        "model": "gpt-4o-mini",
    },
    "new-api": {
        "endpoint": "https://你的API地址/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "cherryin": {
        "endpoint": "https://api.cherryin.ai/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "ollama": {
        "endpoint": "http://localhost:11434/api/chat",
        "model": "llama3.1",
    },
}

# 无密钥提供商
NO_KEY_PROVIDERS = {"ollama"}

DEFAULT_TRANSLATE_PROMPT = (
    "你是一个翻译助手。请将以下Stable Diffusion标签从英文翻译为简体中文。"
    "每行一个标签，输出格式为\"英文|中文\"。只输出翻译结果，不要解释："
)


def _load_translate_config() -> dict:
    cfg = _load_config()
    t = cfg.get("translate", {})
    provider = t.get("provider", "openai")
    defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
    return {
        "provider": provider,
        "endpoint": t.get("endpoint", defaults["endpoint"]),
        "api_key": t.get("api_key", ""),
        "model": t.get("model", defaults["model"]),
        "prompt": t.get("prompt", DEFAULT_TRANSLATE_PROMPT),
    }


def _save_translate_config(data: dict) -> None:
    cfg = _load_config()
    cfg["translate"] = {
        "provider": data.get("provider", "openai"),
        "endpoint": data.get("endpoint", ""),
        "api_key": data.get("api_key", ""),
        "model": data.get("model", ""),
        "prompt": data.get("prompt", ""),
    }
    _save_config(cfg)


# 提供商调用

def _call_openai(endpoint: str, api_key: str, model: str, prompt_text: str) -> str:
    """OpenAI Chat。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def _call_openai_response(endpoint: str, api_key: str, model: str, prompt_text: str) -> str:
    """Responses。"""
    body = json.dumps({
        "model": model,
        "input": prompt_text,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    # Responses 输出
    for item in result.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text" and c.get("text"):
                return c["text"]
    # output_text 兜底
    if result.get("output_text"):
        return result["output_text"]
    raise ValueError("无法解析 Responses API 返回内容")


def _call_gemini(endpoint: str, api_key: str, model: str, prompt_text: str) -> str:
    """Gemini。"""
    url = f"{endpoint}/models/{model}:generateContent"
    url += f"?key={api_key}" if "?" not in url else f"&key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.3},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(endpoint: str, api_key: str, model: str, prompt_text: str) -> str:
    """Anthropic。"""
    body = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"]


def _call_azure(endpoint: str, api_key: str, model: str, prompt_text: str) -> str:
    """Azure。"""
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
        "api-key": api_key,
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def _call_ollama(endpoint: str, api_key: str, model: str, prompt_text: str) -> str:
    """Ollama。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {"temperature": 0.3},
    }).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def _call_llm(provider: str, endpoint: str, api_key: str,
              model: str, prompt_text: str) -> str:
    """分发 LLM。"""
    if provider == "gemini":
        return _call_gemini(endpoint, api_key, model, prompt_text)
    elif provider == "anthropic":
        return _call_anthropic(endpoint, api_key, model, prompt_text)
    elif provider == "openai-response":
        return _call_openai_response(endpoint, api_key, model, prompt_text)
    elif provider == "azure":
        return _call_azure(endpoint, api_key, model, prompt_text)
    elif provider == "ollama":
        return _call_ollama(endpoint, api_key, model, prompt_text)
    else:
        # OpenAI 兼容格式
        return _call_openai(endpoint, api_key, model, prompt_text)


# 翻译取消
_translate_cancel = threading.Event()


class TranslationCancelled(Exception):
    """翻译取消。"""


def _check_cancel():
    """检查取消。"""
    if _translate_cancel.is_set():
        raise TranslationCancelled


# 翻译进度
_translate_progress = []
_translate_progress_lock = threading.Lock()


def _push_progress(en: str, zh: str, source: str = "translated"):
    """推送进度。"""
    with _translate_progress_lock:
        _translate_progress.append({"en": en, "zh": zh, "source": source})


def _call_llm_stream(provider: str, endpoint: str, api_key: str,
                     model: str, prompt_text: str):
    """LLM 流。"""
    STREAMABLE = {"openai", "new-api", "cherryin", "azure"}
    if provider not in STREAMABLE:
        text = _call_llm(provider, endpoint, api_key, model, prompt_text)
        for en, zh in _parse_translations(text).items():
            _check_cancel()
            yield en, zh
        return

    if provider == "azure":
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.3,
            "stream": True,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json", "api-key": api_key}
    else:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.3,
            "stream": True,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    req = urllib.request.Request(endpoint, data=body, headers=headers)
    buffer = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                _check_cancel()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
                if not content:
                    continue
                buffer += content
                # 解析完整行
                while "\n" in buffer:
                    line_text, buffer = buffer.split("\n", 1)
                    line_text = line_text.strip()
                    if "|" in line_text:
                        parts = line_text.split("|", 1)
                        if len(parts) == 2:
                            en, zh = parts[0].strip(), parts[1].strip()
                            if en and zh:
                                yield en, zh
        # 处理尾行
        buffer = buffer.strip()
        if buffer and "|" in buffer:
            parts = buffer.split("|", 1)
            if len(parts) == 2:
                en, zh = parts[0].strip(), parts[1].strip()
                if en and zh:
                    yield en, zh
    except TranslationCancelled:
        # 已取消
        return
    except Exception:
        # 回退非流式
        text = _call_llm(provider, endpoint, api_key, model, prompt_text)
        for en, zh in _parse_translations(text).items():
            _check_cancel()
            yield en, zh


def _parse_translations(text: str) -> dict[str, str]:
    """解析翻译。"""
    translations: dict[str, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if "|" in line:
            parts = line.split("|", 1)
            if len(parts) == 2:
                en, zh = parts[0].strip(), parts[1].strip()
                if en and zh:
                    translations[en] = zh
    return translations


def _clean_tag_list(tags: list[str]) -> list[str]:
    """清理标签。"""
    seen: set[str] = set()
    cleaned: list[str] = []
    for tag in tags:
        en = str(tag).strip()
        if en and en not in seen:
            seen.add(en)
            cleaned.append(en)
    return cleaned


def _chunks(values: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _lookup_translations(
    conn: sqlite3.Connection,
    tags: list[str],
    *,
    increment_hits: bool = False,
) -> dict[str, str]:
    """查词库。"""
    cleaned = _clean_tag_list(tags)
    found: dict[str, str] = {}
    for batch in _chunks(cleaned, DB_LOOKUP_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT en, zh FROM translations WHERE en IN ({placeholders})",
            batch,
        ).fetchall()
        found.update({en: zh for en, zh in rows})
        if increment_hits and rows:
            hit_keys = [en for en, _ in rows]
            hit_placeholders = ",".join("?" for _ in hit_keys)
            conn.execute(
                "UPDATE translations "
                "SET hit_count = hit_count + 1, updated_at = updated_at "
                f"WHERE en IN ({hit_placeholders})",
                hit_keys,
            )
    return found


def _http_error_detail(prefix: str, e: urllib.error.HTTPError) -> str:
    detail = f"{prefix} (HTTP {e.code})"
    try:
        err_body = e.read().decode("utf-8", errors="replace")
        detail += f": {err_body[:300]}"
    except Exception:
        pass
    return detail


def _store_translation(
    conn: sqlite3.Connection,
    en: str,
    zh: str,
    *,
    source: str = "ai",
    overwrite: bool = True,
) -> bool:
    en = str(en).strip()
    zh = str(zh).strip()
    if not en or not zh:
        return False
    if overwrite:
        conn.execute(
            """
            INSERT INTO translations (en, zh, source, hit_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, datetime('now'), datetime('now'))
            ON CONFLICT(en) DO UPDATE SET
                zh = excluded.zh,
                source = excluded.source,
                updated_at = datetime('now')
            """,
            (en, zh, source),
        )
        return True
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO translations (en, zh, source, hit_count, created_at, updated_at)
        VALUES (?, ?, ?, 0, datetime('now'), datetime('now'))
        """,
        (en, zh, source),
    )
    return cursor.rowcount > 0


def _translate_and_store_batches(
    tags: list[str],
    cfg: dict,
    *,
    raise_first_error: bool,
) -> dict[str, str]:
    translations: dict[str, str] = {}
    with _connect_db() as conn:
        for i in range(0, len(tags), BATCH_TRANSLATE_SIZE):
            if _translate_cancel.is_set():
                break
            batch = tags[i : i + BATCH_TRANSLATE_SIZE]
            batch_set = set(batch)
            prompt_text = cfg["prompt"] + "\n" + "\n".join(batch)
            try:
                for en, zh in _call_llm_stream(
                    cfg["provider"],
                    cfg["endpoint"],
                    cfg["api_key"],
                    cfg["model"],
                    prompt_text,
                ):
                    if en not in batch_set:
                        continue
                    translations[en] = zh
                    _store_translation(conn, en, zh, source="ai")
                    _push_progress(en, zh, "translated")
                conn.commit()
            except TranslationCancelled:
                break
            except urllib.error.HTTPError as e:
                if i == 0 and raise_first_error:
                    raise HTTPException(status_code=502, detail=_http_error_detail("API 调用失败", e))
            except Exception as e:
                if i == 0 and raise_first_error:
                    raise HTTPException(status_code=502, detail=f"翻译失败: {e}")
    return translations


# 模型列表

def _fetch_models(provider: str, endpoint: str, api_key: str) -> list[str]:
    """取模型列表。"""
    if provider == "ollama":
        # Ollama 模型
        tags_url = endpoint.replace("/api/chat", "/api/tags")
        req = urllib.request.Request(tags_url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        models = []
        for m in result.get("models", []):
            name = m.get("name", "")
            if name:
                models.append(name)
        return models

    elif provider == "gemini":
        # Gemini 模型
        url = endpoint.rstrip("/") + "/models?key=" + api_key
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        models = []
        for m in result.get("models", []):
            name = m.get("name", "")  # e.g. "models/gemini-2.0-flash"
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                models.append(name)
        return models

    elif provider == "anthropic":
        # Anthropic 模型
        models_url = endpoint.replace("/messages", "/models")
        req = urllib.request.Request(models_url, headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        return [m.get("id", "") for m in result.get("data", []) if m.get("id")]

    elif provider == "azure":
        # Azure 模型
        # Azure endpoint
        # deployments
        import re
        m = re.match(r"(https://[^/]+)/openai/deployments/", endpoint)
        if not m:
            raise ValueError("Azure endpoint 格式无法解析，请检查 API 地址")
        base = m.group(1)
        api_ver = ""
        if "api-version=" in endpoint:
            api_ver = endpoint.split("api-version=")[-1].split("&")[0]
        list_url = f"{base}/openai/deployments?api-version={api_ver}"
        req = urllib.request.Request(list_url, headers={"api-key": api_key})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        return [d.get("id", "") for d in result.get("data", []) if d.get("id")]

    else:
        # OpenAI 模型
        # 推断 models URL
        # chat -> models
        # responses -> models
        models_url = endpoint
        for suffix in ["/chat/completions", "/responses"]:
            if models_url.endswith(suffix):
                models_url = models_url[: -len(suffix)] + "/models"
                break
        else:
            # 替换末段
            parts = models_url.rsplit("/", 1)
            if len(parts) == 2:
                models_url = parts[0] + "/models"
        req = urllib.request.Request(models_url, headers={
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        return [m.get("id", "") for m in result.get("data", []) if m.get("id")]


class TranslateReq(BaseModel):
    tags: list[str]
    force: bool = False


@app.get("/api/translate/config")
def get_translate_config():
    return _load_translate_config()


@app.post("/api/translate/config")
def save_translate_config(req: dict):
    _save_translate_config(req)
    return {"ok": True}


@app.get("/api/translate/models")
def list_translate_models():
    """检测模型。"""
    cfg = _load_translate_config()
    provider = cfg["provider"]
    endpoint = cfg["endpoint"]
    api_key = cfg["api_key"]
    if provider not in NO_KEY_PROVIDERS and not api_key:
        raise HTTPException(status_code=400, detail="未配置 API 密钥，请先填写并保存")
    if not endpoint:
        raise HTTPException(status_code=400, detail="未配置 API 地址")
    try:
        models = _fetch_models(provider, endpoint, api_key)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=_http_error_detail("检测失败", e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"检测失败: {e}")
    return {"models": models}


@app.post("/api/translate/cancel")
def cancel_translate():
    """取消翻译。"""
    _translate_cancel.set()
    return {"ok": True}


@app.get("/api/translate/progress")
def get_translate_progress():
    """取进度。"""
    with _translate_progress_lock:
        items = list(_translate_progress)
        _translate_progress.clear()
    return {"items": items}


@app.post("/api/translate")
def translate_tags(req: TranslateReq):
    cfg = _load_translate_config()
    tags = _clean_tag_list(req.tags)
    if not tags:
        return {"translations": {}}
    _translate_cancel.clear()  # 重置取消
    with _translate_progress_lock:
        _translate_progress.clear()

    cached: dict[str, str] = {}
    if not req.force:
        with _connect_db() as conn:
            cached = _lookup_translations(conn, tags, increment_hits=True)

    missing = [tag for tag in tags if tag not in cached]
    if missing and cfg["provider"] not in NO_KEY_PROVIDERS and not cfg["api_key"]:
        raise HTTPException(status_code=400, detail="未配置 API 密钥，请先在翻译设置中填写")

    new_translations = (
        _translate_and_store_batches(missing, cfg, raise_first_error=True)
        if missing
        else {}
    )
    all_translations = {**cached, **new_translations}
    still_missing = [tag for tag in missing if tag not in new_translations]
    return {
        "translations": all_translations,
        "cached": len(cached),
        "newly_translated": len(new_translations),
        "missing": still_missing,
        "cancelled": _translate_cancel.is_set(),
    }


# 翻译词库
@app.get("/api/translate/cache")
def get_translation_cache():
    """读词库。"""
    with _connect_db() as conn:
        rows = conn.execute("SELECT en, zh FROM translations").fetchall()
    return {en: zh for en, zh in rows}


class CacheUpdateReq(BaseModel):
    translations: dict
    source: str = "ai"


class CacheLookupReq(BaseModel):
    tags: list[str]


@app.post("/api/translate/cache/lookup")
def lookup_translation_cache(req: CacheLookupReq):
    """查词库。"""
    tags = _clean_tag_list(req.tags)
    if not tags:
        return {"translations": {}, "missing": [], "count": 0, "missing_count": 0}
    with _connect_db() as conn:
        translations = _lookup_translations(conn, tags, increment_hits=True)
    missing = [tag for tag in tags if tag not in translations]
    return {
        "translations": translations,
        "missing": missing,
        "count": len(translations),
        "missing_count": len(missing),
    }


@app.post("/api/translate/cache")
def update_translation_cache(req: CacheUpdateReq):
    """保存词库。"""
    source = req.source if req.source in {"ai", "import", "manual"} else "manual"
    with _connect_db() as conn:
        for en, zh in req.translations.items():
            _store_translation(conn, en, zh, source=source)
        count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    return {"ok": True, "count": count}


@app.get("/api/translate/cache/list")
def list_translation_cache(q: str = "", page: int = 1, size: int = 50):
    """分页查词库。"""
    page = max(page, 1)
    size = min(max(size, 1), 500)
    offset = (page - 1) * size
    with _connect_db() as conn:
        if q:
            like = f"%{q}%"
            total = conn.execute(
                "SELECT COUNT(*) FROM translations WHERE en LIKE ? OR zh LIKE ?",
                (like, like),
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT en, zh, source, hit_count, created_at, updated_at "
                "FROM translations WHERE en LIKE ? OR zh LIKE ? "
                "ORDER BY en LIMIT ? OFFSET ?",
                (like, like, size, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
            rows = conn.execute(
                "SELECT en, zh, source, hit_count, created_at, updated_at "
                "FROM translations ORDER BY en LIMIT ? OFFSET ?",
                (size, offset),
            ).fetchall()
    return {
        "items": [
            {
                "en": en,
                "zh": zh,
                "source": source,
                "hit_count": hit_count,
                "created_at": created_at,
                "updated_at": updated_at,
            }
            for en, zh, source, hit_count, created_at, updated_at in rows
        ],
        "total": total,
        "page": page,
        "size": size,
    }


class CacheEditReq(BaseModel):
    old_en: str
    new_en: str
    new_zh: str


@app.delete("/api/translate/cache")
def delete_translation(en: str = Query(..., description="要删除的英文标签")):
    """删翻译。"""
    with _connect_db() as conn:
        conn.execute("DELETE FROM translations WHERE en = ?", (en,))
        count = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    return {"ok": True, "count": count}


@app.put("/api/translate/cache")
def edit_translation(req: CacheEditReq):
    """改翻译。"""
    old_en = req.old_en.strip()
    new_en = req.new_en.strip()
    new_zh = req.new_zh.strip()
    if not new_en or not new_zh:
        raise HTTPException(status_code=400, detail="英文和中文都不能为空")
    with _connect_db() as conn:
        if old_en == new_en:
            # 改中文
            conn.execute(
                "UPDATE translations SET zh = ?, source = 'manual', updated_at = datetime('now') "
                "WHERE en = ?",
                (new_zh, new_en),
            )
        else:
            # 改 key
            conn.execute("DELETE FROM translations WHERE en = ?", (old_en,))
            _store_translation(conn, new_en, new_zh, source="manual")
    return {"ok": True, "old_en": old_en, "new_en": new_en, "new_zh": new_zh}


@app.post("/api/translate/workspace")
def translate_workspace_tags():
    """翻译工作目录。"""
    _translate_cancel.clear()  # 重置取消
    with _translate_progress_lock:
        _translate_progress.clear()
    cfg = _load_config()
    ws = cfg.get("default_workspace", "")
    if not ws:
        raise HTTPException(status_code=400, detail="未设置工作目录")
    d = Path(ws)
    if not d.is_dir():
        raise HTTPException(status_code=400, detail="工作目录不存在")
    # 找标签文件
    image_exts = IMAGE_EXTENSIONS
    txt_files = []
    for img in d.rglob("*"):
        if img.suffix.lower() in image_exts:
            txt = img.with_suffix(".txt")
            if txt.exists():
                txt_files.append(txt)
    if not txt_files:
        raise HTTPException(status_code=400, detail="未找到有对应图片的标签文件")
    # 收集标签
    all_tags = set()
    for txt in txt_files:
        all_tags.update(_read_tags_file(txt))
    if not all_tags:
        raise HTTPException(status_code=400, detail="标签文件中没有有效标签")
    # 过滤已翻译
    sorted_tags = sorted(all_tags)
    with _connect_db() as conn:
        existing = _lookup_translations(conn, sorted_tags, increment_hits=True)
    untranslated = [tag for tag in sorted_tags if tag not in existing]
    if not untranslated:
        return {
            "ok": True,
            "total_images": len(txt_files),
            "total_tags": len(all_tags),
            "already_translated": len(existing),
            "newly_translated": 0,
            "db_total": len(existing),
        }
    # 分批翻译
    tr_cfg = _load_translate_config()
    if tr_cfg["provider"] not in NO_KEY_PROVIDERS and not tr_cfg["api_key"]:
        raise HTTPException(status_code=400, detail="未配置翻译 API 密钥")
    new_translations = _translate_and_store_batches(
        untranslated,
        tr_cfg,
        raise_first_error=False,
    )
    # 统计词库
    with _connect_db() as conn:
        db_total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    return {
        "ok": True,
        "total_images": len(txt_files),
        "total_tags": len(all_tags),
        "already_translated": len(existing),
        "newly_translated": len(new_translations),
        "db_total": db_total,
        "cancelled": _translate_cancel.is_set(),
    }


# 词库导入导出
DICT_MAGIC = b"TAGDICT"   # 魔数
DICT_VERSION = 1           # 版本


def _serialize_dict(entries: list[dict]) -> bytes:
    """序列化词库。"""
    payload = json.dumps(
        {
            "meta": {
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(entries),
                "version": DICT_VERSION,
                "schema": "translations-v2",
            },
            "entries": entries,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    compressed = zlib.compress(payload, level=9)
    header = DICT_MAGIC + bytes([DICT_VERSION]) + len(compressed).to_bytes(4, "big")
    return header + compressed


def _deserialize_dict(data: bytes) -> dict:
    """反序列化词库。"""
    if len(data) < 12:
        raise ValueError("文件过短，不是有效的词库文件")
    if data[:7] != DICT_MAGIC:
        raise ValueError("文件格式不正确，不是有效的词库文件")
    version = data[7]
    if version != DICT_VERSION:
        raise ValueError(f"不支持的词库文件版本: {version}")
    payload_len = int.from_bytes(data[8:12], "big")
    if len(data) != 12 + payload_len:
        raise ValueError("文件已损坏，数据长度不匹配")
    compressed = data[12 : 12 + payload_len]
    try:
        payload = zlib.decompress(compressed)
        return json.loads(payload.decode("utf-8"))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError("词库文件解析失败，格式不正确") from e


def _backup_translation_db(reason: str) -> str:
    """备份词库。"""
    if not TRANSLATIONS_DB.exists():
        return ""
    TRANSLATIONS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = TRANSLATIONS_BACKUP_DIR / f"translations_{reason}_{stamp}.db"
    with sqlite3.connect(str(TRANSLATIONS_DB)) as src, sqlite3.connect(str(backup_path)) as dst:
        src.backup(dst)
    return str(backup_path)


@app.get("/api/translate/export")
def export_translation_cache():
    """导出词库。"""
    with _connect_db() as conn:
        rows = conn.execute(
            "SELECT en, zh, source, hit_count, created_at, updated_at "
            "FROM translations ORDER BY en"
        ).fetchall()
    entries = [
        {
            "en": en,
            "zh": zh,
            "source": source,
            "hit_count": hit_count,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        for en, zh, source, hit_count, created_at, updated_at in rows
    ]
    data = _serialize_dict(entries)
    filename = f"tagdict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tagdict"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/translate/import")
async def import_translation_cache(
    request: Request,
    mode: str = Query("merge", description="导入模式: merge=去重添加, overwrite=覆盖"),
):
    """导入词库。"""
    if mode not in {"merge", "overwrite"}:
        raise HTTPException(status_code=400, detail="导入模式无效")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="未收到文件数据")
    try:
        obj = _deserialize_dict(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="词库文件解析失败，格式不正确")
    entries = obj.get("entries", [])
    if not entries:
        raise HTTPException(status_code=400, detail="词库文件中没有有效条目")
    backup_path = _backup_translation_db("import")
    added = 0
    skipped = 0
    deleted = 0
    total = 0
    with _connect_db() as conn:
        if mode == "overwrite":
            # 覆盖导入
            deleted = conn.execute(
                "SELECT COUNT(*) FROM translations"
            ).fetchone()[0]
            conn.execute("DELETE FROM translations")
            for e in entries:
                en = (e.get("en") or "").strip()
                zh = (e.get("zh") or "").strip()
                source = e.get("source") if e.get("source") in {"ai", "import", "manual"} else "import"
                if not en or not zh:
                    continue
                if _store_translation(conn, en, zh, source=source, overwrite=True):
                    added += 1
        else:
            # 合并导入
            for e in entries:
                en = (e.get("en") or "").strip()
                zh = (e.get("zh") or "").strip()
                source = e.get("source") if e.get("source") in {"ai", "import", "manual"} else "import"
                if not en or not zh:
                    continue
                if _store_translation(conn, en, zh, source=source, overwrite=False):
                    added += 1
                else:
                    skipped += 1
        total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
    return {
        "ok": True,
        "added": added,
        "skipped": skipped,
        "deleted": deleted,
        "total": total,
        "backup": backup_path,
    }


# 前端
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")

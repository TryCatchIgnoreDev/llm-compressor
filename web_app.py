"""FastAPI Web Server and SSE streaming API for LLM Lossless Compression Lab."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import uuid
import zlib

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from arithmetic_coder import DEFAULT_CDF_BITS
from compressor import (
    ComparisonResult,
    HEADER_STRUCT,
    LLMCompressor,
    MAX_INLINE_BITS,
    ModelRunResult,
    bitstring_to_bytes,
    build_sparse_model_payload,
    bytes_to_bitstring,
    bytes_to_hex,
    compare_models,
    comparison_to_json_dict,
    comparison_to_sparse_json_dict,
    discover_gguf_models,
    discover_system_gguf_models,
    hex_to_bytes,
    llama_cpp,
    token_trace_to_dicts,
)


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
RUNS_DIR = APP_DIR / "runs"
MAX_SAVED_RUNS = 12

app = FastAPI(title="LLM Lossless Compression Lab", version="1.0.0")
_RUN_LOCK = threading.Lock()


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=0)
    model_a_path: str = Field(..., min_length=1)
    model_b_path: str = Field(..., min_length=1)
    n_ctx: int = Field(default=2048, ge=16, le=131072)
    n_gpu_layers_a: int = Field(default=-1, ge=-1, le=1000)
    n_gpu_layers_b: int = Field(default=-1, ge=-1, le=1000)
    cdf_bits: int = Field(default=DEFAULT_CDF_BITS, ge=16, le=32)
    streams: int = Field(default=1, ge=1, le=32)
    fast_cdf: bool = True
    collect_trace: bool = True


class SingleCompressRequest(BaseModel):
    text: str = Field(..., min_length=0)
    model_path: str = Field(..., min_length=1)
    n_ctx: int = Field(default=2048, ge=16, le=131072)
    n_gpu_layers: int = Field(default=-1, ge=-1, le=1000)
    cdf_bits: int = Field(default=DEFAULT_CDF_BITS, ge=16, le=32)
    streams: int = Field(default=1, ge=1, le=32)
    fast_cdf: bool = True
    verify_roundtrip: bool = True
    collect_trace: bool = True


class BitDecompressRequest(BaseModel):
    bitstring: str = Field(..., min_length=1)
    model_path: str = Field(..., min_length=1)
    n_ctx: int = Field(default=2048, ge=16, le=131072)
    n_gpu_layers: int = Field(default=-1, ge=-1, le=1000)


class HexDecompressRequest(BaseModel):
    hexstring: str = Field(..., min_length=1)
    model_path: str = Field(..., min_length=1)
    n_ctx: int = Field(default=2048, ge=16, le=131072)
    n_gpu_layers: int = Field(default=-1, ge=-1, le=1000)


def _bitstring_payload_fields(bin_path: Path | None, compressed_bytes: int) -> dict:
    total_bits = compressed_bytes * 8
    if bin_path is not None and total_bits <= MAX_INLINE_BITS and bin_path.is_file():
        raw = bin_path.read_bytes()
        return {
            "total_bits": total_bits,
            "max_inline_bits": MAX_INLINE_BITS,
            "bitstring_available": True,
            "bitstring": bytes_to_bitstring(raw, max_bits=MAX_INLINE_BITS),
            "hexstring": bytes_to_hex(raw, max_bits=MAX_INLINE_BITS),
        }
    return {
        "total_bits": total_bits,
        "max_inline_bits": MAX_INLINE_BITS,
        "bitstring_available": False,
        "bitstring": None,
        "hexstring": None,
    }


def _prune_old_runs() -> None:
    if not RUNS_DIR.is_dir():
        return
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in dirs[MAX_SAVED_RUNS:]:
        shutil.rmtree(old, ignore_errors=True)


def _serialize_compression(comp) -> dict:
    top1_hits = sum(1 for entry in comp.token_trace if entry.model_rank == 0)
    trace_count = len(comp.token_trace)
    return {
        "original_bytes": comp.original_bytes,
        "compressed_bytes": comp.compressed_bytes,
        "header_bytes": comp.header_bytes,
        "payload_bytes": comp.payload_bytes,
        "num_tokens": comp.num_tokens,
        "vocab_size": comp.vocab_size,
        "cdf_bits": comp.cdf_bits,
        "bos_token_id": comp.bos_token_id,
        "streams_used": comp.streams_used,
        "fast_cdf": comp.fast_cdf,
        "theoretical_entropy_bits": round(comp.theoretical_entropy_bits, 3),
        "theoretical_entropy_bytes": round(comp.theoretical_entropy_bits / 8.0, 2),
        "compression_ratio": round(comp.compression_ratio, 4),
        "bits_per_byte": round(comp.bits_per_byte, 4),
        "bits_per_token": round(comp.bits_per_token, 4),
        "space_savings_percent": round(comp.space_savings_percent, 2),
        "coder_efficiency_percent": round(comp.coder_efficiency_percent, 2),
        "elapsed_seconds": round(comp.elapsed_seconds, 3),
        "top1_hit_count": top1_hits,
        "top1_hit_rate_percent": round((top1_hits / trace_count * 100.0), 2) if trace_count > 0 else 0.0,
        "token_trace": token_trace_to_dicts(comp.token_trace),
    }


def _serialize_model_run(
    run: ModelRunResult,
    *,
    run_id: str,
    filename: str,
    include_bits: bool = True,
) -> dict:
    comp = run.compression
    dec = run.decompression
    bin_path = Path(run.binary_path) if (include_bits and run.binary_path) else None
    bits_info = _bitstring_payload_fields(bin_path, comp.compressed_bytes)
    return {
        "label": run.label,
        "model_path": run.model_path,
        "model_name": run.model_name,
        "model_size_gb": run.model_size_gb,
        "n_ctx": run.n_ctx,
        "n_gpu_layers": run.n_gpu_layers,
        "load_seconds": round(run.load_seconds, 3),
        "exact_match": run.exact_match,
        "original_sha256": run.original_sha256,
        "restored_sha256": run.restored_sha256,
        "header_hex": run.header_hex,
        "restored_text": run.restored_text,
        "download_url": f"/api/runs/{run_id}/{filename}",
        "binary_filename": filename,
        **bits_info,
        "compression": _serialize_compression(comp),
        "decompression": {
            "compressed_bytes": dec.compressed_bytes,
            "restored_bytes": dec.restored_bytes,
            "num_tokens": dec.num_tokens,
            "cdf_bits": dec.cdf_bits,
            "streams_used": dec.streams_used,
            "fast_cdf": dec.fast_cdf,
            "elapsed_seconds": round(dec.elapsed_seconds, 3),
        },
    }


def _serialize_comparison(res: ComparisonResult, *, run_id: str) -> dict:
    a_dict = _serialize_model_run(res.model_a, run_id=run_id, filename="model_a.llmc")
    b_dict = _serialize_model_run(res.model_b, run_id=run_id, filename="model_b.llmc")

    a_bytes = res.model_a.compression.compressed_bytes
    b_bytes = res.model_b.compression.compressed_bytes
    a_bpb = res.model_a.compression.bits_per_byte
    b_bpb = res.model_b.compression.bits_per_byte
    a_total_time = res.model_a.compression.elapsed_seconds + res.model_a.decompression.elapsed_seconds
    b_total_time = res.model_b.compression.elapsed_seconds + res.model_b.decompression.elapsed_seconds

    return {
        "run_id": run_id,
        "original_text": res.original_text,
        "original_bytes": res.original_bytes,
        "original_chars": res.original_chars,
        "zlib_bytes": res.zlib_bytes,
        "zlib_ratio": round(res.original_bytes / res.zlib_bytes, 4) if res.zlib_bytes > 0 else 0.0,
        "winner_by_size": res.winner_by_size,
        "faster_model": "A" if a_total_time < b_total_time else ("B" if b_total_time < a_total_time else "TIE"),
        "byte_difference_b_minus_a": b_bytes - a_bytes,
        "bpb_difference_b_minus_a": round(b_bpb - a_bpb, 4),
        "ratio_difference_a_minus_b": round(res.ratio_difference, 4),
        "entropy_difference_b_minus_a_bits": round(res.entropy_difference_bits, 3),
        "time_difference_b_minus_a_sec": round(b_total_time - a_total_time, 3),
        "report_json_url": f"/api/runs/{run_id}/comparison_report.json",
        "model_a_json_url": f"/api/runs/{run_id}/model_a_tokens.json",
        "model_b_json_url": f"/api/runs/{run_id}/model_b_tokens.json",
        "sparse_report_json_url": f"/api/runs/{run_id}/comparison_report_sparse.json",
        "model_a_sparse_json_url": f"/api/runs/{run_id}/model_a_tokens_sparse.json",
        "model_b_sparse_json_url": f"/api/runs/{run_id}/model_b_tokens_sparse.json",
        "model_a": a_dict,
        "model_b": b_dict,
    }


def _sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="UI file not found.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/system")
async def api_system(
    scan_dir: str | None = Query(default=None),
    scan_system: bool = Query(default=False),
) -> dict:
    if scan_system:
        models = await asyncio.to_thread(discover_system_gguf_models)
        scan_mode = "system"
    elif scan_dir:
        custom = Path(scan_dir).expanduser()
        if not custom.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory not found: {custom}")
        models = await asyncio.to_thread(discover_gguf_models, [custom])
        scan_mode = "folder"
    else:
        models = await asyncio.to_thread(discover_gguf_models, None)
        scan_mode = "env" if models else "none"
    return {
        "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
        "default_cdf_bits": DEFAULT_CDF_BITS,
        "max_inline_bits": MAX_INLINE_BITS,
        "scan_mode": scan_mode,
        "models": models,
    }


@app.get("/api/system/scan_stream")
async def api_system_scan_stream(request: Request) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    cancel_event = threading.Event()

    def push_event(ev: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def worker() -> None:
        t0 = time.perf_counter()
        try:
            models = discover_system_gguf_models(
                progress_callback=push_event,
                cancel_event=cancel_event,
            )
            push_event(
                {
                    "type": "complete",
                    "cancelled": cancel_event.is_set(),
                    "elapsed_seconds": round(time.perf_counter() - t0, 2),
                    "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
                    "default_cdf_bits": DEFAULT_CDF_BITS,
                    "max_inline_bits": MAX_INLINE_BITS,
                    "scan_mode": "system",
                    "models": models,
                }
            )
        except Exception as exc:
            push_event({"type": "error", "message": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                item = await queue.get()
                if item is None:
                    break
                yield _sse_line(item)
        finally:
            cancel_event.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/compare")
async def api_compare(req: CompareRequest) -> StreamingResponse:
    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another compression/comparison job is already running. Please wait for it to finish.",
        )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_runs()

    last_emit = {"t": 0.0}

    def push_event(ev: dict) -> None:
        if ev.get("type") == "model_done":
            # Strip full trace during intermediate event; final complete carries everything.
            m_res: ModelRunResult = ev["result"]
            fname = "model_a.llmc" if m_res.label == "A" else "model_b.llmc"
            summary = _serialize_model_run(
                m_res,
                run_id=run_id,
                filename=fname,
                include_bits=False,
            )
            summary["compression"].pop("token_trace", None)
            ev = {"type": "model_done", "model": ev["model"], "result": summary}
        elif ev.get("type") == "progress":
            now = time.monotonic()
            cur, tot = ev["current"], ev["total"]
            if cur not in (1, tot) and (now - last_emit["t"]) < 0.06:
                return
            last_emit["t"] = now
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def worker() -> None:
        try:
            res = compare_models(
                text=req.text,
                model_a_path=req.model_a_path,
                model_b_path=req.model_b_path,
                output_dir=run_dir,
                n_ctx=req.n_ctx,
                n_gpu_layers_a=req.n_gpu_layers_a,
                n_gpu_layers_b=req.n_gpu_layers_b,
                cdf_bits=req.cdf_bits,
                streams=req.streams,
                fast_cdf=req.fast_cdf,
                collect_trace=req.collect_trace,
                event_callback=push_event,
            )
            full_report = comparison_to_json_dict(res)
            sparse_report = comparison_to_sparse_json_dict(res)
            json_outputs = (
                ("comparison_report.json", full_report),
                ("model_a_tokens.json", full_report["model_a"]),
                ("model_b_tokens.json", full_report["model_b"]),
                ("comparison_report_sparse.json", sparse_report),
                ("model_a_tokens_sparse.json", sparse_report["model_a"]),
                ("model_b_tokens_sparse.json", sparse_report["model_b"]),
            )
            for fname, doc in json_outputs:
                (run_dir / fname).write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            (RUNS_DIR / "last_comparison_run_id.txt").write_text(run_id, encoding="utf-8")
            push_event({"type": "complete", "comparison": _serialize_comparison(res, run_id=run_id)})
        except Exception as exc:
            push_event({"type": "error", "message": str(exc)})
        finally:
            _RUN_LOCK.release()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse_line(item)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/single/compress")
async def api_single_compress(req: SingleCompressRequest) -> StreamingResponse:
    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another job is already running. Please wait for it to finish.",
        )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_runs()

    last_emit = {"t": 0.0}

    def push_event(ev: dict) -> None:
        if ev.get("type") == "progress":
            now = time.monotonic()
            if ev["current"] not in (1, ev["total"]) and (now - last_emit["t"]) < 0.06:
                return
            last_emit["t"] = now
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def worker() -> None:
        try:
            resolved = Path(req.model_path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Model not found: {resolved}")

            bin_path = run_dir / "compressed.llmc"
            raw_bytes = req.text.encode("utf-8")
            orig_sha = hashlib.sha256(raw_bytes).hexdigest()

            push_event({"type": "status", "stage": "loading", "message": f"Loading model: {resolved.name}"})
            t_load0 = time.perf_counter()
            with LLMCompressor(
                resolved,
                n_ctx=req.n_ctx,
                n_gpu_layers=req.n_gpu_layers,
                cdf_bits=req.cdf_bits,
                fast_cdf=req.fast_cdf,
                streams=req.streams,
            ) as comp:
                load_seconds = time.perf_counter() - t_load0

                def on_prog(phase: str, cur: int, tot: int) -> None:
                    push_event({"type": "progress", "phase": phase, "current": cur, "total": tot})

                push_event({"type": "status", "stage": "encode", "message": "Encoding..."})
                comp_stats = comp.compress_to_file(
                    req.text,
                    bin_path,
                    collect_trace=req.collect_trace,
                    progress_callback=on_prog,
                )

                restored_text = ""
                dec_stats = None
                exact_match = None
                restored_sha = None
                if req.verify_roundtrip:
                    push_event({"type": "status", "stage": "decode", "message": "Decoding from binary file..."})
                    restored_text, dec_stats = comp.decompress_from_file(bin_path, progress_callback=on_prog)
                    restored_bytes = restored_text.encode("utf-8")
                    restored_sha = hashlib.sha256(restored_bytes).hexdigest()
                    exact_match = restored_bytes == raw_bytes

            header_hex = bin_path.read_bytes()[: HEADER_STRUCT.size].hex(" ")
            zlib_bytes = len(zlib.compress(raw_bytes, level=9))
            bits_info = _bitstring_payload_fields(bin_path, comp_stats.compressed_bytes)

            payload = {
                "run_id": run_id,
                "model_name": resolved.name,
                "model_size_gb": round(resolved.stat().st_size / (1024 ** 3), 2),
                "load_seconds": round(load_seconds, 3),
                "zlib_bytes": zlib_bytes,
                "download_url": f"/api/runs/{run_id}/compressed.llmc",
                "json_url": f"/api/runs/{run_id}/single_tokens.json",
                "sparse_json_url": f"/api/runs/{run_id}/single_tokens_sparse.json",
                "header_hex": header_hex,
                "original_sha256": orig_sha,
                "restored_sha256": restored_sha,
                "exact_match": exact_match,
                "restored_text": restored_text if req.verify_roundtrip else None,
                **bits_info,
                "compression": _serialize_compression(comp_stats),
                "decompression": (
                    {
                        "compressed_bytes": dec_stats.compressed_bytes,
                        "restored_bytes": dec_stats.restored_bytes,
                        "num_tokens": dec_stats.num_tokens,
                        "streams_used": dec_stats.streams_used,
                        "fast_cdf": dec_stats.fast_cdf,
                        "elapsed_seconds": round(dec_stats.elapsed_seconds, 3),
                    }
                    if dec_stats is not None
                    else None
                ),
            }
            (run_dir / "single_tokens.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "model_name": payload["model_name"],
                        "model_path": str(resolved),
                        "original_text": req.text,
                        "zlib_bytes": zlib_bytes,
                        "summary": {k: v for k, v in payload["compression"].items() if k != "token_trace"},
                        "decompression": payload["decompression"],
                        "tokens": payload["compression"]["token_trace"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            sparse_single = build_sparse_model_payload(
                comp=comp_stats,
                dec=dec_stats,
                model_name=payload["model_name"],
                exact_match=exact_match,
            )
            (run_dir / "single_tokens_sparse.json").write_text(
                json.dumps(sparse_single, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            push_event({"type": "complete", "result": payload})
        except Exception as exc:
            push_event({"type": "error", "message": str(exc)})
        finally:
            _RUN_LOCK.release()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse_line(item)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _decode_blob_sync(*, blob: bytes, model_path: str, n_ctx: int, n_gpu_layers: int) -> dict:
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model not found: {resolved}")
    with LLMCompressor(resolved, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers) as comp:
        restored, dec_stats = comp.decompress_bytes(blob)
    return {
        "model_name": resolved.name,
        "restored_text": restored,
        "compressed_bytes": dec_stats.compressed_bytes,
        "compressed_bits": dec_stats.compressed_bytes * 8,
        "restored_bytes": dec_stats.restored_bytes,
        "num_tokens": dec_stats.num_tokens,
        "cdf_bits": dec_stats.cdf_bits,
        "streams_used": dec_stats.streams_used,
        "fast_cdf": dec_stats.fast_cdf,
        "elapsed_seconds": round(dec_stats.elapsed_seconds, 3),
        "restored_sha256": hashlib.sha256(restored.encode("utf-8")).hexdigest(),
    }


async def _decode_blob_locked(*, blob: bytes, model_path: str, n_ctx: int, n_gpu_layers: int) -> dict:
    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another job is already running.")
    try:
        return await asyncio.to_thread(
            _decode_blob_sync,
            blob=blob,
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _RUN_LOCK.release()


@app.post("/api/single/decompress")
async def api_single_decompress(
    request: Request,
    model_path: str = Query(..., min_length=1),
    n_ctx: int = Query(default=2048, ge=16, le=131072),
    n_gpu_layers: int = Query(default=-1, ge=-1, le=1000),
) -> dict:
    blob = await request.body()
    if not blob:
        raise HTTPException(status_code=400, detail="Empty binary file uploaded.")
    return await _decode_blob_locked(
        blob=blob,
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )


@app.post("/api/single/decompress_bits")
async def api_single_decompress_bits(req: BitDecompressRequest) -> dict:
    try:
        blob = bitstring_to_bytes(req.bitstring, max_bits=MAX_INLINE_BITS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _decode_blob_locked(
        blob=blob,
        model_path=req.model_path,
        n_ctx=req.n_ctx,
        n_gpu_layers=req.n_gpu_layers,
    )


@app.post("/api/single/decompress_hex")
async def api_single_decompress_hex(req: HexDecompressRequest) -> dict:
    try:
        blob = hex_to_bytes(req.hexstring, max_bits=MAX_INLINE_BITS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _decode_blob_locked(
        blob=blob,
        model_path=req.model_path,
        n_ctx=req.n_ctx,
        n_gpu_layers=req.n_gpu_layers,
    )


@app.get("/api/compare/last/json")
def api_compare_last_json(
    model: str | None = Query(default=None, pattern="^(a|b|A|B)$"),
    detail: str = Query(default="full", pattern="^(full|sparse)$"),
    download: bool = Query(default=False),
) -> JSONResponse:
    pointer = RUNS_DIR / "last_comparison_run_id.txt"
    if not pointer.is_file():
        raise HTTPException(status_code=404, detail="No completed comparison run found yet.")
    last_id = pointer.read_text(encoding="utf-8").strip()
    if not last_id.isalnum():
        raise HTTPException(status_code=404, detail="Last comparison run record is invalid.")

    suffix = "_sparse" if detail == "sparse" else ""
    if model is None:
        filename = f"comparison_report{suffix}.json"
    elif model.lower() == "a":
        filename = f"model_a_tokens{suffix}.json"
    else:
        filename = f"model_b_tokens{suffix}.json"

    target = RUNS_DIR / last_id / filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Last comparison JSON report not found.")

    data = json.loads(target.read_text(encoding="utf-8"))
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else None
    return JSONResponse(content=data, headers=headers)


@app.get("/api/runs/{run_id}/{filename}")
def api_download_artifact(run_id: str, filename: str) -> FileResponse:
    allowed = {
        "model_a.llmc",
        "model_b.llmc",
        "compressed.llmc",
        "comparison_report.json",
        "model_a_tokens.json",
        "model_b_tokens.json",
        "comparison_report_sparse.json",
        "model_a_tokens_sparse.json",
        "model_b_tokens_sparse.json",
        "single_tokens.json",
        "single_tokens_sparse.json",
    }
    if not run_id.isalnum() or filename not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    path = RUNS_DIR / run_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    media = "application/json" if filename.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=filename)

"""Command-line interface for lossless LLM + Arithmetic Coding text compression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import zlib

from arithmetic_coder import DEFAULT_CDF_BITS
from compressor import (
    LLMCompressor,
    MAX_INLINE_BITS,
    bytes_to_bitstring,
    bytes_to_hex,
    compare_models,
    comparison_to_json_dict,
    comparison_to_sparse_json_dict,
)


def _read_text_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input is not None:
        return Path(args.input).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide either --text, --input <file>, or pipe text via stdin.")


def _print_compression_summary(stats, *, raw_bytes: bytes) -> None:
    zlib_size = len(zlib.compress(raw_bytes, level=9))
    print(f"Model                 : {stats.model_name}")
    print(f"Mode                  : streams={stats.streams_used}, cdf={'sparse' if stats.fast_cdf else 'exact-dense'}")
    print(f"Vocab size / CDF bits : {stats.vocab_size:,} / {stats.cdf_bits} (total_freq=1<<{stats.cdf_bits})")
    print(f"BOS token id          : {stats.bos_token_id}")
    print(f"Tokens encoded        : {stats.num_tokens:,}")
    print(f"Original size         : {stats.original_bytes:,} bytes")
    print(
        f"Compressed size       : {stats.compressed_bytes:,} bytes "
        f"(header={stats.header_bytes} B, payload={stats.payload_bytes} B)"
    )
    print(f"Compression ratio     : {stats.compression_ratio:.3f}x ({stats.space_savings_percent:.2f}% saved)")
    print(f"Bits per byte         : {stats.bits_per_byte:.3f} bpb (raw=8.000 bpb)")
    print(f"Bits per token        : {stats.bits_per_token:.3f} bpt")
    print(
        f"Theoretical entropy   : {stats.theoretical_entropy_bits:.2f} bits "
        f"({stats.theoretical_entropy_bits / 8.0:.1f} bytes)"
    )
    print(f"Baseline zlib -9      : {zlib_size:,} bytes")
    print(f"Encode time           : {stats.elapsed_seconds:.3f} s")


def cmd_compress(args: argparse.Namespace) -> int:
    if not args.output and not args.bits and not args.hex:
        raise ValueError("Provide --output <file.llmc>, --bits, and/or --hex.")

    text = _read_text_input(args)
    with LLMCompressor(
        args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.gpu_layers,
        n_threads=args.threads,
        cdf_bits=args.cdf_bits,
        fast_cdf=not args.exact_cdf,
        streams=args.streams,
        verbose=args.verbose,
    ) as comp:
        blob, stats = comp.compress_bytes(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)
        print(f"Wrote compressed binary to: {out_path.resolve()}")

    _print_compression_summary(stats, raw_bytes=text.encode("utf-8"))

    total_bits = len(blob) * 8
    if args.hex:
        if total_bits <= MAX_INLINE_BITS:
            hexstr = bytes_to_hex(blob, max_bits=MAX_INLINE_BITS)
            print(f"Hex ({len(hexstr):,} chars / {total_bits:,} / {MAX_INLINE_BITS:,} bit limit):")
            print(hexstr)
        else:
            print(
                f"Inline hex output omitted: {total_bits:,} bits exceeds "
                f"{MAX_INLINE_BITS:,}-bit limit (use --output binary file instead)."
            )

    if args.bits:
        if total_bits <= MAX_INLINE_BITS:
            bitstr = bytes_to_bitstring(blob, max_bits=MAX_INLINE_BITS)
            print(f"Bits ({total_bits:,} / {MAX_INLINE_BITS:,} limit):")
            print(bitstr)
        else:
            print(
                f"Inline bit output omitted: {total_bits:,} bits exceeds "
                f"{MAX_INLINE_BITS:,}-bit limit (use --output binary file instead)."
            )
    return 0


def cmd_decompress(args: argparse.Namespace) -> int:
    sources = sum(bool(x) for x in (args.input, args.bits, args.hex))
    if sources != 1:
        raise ValueError("Specify exactly one of --input <file.llmc>, --bits <bitstring>, or --hex <hexstring>.")

    with LLMCompressor(
        args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.gpu_layers,
        n_threads=args.threads,
        verbose=args.verbose,
    ) as comp:
        if args.bits:
            restored, stats = comp.decompress_bitstring(args.bits, max_bits=MAX_INLINE_BITS)
        elif args.hex:
            restored, stats = comp.decompress_hex(args.hex, max_bits=MAX_INLINE_BITS)
        else:
            restored, stats = comp.decompress_from_file(args.input)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(restored, encoding="utf-8", newline="")
        print(f"Wrote restored text ({stats.restored_bytes:,} bytes) to: {out_path.resolve()}")
    else:
        print(restored)
    print(
        f"Decoded {stats.num_tokens:,} tokens ({stats.compressed_bytes:,} -> "
        f"{stats.restored_bytes:,} bytes) in {stats.elapsed_seconds:.3f} s"
    )
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    text = _read_text_input(args)
    raw_bytes = text.encode("utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir:
        bin_path = Path(args.output) if args.output else (Path(tmp_dir) / "compressed.llmc")

        with LLMCompressor(
            args.model,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.gpu_layers,
            n_threads=args.threads,
            cdf_bits=args.cdf_bits,
            fast_cdf=not args.exact_cdf,
            streams=args.streams,
            verbose=args.verbose,
        ) as comp:
            comp_stats = comp.compress_to_file(text, bin_path)
            restored, dec_stats = comp.decompress_from_file(bin_path)

    exact_match = restored == text
    print("=== COMPRESSION ===")
    print(f"Binary file           : {bin_path}")
    _print_compression_summary(comp_stats, raw_bytes=raw_bytes)
    print("=== DECOMPRESSION ===")
    print(f"Decode time           : {dec_stats.elapsed_seconds:.3f} s")
    print(f"Restored size         : {dec_stats.restored_bytes:,} bytes")
    print(f"Lossless exact match  : {exact_match}")

    if not exact_match:
        print("ERROR: Restored text does not match original!", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    text = _read_text_input(args)
    gpu_a = args.gpu_layers_a if args.gpu_layers_a is not None else args.gpu_layers
    gpu_b = args.gpu_layers_b if args.gpu_layers_b is not None else args.gpu_layers
    want_full = bool(args.json or args.json_out or args.json_dir)
    want_sparse = bool(args.sparse_json or args.sparse_json_out or args.json_dir)
    want_trace = want_full or want_sparse

    result = compare_models(
        text=text,
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        output_dir=args.output_dir,
        n_ctx=args.n_ctx,
        n_gpu_layers_a=gpu_a,
        n_gpu_layers_b=gpu_b,
        n_threads=args.threads,
        cdf_bits=args.cdf_bits,
        streams=args.streams,
        fast_cdf=not args.exact_cdf,
        collect_trace=want_trace,
    )

    report = comparison_to_json_dict(result) if want_full else None
    sparse_report = comparison_to_sparse_json_dict(result) if want_sparse else None

    if args.json_out and report is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.sparse_json_out and sparse_report is not None:
        sout_path = Path(args.sparse_json_out)
        sout_path.parent.mkdir(parents=True, exist_ok=True)
        sout_path.write_text(json.dumps(sparse_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json_dir and report is not None and sparse_report is not None:
        jdir = Path(args.json_dir)
        jdir.mkdir(parents=True, exist_ok=True)
        for fname, doc in (
            ("comparison_report.json", report),
            ("model_a_tokens.json", report["model_a"]),
            ("model_b_tokens.json", report["model_b"]),
            ("comparison_report_sparse.json", sparse_report),
            ("model_a_tokens_sparse.json", sparse_report["model_a"]),
            ("model_b_tokens_sparse.json", sparse_report["model_b"]),
        ):
            (jdir / fname).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    a, b = result.model_a, result.model_b
    if args.json and report is not None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.sparse_json and sparse_report is not None:
        print(json.dumps(sparse_report, ensure_ascii=False, indent=2))
    else:
        print("=== MODEL COMPARISON ===")
        print(f"Original size         : {result.original_bytes:,} bytes ({result.original_chars:,} chars)")
        print(f"Baseline zlib -9      : {result.zlib_bytes:,} bytes")
        print("-" * 72)
        for m in (a, b):
            c, d = m.compression, m.decompression
            print(f"[Model {m.label}] {m.model_name} ({m.model_size_gb:.2f} GB, vocab={c.vocab_size:,})")
            print(
                f"  Compressed          : {c.compressed_bytes:,} B "
                f"(payload={c.payload_bytes:,} B, ratio={c.compression_ratio:.3f}x, "
                f"bpb={c.bits_per_byte:.3f}, tokens={c.num_tokens:,})"
            )
            print(
                f"  Entropy / Times     : {c.theoretical_entropy_bits:.2f} bits "
                f"({c.theoretical_entropy_bits / 8.0:.1f} B) | "
                f"enc={c.elapsed_seconds:.3f}s, dec={d.elapsed_seconds:.3f}s | "
                f"lossless={m.exact_match}"
            )
        print("-" * 72)
        print(
            f"Winner (smaller file) : Model {result.winner_by_size} "
            f"(byte delta B-A: {result.byte_difference:+d} B, "
            f"entropy delta B-A: {result.entropy_difference_bits:+.2f} bits)"
        )
        if args.json_out:
            print(f"Wrote combined JSON   : {Path(args.json_out).resolve()}")
        if args.sparse_json_out:
            print(f"Wrote sparse JSON     : {Path(args.sparse_json_out).resolve()}")
        if args.json_dir:
            print(f"Wrote separate JSONs  : {Path(args.json_dir).resolve()}")

    if not (a.exact_match and b.exact_match):
        print("ERROR: Lossless verification failed!", file=sys.stderr)
        return 1
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("web_app:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lossless text compression using GGUF LLMs (llama-cpp-python) and Arithmetic Coding."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, include_cdf: bool = False, include_model: bool = True) -> None:
        if include_model:
            p.add_argument("--model", required=True, help="Path to any .gguf model file.")
        p.add_argument("--n-ctx", type=int, default=2048, help="Context window size (default: 2048).")
        p.add_argument(
            "--gpu-layers",
            type=int,
            default=-1,
            help="Number of layers to offload to GPU (-1 for all, 0 for CPU-only).",
        )
        p.add_argument("--threads", type=int, default=None, help="Optional CPU thread count.")
        p.add_argument("--verbose", action="store_true", help="Enable llama.cpp verbose logging.")
        if include_cdf:
            p.add_argument(
                "--cdf-bits",
                type=int,
                default=DEFAULT_CDF_BITS,
                help=f"Integer CDF precision bits (total_freq = 1 << cdf_bits, default: {DEFAULT_CDF_BITS}).",
            )
            p.add_argument(
                "--streams",
                type=int,
                default=1,
                help="Parallel streams (e.g. 1, 4, 8, 16) for batched GPU speed on longer texts.",
            )
            p.add_argument(
                "--exact-cdf",
                action="store_true",
                help="Disable Sparse CDF and use full-vocabulary Softmax CDF.",
            )

    p_comp = sub.add_parser("compress", help="Compress text into a binary .llmc file, hex, or inline bits.")
    add_common(p_comp, include_cdf=True)
    p_comp.add_argument("--text", help="Inline text to compress.")
    p_comp.add_argument("--input", help="Path to UTF-8 text file to compress.")
    p_comp.add_argument("--output", help="Output binary .llmc file path.")
    p_comp.add_argument(
        "--hex",
        action="store_true",
        help=f"Print copyable hex string if <= {MAX_INLINE_BITS:,} bits.",
    )
    p_comp.add_argument(
        "--bits",
        action="store_true",
        help=f"Print copyable MSB-first bitstring if <= {MAX_INLINE_BITS:,} bits.",
    )
    p_comp.set_defaults(func=cmd_compress)

    p_dec = sub.add_parser("decompress", help="Decompress a binary .llmc file, hex, or inline bits back to text.")
    add_common(p_dec, include_cdf=False)
    p_dec.add_argument("--input", help="Input binary .llmc file path.")
    p_dec.add_argument(
        "--hex",
        help=f"Inline hex string to decode (max {MAX_INLINE_BITS:,} bits / {MAX_INLINE_BITS // 4:,} chars).",
    )
    p_dec.add_argument(
        "--bits",
        help=f"Inline '0'/'1' bitstring to decode (max {MAX_INLINE_BITS:,} bits).",
    )
    p_dec.add_argument("--output", help="Optional output UTF-8 text file path (prints to stdout if omitted).")
    p_dec.set_defaults(func=cmd_decompress)

    p_rt = sub.add_parser(
        "roundtrip",
        help="Compress to a binary file, read back, decompress, and verify lossless equality.",
    )
    add_common(p_rt, include_cdf=True)
    p_rt.add_argument("--text", help="Inline text to test.")
    p_rt.add_argument("--input", help="Path to UTF-8 text file to test.")
    p_rt.add_argument("--output", help="Optional persistent binary .llmc output file path.")
    p_rt.set_defaults(func=cmd_roundtrip)

    p_cmp = sub.add_parser(
        "compare",
        help="Run compress+decompress with Model A then Model B and compare results.",
    )
    add_common(p_cmp, include_cdf=True, include_model=False)
    p_cmp.add_argument("--model-a", required=True, help="Path to Model A .gguf file.")
    p_cmp.add_argument("--model-b", required=True, help="Path to Model B .gguf file.")
    p_cmp.add_argument("--gpu-layers-a", type=int, default=None, help="Optional GPU layers override for Model A.")
    p_cmp.add_argument("--gpu-layers-b", type=int, default=None, help="Optional GPU layers override for Model B.")
    p_cmp.add_argument("--text", help="Inline text to test.")
    p_cmp.add_argument("--input", help="Path to UTF-8 text file to test.")
    p_cmp.add_argument("--output-dir", help="Optional directory to keep model_a.llmc and model_b.llmc.")
    p_cmp.add_argument("--json", action="store_true", help="Print detailed token-by-token JSON report to stdout.")
    p_cmp.add_argument("--json-out", help="Write combined A+B detailed token-by-token JSON to file.")
    p_cmp.add_argument(
        "--sparse-json",
        action="store_true",
        help="Print compact sparse JSON report (summary, top surprises, 5-field tokens) to stdout.",
    )
    p_cmp.add_argument(
        "--sparse-json-out",
        help="Write compact sparse combined A+B JSON report to file.",
    )
    p_cmp.add_argument(
        "--json-dir",
        help="Write full and sparse comparison_report / model_a_tokens / model_b_tokens JSONs into directory.",
    )
    p_cmp.set_defaults(func=cmd_compare)

    p_web = sub.add_parser("web", help="Launch the interactive Web UI server.")
    p_web.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p_web.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Ensure Windows console outputs UTF-8 cleanly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

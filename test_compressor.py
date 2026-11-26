"""End-to-end integration tests for LLMCompressor using a small local GGUF model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from compressor import (
    LLMCompressor,
    MAX_INLINE_BITS,
    bitstring_to_bytes,
    bytes_to_bitstring,
    bytes_to_hex,
    compare_models,
    comparison_to_json_dict,
    comparison_to_sparse_json_dict,
    discover_gguf_models,
    hex_to_bytes,
    model_run_to_json_dict,
    model_run_to_sparse_json_dict,
)


DEFAULT_SMALL_MODEL = (
    Path.home()
    / ".lmstudio"
    / "models"
    / "lmstudio-community"
    / "gemma-4-E2B-it-GGUF"
    / "gemma-4-E2B-it-Q4_K_M.gguf"
)
SECOND_SMALL_MODEL = (
    Path.home()
    / ".lmstudio"
    / "models"
    / "lmstudio-community"
    / "Meta-Llama-3.1-8B-Instruct-GGUF"
    / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)


class TestLLMCompressor(unittest.TestCase):
    compressor: LLMCompressor
    model_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        env_model = os.environ.get("LLM_COMPRESSOR_TEST_MODEL")
        cls.model_path = Path(env_model) if env_model else DEFAULT_SMALL_MODEL
        if not cls.model_path.is_file():
            raise unittest.SkipTest(f"Small test GGUF model not found: {cls.model_path}")
        cls.compressor = LLMCompressor(cls.model_path, n_ctx=512, n_gpu_layers=-1)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "compressor"):
            cls.compressor.close()

    def test_binary_file_roundtrip_ascii_and_unicode(self) -> None:
        samples = [
            (
                "3Blue1Brown explained that compression is deeply connected to intelligence: "
                "the better a language model predicts the next token, the fewer bits an "
                "arithmetic coder needs to represent the message losslessly."
            ),
            (
                "Lossless UTF-8 roundtrip test: accented characters (é, è, ñ, ü, ö, ä, "
                "É, Ñ, Ü, Ω, π, ∑), digits 1234567890, newlines\nand\ttabs with emojis 🚀!"
            ),
            "  Leading spaces,  double spaces, and trailing newline.\r\nSecond line.\n",
            "",
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, text in enumerate(samples):
                bin_path = Path(tmp_dir) / f"sample_{i}.llmc"
                stats = self.compressor.compress_to_file(text, bin_path)
                self.assertTrue(bin_path.is_file())
                self.assertEqual(bin_path.stat().st_size, stats.compressed_bytes)

                restored, dec_stats = self.compressor.decompress_from_file(bin_path)
                self.assertEqual(restored, text)
                self.assertEqual(dec_stats.num_tokens, stats.num_tokens)

                if len(text) > 100:
                    self.assertLess(
                        stats.compressed_bytes,
                        stats.original_bytes,
                        "Natural language text should compress smaller than raw UTF-8 bytes.",
                    )

    def test_exact_cdf_and_multi_stream_roundtrip(self) -> None:
        text = (
            "Multi-stream parallel compression splits long token sequences across independent "
            "KV-cache lanes so GPU tensor cores process multiple tokens per decode step while "
            "maintaining exact 1:1 lossless reconstruction across every chunk boundary. "
            "Unicode check: café naïve Ωπ!"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            for fast_cdf, req_streams in [(False, 1), (True, 1), (True, 4)]:
                bin_path = Path(tmp_dir) / f"mode_{fast_cdf}_{req_streams}.llmc"
                stats = self.compressor.compress_to_file(
                    text,
                    bin_path,
                    streams=req_streams,
                    fast_cdf=fast_cdf,
                    collect_trace=True,
                )
                restored, dec_stats = self.compressor.decompress_from_file(bin_path)
                self.assertEqual(restored, text)
                self.assertEqual(dec_stats.streams_used, stats.streams_used)
                self.assertEqual(len(stats.token_trace), stats.num_tokens)
                self.assertEqual("".join(t.piece for t in stats.token_trace), text)

    def test_bitstring_roundtrip_and_10k_limit(self) -> None:
        text = "Copyable bitstring lossless compression test: café naïve 🚀!"
        bitstr, stats = self.compressor.compress_to_bitstring(text)
        self.assertEqual(len(bitstr), stats.compressed_bytes * 8)
        self.assertLessEqual(len(bitstr), MAX_INLINE_BITS)
        self.assertTrue(set(bitstr).issubset({"0", "1"}))

        # Whitespace-formatted bitstrings should also decode cleanly.
        spaced = " ".join(bitstr[i : i + 8] for i in range(0, len(bitstr), 8))
        restored, _ = self.compressor.decompress_bitstring(spaced)
        self.assertEqual(restored, text)

        # Enforce 10,000-bit ceiling in both directions.
        too_many_bytes = b"\x00" * ((MAX_INLINE_BITS // 8) + 1)  # 10,008 bits
        with self.assertRaises(ValueError):
            bytes_to_bitstring(too_many_bytes)
        with self.assertRaises(ValueError):
            bitstring_to_bytes("0" * (MAX_INLINE_BITS + 8))
        with self.assertRaises(ValueError):
            bitstring_to_bytes("01012")
        with self.assertRaises(ValueError):
            bitstring_to_bytes("010101")  # not a multiple of 8

    def test_hex_roundtrip_and_10k_limit(self) -> None:
        text = "Compact hex mode lossless compression test: café naïve 🚀!"
        hexstr, stats = self.compressor.compress_to_hex(text)
        self.assertEqual(len(hexstr), stats.compressed_bytes * 2)
        self.assertLessEqual(len(hexstr) * 4, MAX_INLINE_BITS)
        self.assertTrue(hexstr.startswith("4c4c4d43"))  # 'LLMC' magic

        # Support 0x prefix and spaced hex formatting.
        spaced = "0x" + " ".join(hexstr[i : i + 2] for i in range(0, len(hexstr), 2))
        restored, _ = self.compressor.decompress_hex(spaced)
        self.assertEqual(restored, text)

        too_many_bytes = b"\x00" * ((MAX_INLINE_BITS // 8) + 1)
        with self.assertRaises(ValueError):
            bytes_to_hex(too_many_bytes)
        with self.assertRaises(ValueError):
            hex_to_bytes("00" * ((MAX_INLINE_BITS // 8) + 1))
        with self.assertRaises(ValueError):
            hex_to_bytes("4c4")  # odd length
        with self.assertRaises(ValueError):
            hex_to_bytes("4c4z")  # invalid hex digit


class TestLLMCompressorLifecycle(unittest.TestCase):
    model_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        env_model = os.environ.get("LLM_COMPRESSOR_TEST_MODEL")
        cls.model_path = Path(env_model) if env_model else DEFAULT_SMALL_MODEL
        if not cls.model_path.is_file():
            raise unittest.SkipTest(f"Small test GGUF model not found: {cls.model_path}")

    def test_separate_encoder_and_decoder_instances(self) -> None:
        text = (
            "In information theory, Claude Shannon proved that the optimal expected code length "
            "for a symbol with probability p is -log2(p) bits. Arithmetic coding approaches this "
            "limit arbitrarily closely without requiring integer-bit codewords per symbol."
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            bin_path = Path(tmp_dir) / "cross_instance.llmc"
            with LLMCompressor(self.model_path, n_ctx=512, n_gpu_layers=-1) as encoder_comp:
                stats = encoder_comp.compress_to_file(text, bin_path)

            # Decode using a brand-new LLMCompressor instance reading from disk.
            with LLMCompressor(self.model_path, n_ctx=512, n_gpu_layers=-1) as decoder_comp:
                restored, dec_stats = decoder_comp.decompress_from_file(bin_path)

            self.assertEqual(restored, text)
            self.assertEqual(dec_stats.num_tokens, stats.num_tokens)
            self.assertGreater(stats.compression_ratio, 2.0)

    def test_context_window_rollover_lossless(self) -> None:
        # Force context rollover by setting n_ctx=32 on a >32-token sequence.
        text = (
            "The quick brown fox jumps over the lazy dog while an autoregressive transformer "
            "predicts each subsequent token probability distribution step by step across "
            "context boundaries without losing a single bit of information."
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with LLMCompressor(self.model_path, n_ctx=32, n_gpu_layers=-1) as small_ctx_comp:
                tokens = small_ctx_comp.tokenize(text)
                self.assertGreater(len(tokens), 32)
                bin_path = Path(tmp_dir) / "rollover.llmc"
                stats = small_ctx_comp.compress_to_file(text, bin_path)
                restored, _ = small_ctx_comp.decompress_from_file(bin_path)
                self.assertEqual(restored, text)
                self.assertEqual(stats.num_tokens, len(tokens))

            # Also verify multi-stream rollover with n_ctx=20 on 2 streams
            bin_multi = Path(tmp_dir) / "rollover_multi.llmc"
            with LLMCompressor(self.model_path, n_ctx=20, n_gpu_layers=-1) as multi_roll:
                m_stats = multi_roll.compress_to_file(text, bin_multi, streams=2)
                m_restored, _ = multi_roll.decompress_from_file(bin_multi)
                self.assertEqual(m_restored, text)
                self.assertEqual(m_stats.streams_used, 2)

    def test_model_hot_swap(self) -> None:
        if not SECOND_SMALL_MODEL.is_file():
            self.skipTest(f"Second test model not found: {SECOND_SMALL_MODEL}")

        text = "Swapping GGUF models dynamically should compress and decompress cleanly."
        with LLMCompressor(self.model_path, n_ctx=512, n_gpu_layers=-1) as swappable:
            blob1 = swappable.compress(text)
            self.assertEqual(swappable.decompress(blob1), text)

            swappable.load_model(SECOND_SMALL_MODEL, n_ctx=512, n_gpu_layers=-1)
            blob2 = swappable.compress(text)
            self.assertEqual(swappable.decompress(blob2), text)
            self.assertNotEqual(blob1, blob2)

    def test_compare_models_with_trace_and_discovery(self) -> None:
        if not SECOND_SMALL_MODEL.is_file():
            self.skipTest(f"Second test model not found: {SECOND_SMALL_MODEL}")

        discovered = discover_gguf_models()
        self.assertGreaterEqual(len(discovered), 2)
        self.assertFalse(any(m["name"].lower().startswith("mmproj") for m in discovered))

        text = (
            "Compression is intelligence: a stronger language model predicts the next token "
            "more accurately and compresses natural language into fewer bits."
        )
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = compare_models(
                text=text,
                model_a_path=self.model_path,
                model_b_path=SECOND_SMALL_MODEL,
                output_dir=tmp_dir,
                n_ctx=512,
                collect_trace=True,
                event_callback=lambda ev: events.append(ev["type"]),
            )
            self.assertTrue(res.model_a.exact_match)
            self.assertTrue(res.model_b.exact_match)
            self.assertEqual(res.model_a.restored_text, text)
            self.assertEqual(res.model_b.restored_text, text)
            self.assertTrue(Path(res.model_a.binary_path).is_file())
            self.assertTrue(Path(res.model_b.binary_path).is_file())
            self.assertEqual(len(res.model_a.compression.token_trace), res.model_a.compression.num_tokens)
            self.assertEqual(len(res.model_b.compression.token_trace), res.model_b.compression.num_tokens)
            self.assertIn("model_done", events)
            self.assertIn(res.winner_by_size, ("A", "B", "TIE"))

            combined_json = comparison_to_json_dict(res)
            model_a_json = model_run_to_json_dict(res.model_a)
            model_b_json = model_run_to_json_dict(res.model_b)
            self.assertEqual(combined_json["model_a"]["label"], "A")
            self.assertEqual(combined_json["model_b"]["label"], "B")
            self.assertEqual(model_a_json["summary"]["num_tokens"], res.model_a.compression.num_tokens)
            self.assertEqual(model_b_json["summary"]["num_tokens"], res.model_b.compression.num_tokens)

            for m_json, stats in (
                (model_a_json, res.model_a.compression),
                (model_b_json, res.model_b.compression),
            ):
                tokens = m_json["tokens"]
                self.assertEqual(len(tokens), stats.num_tokens)
                self.assertGreater(len(tokens), 0)
                self.assertEqual("".join(t["piece"] for t in tokens), text)
                self.assertAlmostEqual(
                    tokens[-1]["cumulative_bits"],
                    stats.theoretical_entropy_bits,
                    delta=0.2,
                )
                for tok in tokens:
                    self.assertGreater(tok["bits"], 0.0)
                    self.assertGreaterEqual(tok["piece_bytes"], 1)
                    self.assertEqual(tok["raw_bits"], tok["piece_bytes"] * 8)
                    self.assertGreaterEqual(tok["emitted_bits"], 0)
                    self.assertGreaterEqual(tok["model_rank"], 0)
                    self.assertEqual(tok["is_top1"], tok["model_rank"] == 0)
                    self.assertIn("top1_piece", tok)

            sparse_combined = comparison_to_sparse_json_dict(res)
            sparse_a = model_run_to_sparse_json_dict(res.model_a)
            sparse_b = model_run_to_sparse_json_dict(res.model_b)
            self.assertEqual(sparse_combined["detail"], "sparse")
            self.assertEqual(sparse_a["label"], "A")
            self.assertEqual(sparse_b["label"], "B")
            self.assertLess(
                len(json.dumps(sparse_combined, ensure_ascii=False)),
                len(json.dumps(combined_json, ensure_ascii=False)),
            )
            for s_json, full_json, stats in (
                (sparse_a, model_a_json, res.model_a.compression),
                (sparse_b, model_b_json, res.model_b.compression),
            ):
                self.assertEqual(len(s_json["tokens"]), stats.num_tokens)
                self.assertEqual("".join(t["piece"] for t in s_json["tokens"]), text)
                self.assertLess(
                    len(json.dumps(s_json, ensure_ascii=False)),
                    len(json.dumps(full_json, ensure_ascii=False)),
                )
                for st in s_json["tokens"]:
                    self.assertEqual(set(st.keys()), {"piece", "prob", "bits", "emitted_bits", "rank"})
                surprises = s_json["top_surprises"]
                self.assertGreater(len(surprises), 0)
                self.assertLessEqual(len(surprises), 10)
                self.assertEqual(surprises, sorted(surprises, key=lambda x: x["bits"], reverse=True))


if __name__ == "__main__":
    unittest.main()

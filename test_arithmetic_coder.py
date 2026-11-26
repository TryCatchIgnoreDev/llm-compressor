"""Unit tests for 62-bit Arithmetic Coding and Integer CDF conversion."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from arithmetic_coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitInputStream,
    BitOutputStream,
    DEFAULT_CDF_BITS,
    SparseCDF,
    logits_to_cdf,
    logits_to_sparse_cdf,
)


class TestArithmeticCoder(unittest.TestCase):
    def test_cdf_invariants_extreme_logits(self) -> None:
        vocab = 2048
        total_freq = 1 << 20
        cases = [
            np.zeros(vocab, dtype=np.float32),
            np.linspace(-500.0, 50.0, vocab, dtype=np.float32),
            np.array([1000.0] + [-1e9] * (vocab - 1), dtype=np.float32),
            np.array([-np.inf] * (vocab - 1) + [0.0], dtype=np.float32),
            np.array([-np.inf] * vocab, dtype=np.float32),
        ]
        for logits in cases:
            cdf = logits_to_cdf(logits, total_freq=total_freq)
            self.assertEqual(cdf.shape, (vocab + 1,))
            self.assertEqual(int(cdf[0]), 0)
            self.assertEqual(int(cdf[-1]), total_freq)
            diffs = np.diff(cdf)
            self.assertTrue(np.all(diffs >= 1), "Every symbol must have freq >= 1")

    def test_roundtrip_dynamic_cdfs_large_vocab(self) -> None:
        rng = np.random.default_rng(42)
        vocab = 128_256
        total_freq = 1 << DEFAULT_CDF_BITS
        n_steps = 80

        # Simulate peaked LLM logits at every step.
        step_logits = []
        true_symbols = []
        theoretical_bits = 0.0

        for _ in range(n_steps):
            logits = rng.normal(-8.0, 2.5, size=vocab).astype(np.float32)
            # Pick a few likely candidates
            top_k = rng.choice(vocab, size=10, replace=False)
            logits[top_k] = rng.uniform(0.0, 10.0, size=10).astype(np.float32)

            shifted = logits.astype(np.float64) - float(np.max(logits))
            probs = np.exp(shifted)
            probs /= np.sum(probs)

            sym = int(rng.choice(vocab, p=probs))
            theoretical_bits += -math.log2(float(probs[sym]))
            step_logits.append(logits)
            true_symbols.append(sym)

        bit_out = BitOutputStream()
        encoder = ArithmeticEncoder(bit_out)
        for logits, sym in zip(step_logits, true_symbols):
            cdf = logits_to_cdf(logits, total_freq=total_freq)
            encoder.encode_symbol(sym, cdf)
        compressed = encoder.finish()

        # Compression should closely match theoretical Shannon cross-entropy.
        actual_bits = len(compressed) * 8
        self.assertLess(actual_bits, theoretical_bits + 128.0)

        bit_in = BitInputStream(compressed)
        decoder = ArithmeticDecoder(bit_in)
        decoded_symbols = []
        for logits in step_logits:
            cdf = logits_to_cdf(logits, total_freq=total_freq)
            decoded_symbols.append(decoder.decode_symbol(cdf))

        self.assertEqual(decoded_symbols, true_symbols)

    def test_underflow_stress_near_half(self) -> None:
        # Stress E3 underflow chains with balanced 50/50 probabilities.
        total_freq = 1 << 16
        logits = np.array([0.0, 0.0, -50.0, -50.0], dtype=np.float32)
        cdf = logits_to_cdf(logits, total_freq=total_freq)
        rng = np.random.default_rng(123)
        symbols = rng.integers(0, 2, size=500).tolist()

        bit_out = BitOutputStream()
        enc = ArithmeticEncoder(bit_out)
        for s in symbols:
            enc.encode_symbol(s, cdf)
        data = enc.finish()

        dec = ArithmeticDecoder(BitInputStream(data))
        decoded = [dec.decode_symbol(cdf) for _ in range(len(symbols))]
        self.assertEqual(decoded, symbols)

    def test_empty_and_single_symbol(self) -> None:
        total_freq = 1 << 16
        cdf = logits_to_cdf(np.zeros(10, dtype=np.float32), total_freq=total_freq)

        # Empty
        enc = ArithmeticEncoder(BitOutputStream())
        data = enc.finish()
        dec = ArithmeticDecoder(BitInputStream(data))
        self.assertEqual([], [])
        del dec

        # Single symbol at boundary 0 and last
        for target in (0, 9):
            enc = ArithmeticEncoder(BitOutputStream())
            enc.encode_symbol(target, cdf)
            data = enc.finish()
            dec = ArithmeticDecoder(BitInputStream(data))
            self.assertEqual(dec.decode_symbol(cdf), target)

    def test_cdf_speed_262k(self) -> None:
        vocab = 262_144
        logits = np.random.default_rng(7).normal(0.0, 3.0, size=vocab).astype(np.float32)
        t0 = time.perf_counter()
        cdf = logits_to_cdf(logits, total_freq=1 << DEFAULT_CDF_BITS)
        dt = time.perf_counter() - t0
        self.assertEqual(int(cdf[-1]), 1 << DEFAULT_CDF_BITS)
        self.assertLess(dt, 0.5)

    def test_sparse_cdf_exact_partition_and_roundtrip(self) -> None:
        vocab = 2000
        total_freq = 1 << 20
        logits = np.full(vocab, -30.0, dtype=np.float32)
        logits[[0, 5, 42, 999, 1999]] = [2.0, 4.0, 1.5, 3.0, 0.5]
        scdf = logits_to_sparse_cdf(logits, total_freq=total_freq, logit_delta=12.0, max_top_k=32)
        self.assertIsInstance(scdf, SparseCDF)
        assert isinstance(scdf, SparseCDF)

        # Every symbol 0..vocab-1 must form an exact contiguous partition of [0, total_freq)
        expected_cursor = 0
        for sym in range(vocab):
            lo, hi = scdf.interval(sym)
            self.assertEqual(lo, expected_cursor)
            self.assertGreater(hi, lo)
            found_sym, f_lo, f_hi = scdf.find_symbol(lo)
            self.assertEqual(found_sym, sym)
            self.assertEqual((f_lo, f_hi), (lo, hi))
            if hi - lo > 1:
                found_end, _, _ = scdf.find_symbol(hi - 1)
                self.assertEqual(found_end, sym)
            expected_cursor = hi
        self.assertEqual(expected_cursor, total_freq)

        # Roundtrip with mix of top symbols and tail symbols (including 0, 1, 1998, 1999)
        symbols = [0, 1, 5, 7, 42, 100, 999, 1500, 1998, 1999]
        enc = ArithmeticEncoder(BitOutputStream())
        for s in symbols:
            enc.encode_symbol(s, scdf)
        blob = enc.finish()

        dec = ArithmeticDecoder(BitInputStream(blob))
        decoded = [dec.decode_symbol(scdf) for _ in symbols]
        self.assertEqual(decoded, symbols)

    def test_sparse_cdf_262k_roundtrip_and_speed(self) -> None:
        rng = np.random.default_rng(99)
        vocab = 262_144
        total_freq = 1 << DEFAULT_CDF_BITS
        logits_steps = []
        symbols = []
        for step in range(40):
            logits = rng.normal(-25.0, 2.0, size=vocab).astype(np.float32)
            top = rng.choice(vocab, size=50, replace=False)
            logits[top] = rng.uniform(0.0, 8.0, size=50).astype(np.float32)
            logits_steps.append(logits)
            # Include occasional tail symbol too
            symbols.append(int(top[0]) if step % 7 != 0 else int(rng.integers(0, vocab)))

        t0 = time.perf_counter()
        enc = ArithmeticEncoder(BitOutputStream())
        for l, s in zip(logits_steps, symbols):
            scdf = logits_to_sparse_cdf(l, total_freq=total_freq)
            enc.encode_symbol(s, scdf)
        blob = enc.finish()
        dt = time.perf_counter() - t0
        self.assertLess(dt, 0.5)

        dec = ArithmeticDecoder(BitInputStream(blob))
        recovered = [
            dec.decode_symbol(logits_to_sparse_cdf(l, total_freq=total_freq))
            for l in logits_steps
        ]
        self.assertEqual(recovered, symbols)


if __name__ == "__main__":
    unittest.main()

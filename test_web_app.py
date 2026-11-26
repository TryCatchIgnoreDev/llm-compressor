"""End-to-end HTTP/SSE integration tests for the FastAPI web application."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

import uvicorn

import web_app


SMALL_MODEL_A = (
    Path.home()
    / ".lmstudio"
    / "models"
    / "lmstudio-community"
    / "gemma-4-E2B-it-GGUF"
    / "gemma-4-E2B-it-Q4_K_M.gguf"
)
SMALL_MODEL_B = (
    Path.home()
    / ".lmstudio"
    / "models"
    / "lmstudio-community"
    / "Meta-Llama-3.1-8B-Instruct-GGUF"
    / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _parse_sse(raw_bytes: bytes) -> list[dict]:
    events = []
    text = raw_bytes.decode("utf-8")
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class TestWebApp(unittest.TestCase):
    server: uvicorn.Server
    thread: threading.Thread
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        if not (SMALL_MODEL_A.is_file() and SMALL_MODEL_B.is_file()):
            raise unittest.SkipTest("Small local GGUF models not found for Web UI tests.")
        port = _find_free_port()
        cls.base_url = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(web_app.app, host="127.0.0.1", port=port, log_level="warning")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{cls.base_url}/api/system", timeout=1.0) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("Web server failed to start within timeout.")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "server"):
            cls.server.should_exit = True
        if hasattr(cls, "thread"):
            cls.thread.join(timeout=5.0)

    def test_index_and_system_endpoints(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/") as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("LLM Compression Lab", html)
            self.assertIn("Model A", html)
            self.assertIn("Model B", html)

        with urllib.request.urlopen(f"{self.base_url}/api/system") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("models", data)
            self.assertEqual(data["max_inline_bits"], 10000)

        scan_root = urllib.parse.quote(str(SMALL_MODEL_A.parent.parent))
        with urllib.request.urlopen(f"{self.base_url}/api/system?scan_dir={scan_root}") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["scan_mode"], "folder")
            self.assertGreaterEqual(len(data["models"]), 2)

    def test_compare_sse_download_and_single_decompress(self) -> None:
        sample_text = (
            "Web UI comparison test: Claude Shannon and 3Blue1Brown compression demo "
            "with Unicode characters café naïve 🚀! " * 3
        )
        payload = json.dumps(
            {
                "text": sample_text,
                "model_a_path": str(SMALL_MODEL_A),
                "model_b_path": str(SMALL_MODEL_B),
                "n_ctx": 512,
                "n_gpu_layers_a": -1,
                "n_gpu_layers_b": -1,
                "cdf_bits": 28,
                "streams": 4,
                "fast_cdf": True,
                "collect_trace": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/compare",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.assertEqual(resp.status, 200)
            events = _parse_sse(resp.read())

        event_types = [e["type"] for e in events]
        self.assertIn("status", event_types)
        self.assertIn("progress", event_types)
        self.assertIn("model_done", event_types)
        self.assertEqual(event_types[-1], "complete")

        cmp_data = events[-1]["comparison"]
        self.assertTrue(cmp_data["model_a"]["exact_match"])
        self.assertTrue(cmp_data["model_b"]["exact_match"])
        self.assertEqual(cmp_data["model_a"]["restored_text"], sample_text)
        self.assertEqual(cmp_data["model_b"]["restored_text"], sample_text)
        self.assertTrue(cmp_data["model_a"]["bitstring_available"])
        self.assertIsNotNone(cmp_data["model_a"]["bitstring"])
        self.assertIsNotNone(cmp_data["model_a"]["hexstring"])

        # Decode Model A directly from its inline hexstring via /api/single/decompress_hex
        hex_payload = json.dumps(
            {
                "hexstring": cmp_data["model_a"]["hexstring"],
                "model_path": str(SMALL_MODEL_A),
                "n_ctx": 512,
                "n_gpu_layers": -1,
            }
        ).encode("utf-8")
        hex_req = urllib.request.Request(
            f"{self.base_url}/api/single/decompress_hex",
            data=hex_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(hex_req, timeout=60) as resp:
            self.assertEqual(resp.status, 200)
            hex_dec_json = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(hex_dec_json["restored_text"], sample_text)

        # Decode Model A directly from its inline bitstring via /api/single/decompress_bits
        bits_payload = json.dumps(
            {
                "bitstring": cmp_data["model_a"]["bitstring"],
                "model_path": str(SMALL_MODEL_A),
                "n_ctx": 512,
                "n_gpu_layers": -1,
            }
        ).encode("utf-8")
        bits_req = urllib.request.Request(
            f"{self.base_url}/api/single/decompress_bits",
            data=bits_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(bits_req, timeout=60) as resp:
            self.assertEqual(resp.status, 200)
            bits_dec_json = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(bits_dec_json["restored_text"], sample_text)

        # Reject > 10,000-bit input on /api/single/decompress_bits
        over_limit_req = urllib.request.Request(
            f"{self.base_url}/api/single/decompress_bits",
            data=json.dumps(
                {
                    "bitstring": "0" * 10008,
                    "model_path": str(SMALL_MODEL_A),
                    "n_ctx": 512,
                    "n_gpu_layers": -1,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(over_limit_req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)

        # Verify _bitstring_payload_fields suppresses bitstring/hexstring when > 10,000 bits
        over_meta = web_app._bitstring_payload_fields(None, 1251)
        self.assertFalse(over_meta["bitstring_available"])
        self.assertIsNone(over_meta["bitstring"])
        self.assertIsNone(over_meta["hexstring"])
        self.assertEqual(over_meta["total_bits"], 10008)

        # Download generated .llmc binary artifact for Model A and decode via /api/single/decompress
        dl_url = f"{self.base_url}{cmp_data['model_a']['download_url']}"
        with urllib.request.urlopen(dl_url) as resp:
            self.assertEqual(resp.status, 200)
            llmc_bytes = resp.read()
            self.assertTrue(llmc_bytes.startswith(b"LLMC"))
            self.assertEqual(len(llmc_bytes), cmp_data["model_a"]["compression"]["compressed_bytes"])

        query = urllib.parse.urlencode(
            {
                "model_path": str(SMALL_MODEL_A),
                "n_ctx": 512,
                "n_gpu_layers": -1,
            }
        )
        dec_req = urllib.request.Request(
            f"{self.base_url}/api/single/decompress?{query}",
            data=llmc_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(dec_req, timeout=60) as resp:
            self.assertEqual(resp.status, 200)
            dec_json = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(dec_json["restored_text"], sample_text)

        # Verify separate Model A / Model B / combined token JSON downloads and /api/compare/last/json
        for url_key, expected_label in (("model_a_json_url", "A"), ("model_b_json_url", "B")):
            with urllib.request.urlopen(f"{self.base_url}{cmp_data[url_key]}") as resp:
                self.assertEqual(resp.status, 200)
                m_json = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(m_json["label"], expected_label)
                self.assertGreater(len(m_json["tokens"]), 0)
                self.assertIn("bits", m_json["tokens"][0])
                self.assertIn("emitted_bits", m_json["tokens"][0])
                self.assertIn("model_rank", m_json["tokens"][0])

        with urllib.request.urlopen(f"{self.base_url}/api/compare/last/json") as resp:
            self.assertEqual(resp.status, 200)
            last_all = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(last_all["original_text"], sample_text)
            self.assertEqual(last_all["model_a"]["label"], "A")
            self.assertEqual(last_all["model_b"]["label"], "B")

        with urllib.request.urlopen(f"{self.base_url}/api/compare/last/json?model=a") as resp:
            self.assertEqual(resp.status, 200)
            last_a = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(last_a["label"], "A")
            self.assertEqual(last_a["summary"]["num_tokens"], cmp_data["model_a"]["compression"]["num_tokens"])

        with urllib.request.urlopen(f"{self.base_url}/api/compare/last/json?model=b") as resp:
            self.assertEqual(resp.status, 200)
            last_b = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(last_b["label"], "B")
            self.assertEqual(last_b["summary"]["num_tokens"], cmp_data["model_b"]["compression"]["num_tokens"])

        # Verify compact sparse JSON URLs and ?detail=sparse endpoints
        for url_key, expected_label in (("model_a_sparse_json_url", "A"), ("model_b_sparse_json_url", "B")):
            with urllib.request.urlopen(f"{self.base_url}{cmp_data[url_key]}") as resp:
                self.assertEqual(resp.status, 200)
                s_json = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(s_json["detail"], "sparse")
                self.assertEqual(s_json["label"], expected_label)
                self.assertEqual(
                    set(s_json["tokens"][0].keys()),
                    {"piece", "prob", "bits", "emitted_bits", "rank"},
                )

        with urllib.request.urlopen(f"{self.base_url}/api/compare/last/json?detail=sparse") as resp:
            self.assertEqual(resp.status, 200)
            sparse_all = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(sparse_all["detail"], "sparse")
            self.assertNotIn("original_text", sparse_all)
            self.assertGreater(len(sparse_all["model_a"]["top_surprises"]), 0)

        with urllib.request.urlopen(f"{self.base_url}/api/compare/last/json?model=a&detail=sparse") as resp:
            self.assertEqual(resp.status, 200)
            sparse_a = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(sparse_a["detail"], "sparse")
            self.assertEqual(sparse_a["label"], "A")


if __name__ == "__main__":
    unittest.main()

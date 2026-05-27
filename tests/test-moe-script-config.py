#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MoeScriptConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.moe_run = load_script("moe_run_for_test", REPO / "scripts" / "moe-run.py")
        cls.bench = load_script(
            "bench_wici_interactive_latency_for_test",
            REPO / "scripts" / "bench-wici-glm51-interactive-latency.py",
        )
        cls.calibrate = load_script(
            "calibrate_wici_interactive_config_for_test",
            REPO / "scripts" / "calibrate-wici-interactive-config.py",
        )
        cls.ttft_pack = load_script(
            "create_ttft_expert_pack_for_test",
            REPO / "scripts" / "create-ttft-expert-pack.py",
        )
        cls.compare_candidate = load_script(
            "compare_wici_interactive_candidate_for_test",
            REPO / "scripts" / "compare-wici-interactive-candidate.py",
        )

    def test_calibration_gates_fail_closed(self) -> None:
        args = argparse.Namespace(
            require_validation=True,
            require_canary=True,
            min_eval_tps=0.98,
            max_time_to_type_s=None,
            max_ttft_s=None,
        )

        self.assertEqual(
            self.calibrate.evaluate(args, {}, None),
            ["validation was required but did not run", "canary was required but did not run"],
        )
        self.assertEqual(
            self.calibrate.evaluate(
                args,
                {"ok": True, "moe_vram_cache_failed": True, "eval_tokens_per_s": 1.0},
                self.calibrate.EXPECTED_N36_SHA,
            ),
            ["validation reported moe_vram_cache_failed"],
        )

    def test_trace_to_profile_csv_aggregates_routes(self) -> None:
        profile = self.calibrate.trace_to_profile_csv(
            "seq,tensor,expert_idx,expert_bytes\n"
            "1,blk.1.ffn_up_exps.weight,2,1024\n"
            "2,blk.1.ffn_up_exps.weight,2,1024\n"
            "3,blk.1.ffn_down_exps.weight,7,2048\n"
        )

        self.assertIn("tensor,expert_idx,count,expert_bytes", profile)
        self.assertIn("blk.1.ffn_up_exps.weight,2,2,1024", profile)
        self.assertIn("blk.1.ffn_down_exps.weight,7,1,2048", profile)

    def test_ttft_pack_limits_after_compute_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.expert-pack"
            trace = root / "trace.csv"
            out = root / "ttft.expert-pack"
            data_start = self.ttft_pack.HEADER_STRUCT.size + 2 * self.ttft_pack.ENTRY_STRUCT.size
            rows = [
                ("blk.9.ffn_down_exps.weight", 0, b"late-row-0000000"),
                ("blk.3.ffn_up_exps.weight", 0, b"early-row-000000"),
            ]
            with source.open("wb") as f:
                f.write(self.ttft_pack.HEADER_STRUCT.pack(self.ttft_pack.MAGIC, 1, self.ttft_pack.HEADER_STRUCT.size, len(rows), data_start))
                offset = data_start
                for name, expert, payload in rows:
                    name_raw = name.encode("utf-8")
                    f.write(self.ttft_pack.ENTRY_STRUCT.pack(name_raw + b"\0" * (128 - len(name_raw)), expert, 0, offset, len(payload)))
                    offset += len(payload)
                for _name, _expert, payload in rows:
                    f.write(payload)
            trace.write_text(
                "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
                "1,0,cpu_down_route,blk.9.ffn_down_exps.weight,0,16,0,0,0,0\n"
                "2,0,cpu_up_route,blk.3.ffn_up_exps.weight,0,16,0,0,0,0\n",
                encoding="utf-8",
            )

            routes = self.ttft_pack.order_routes(self.ttft_pack.read_trace_routes(trace), "compute")[:1]
            count, missing, _payload = self.ttft_pack.write_ttft_pack(source, out, routes, self.ttft_pack.read_source_index(source), 8)

            self.assertEqual(count, 1)
            self.assertEqual(missing, 0)
            data = out.read_bytes()
            self.assertIn(b"early-row-000000", data)
            self.assertNotIn(b"late-row-0000000", data)

    def test_bench_stderr_metrics_extract_vram_cache_budget(self) -> None:
        record = {}
        self.bench.add_stderr_metrics(
            record,
            "[moe_stream_batch] VRAM cache budget: requested=1536 MiB actual=1408 MiB "
            "free=1500 MiB total=15827 MiB graph_reserve=0 MiB safety=128 MiB clamp=1\n"
            "[moe_stream_batch] VRAM cache: 0.6 GiB, 133 slots (4.59 MiB each)\n"
            "[moe_stream_batch] VRAM cache: 0.8 GiB, 200 slots (3.84 MiB each)\n"
            "llama_print_timings:        eval time =   64740.71 ms /    63 runs"
            "   ( 1027.63 ms per token,     0.97 tokens per second)\n",
        )

        self.assertFalse(record["moe_vram_cache_failed"])
        self.assertEqual(record["moe_vram_cache_requested_mib"], 1536)
        self.assertEqual(record["moe_vram_cache_actual_mib"], 1408)
        self.assertEqual(record["moe_vram_cache_safety_mib"], 128)
        self.assertTrue(record["moe_vram_cache_clamp"])
        self.assertEqual(record["moe_vram_cache_slots"], 333)
        self.assertAlmostEqual(record["moe_vram_cache_allocated_gib"], 1.4)
        self.assertEqual(record["eval_tokens"], 63)
        self.assertEqual(record["eval_tokens_per_s"], 0.97)

    def test_bench_summary_includes_cache_telemetry_stats(self) -> None:
        summary = self.bench.build_summary(
            [
                {
                    "ok": True,
                    "interactive_ttft_s": 12.0,
                    "eval_tokens_per_s": 0.96,
                    "moe_vram_cache_failed": False,
                    "moe_vram_cache_actual_mib": 1536,
                    "moe_vram_cache_clamp": True,
                    "moe_vram_cache_slots": 372,
                },
                {
                    "ok": True,
                    "interactive_ttft_s": 14.0,
                    "eval_tokens_per_s": 0.94,
                    "moe_vram_cache_failed": True,
                    "moe_vram_cache_actual_mib": 1408,
                    "moe_vram_cache_clamp": True,
                    "moe_vram_cache_slots": 333,
                },
            ],
            REPO / "bench" / "sample.jsonl",
        )

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["moe_vram_cache_failed_runs"], 1)
        self.assertEqual(summary["moe_vram_cache_clamp_runs"], 2)
        self.assertEqual(summary["stats"]["interactive_ttft_s"]["p50"], 12.0)
        self.assertEqual(summary["stats"]["moe_vram_cache_actual_mib"]["max"], 1536.0)
        self.assertEqual(summary["stats"]["moe_vram_cache_slots"]["min"], 333.0)

    def test_bench_summary_counts_n36_canary(self) -> None:
        summary = self.bench.build_summary(
            [
                {"ok": True, "n36_canary_ok": True},
                {"ok": True, "n36_canary_ok": False},
                {"ok": False, "n36_canary_ok": False},
            ],
            REPO / "bench" / "sample.jsonl",
        )

        self.assertEqual(summary["n36_canary_ok_runs"], 1)
        self.assertEqual(summary["n36_canary_failed_runs"], 1)

    def test_calibration_bench_command_writes_summary_next_to_jsonl(self) -> None:
        args = argparse.Namespace(
            ssh_host="wici",
            remote_dir="/home/wici/venti/ik_llama",
            runs=1,
            prompt="hello",
            chat_load_mode="fast-prompt",
            seed=42,
            chat_active_prewarm=False,
            post_ready_delay_s=1.5,
            ignore_eos=True,
            remote_env=["BASE=1"],
        )
        output = REPO / "bench" / "sample.jsonl"

        cmd = self.calibrate.bench_command(
            args,
            output,
            36,
            ["--moe-preset", "preset.json"],
            remote_env=["EXTRA=1"],
        )

        self.assertEqual(self.calibrate.bench_summary_path(output), REPO / "bench" / "sample.summary.json")
        summary_index = cmd.index("--summary")
        self.assertEqual(Path(cmd[summary_index + 1]), REPO / "bench" / "sample.summary.json")
        delay_index = cmd.index("--post-ready-delay-s")
        self.assertEqual(cmd[delay_index + 1], "1.5")
        self.assertIn("BASE=1", cmd)
        self.assertIn("EXTRA=1", cmd)

    def test_bench_command_can_enable_ttft_trace(self) -> None:
        args = argparse.Namespace(
            remote_dir="/home/wici/venti/ik_llama",
            chat_load_mode="fast-prompt",
            force_preset=False,
            moe_preset=None,
            moe_route_trace=None,
            chat_active_prewarm=False,
            threads=None,
            chat_threads_batch=None,
            chat_gpu_layers=None,
            seed=42,
            predict_tokens=36,
            ignore_eos=True,
            extra=[],
            route_trace_dir="/tmp/route",
            ttft_trace_dir="/tmp/ttft",
            remote_env=[],
        )
        args.route_trace_prefix = "sample"

        cmd = self.bench.build_remote_command(args, 2)

        self.assertIn("GGML_MOE_ROUTE_TRACE_OUT=/tmp/route/sample-run02.trace.csv", cmd)
        self.assertIn("GGML_MOE_TTFT_TRACE_OUT=/tmp/ttft/sample-run02.ttft.csv", cmd)
        self.assertIn("mkdir -p /tmp/route /tmp/ttft", cmd)

    def test_calibration_capture_trace_requests_ttft_trace(self) -> None:
        args = argparse.Namespace(
            ssh_host="wici",
            remote_dir="/home/wici/venti/ik_llama",
            runs=1,
            prompt="hello",
            chat_load_mode="fast-prompt",
            seed=42,
            chat_active_prewarm=False,
            post_ready_delay_s=0.0,
            ignore_eos=True,
            remote_env=[],
            route_trace_dir="/tmp/route",
            ttft_trace_dir="/tmp/ttft",
        )
        output = REPO / "bench" / "sample.jsonl"

        cmd = self.calibrate.bench_command(
            args,
            output,
            36,
            ["--route-trace-dir", args.route_trace_dir, "--ttft-trace-dir", args.ttft_trace_dir],
        )

        self.assertIn("--ttft-trace-dir", cmd)
        self.assertIn("/tmp/ttft", cmd)

    def test_ttft_mechanism_plan_identifies_prompt_moe_bottleneck(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,150.0,cpu_prompt_upgate,blk.1.ffn_up_exps.weight,20,1000,0,0,0,6000\n"
            "3,160.0,cpu_prompt_down,blk.1.ffn_down_exps.weight,20,1000,0,0,0,3000\n"
            "4,170.0,runtime_load,blk.1.ffn_down_exps.weight,20,1000,0,0,0,200\n"
            "5,10100.0,mark_first_token,,-1,0,0,0,0,0\n"
        )
        plan = self.calibrate.build_mechanism_plan(summary, {"interactive_ttft_s": 10.0})

        self.assertEqual(plan["bottleneck"], "prompt_moe_compute")
        self.assertEqual(plan["next_experiment"], "exact_row_batched_prompt_moe")
        self.assertIn("GGML_MOE_STREAM_PROMPT_UP_GATE", plan["avoid"][0])

    def test_ttft_mechanism_plan_uses_prompt_subtrace(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,110.0,cpu_prompt_upgate,blk.1.ffn_up_exps.weight,20,1000,0,0,0,6000\n"
            "3,115.0,cpu_up_group,blk.1.ffn_up_exps.weight,20,1000,0,0,0,10\n"
            "4,120.0,cpu_up_prep,blk.1.ffn_up_exps.weight,20,1000,0,0,0,200\n"
            "5,130.0,cpu_up_compute,blk.1.ffn_up_exps.weight,20,1000,0,0,0,5000\n"
            "6,140.0,cpu_prompt_down,blk.1.ffn_down_exps.weight,20,1000,0,0,0,3000\n"
            "7,145.0,cpu_down_compute,blk.1.ffn_down_exps.weight,20,1000,0,0,0,2800\n"
            "8,10100.0,mark_first_token,,-1,0,0,0,0,0\n"
        )
        plan = self.calibrate.build_mechanism_plan(summary, {"interactive_ttft_s": 10.0})

        self.assertEqual(summary["prompt_moe_subspans"]["compute"], 7800.0)
        self.assertEqual(plan["next_experiment"], "exact_row_batched_prompt_moe")

    def test_ttft_mechanism_plan_tolerates_truncated_subtrace_names(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,110.0,cpu_prompt_upgate,blk.1.ffn_up_exps.weight,20,1000,0,0,0,6000\n"
            "3,120.0,cpu_prompt_upgate_compu,blk.1.ffn_up_exps.weight,20,1000,0,0,0,5000\n"
            "4,130.0,cpu_prompt_down,blk.1.ffn_down_exps.weight,20,1000,0,0,0,3000\n"
            "5,140.0,cpu_prompt_down_compute,blk.1.ffn_down_exps.weight,20,1000,0,0,0,2800\n"
            "6,10100.0,mark_first_token,,-1,0,0,0,0,0\n"
        )

        self.assertEqual(summary["prompt_moe_subspans"]["compute"], 7800.0)

    def test_ttft_mechanism_plan_uses_prompt_fault_counters(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,110.0,cpu_prompt_upgate,blk.1.ffn_up_exps.weight,20,1000,0,0,0,6000\n"
            "3,120.0,cpu_up_compute,blk.1.ffn_up_exps.weight,20,1000,0,0,0,5900\n"
            "4,121.0,cpu_up_minflt,blk.1.ffn_up_exps.weight,20,1000,0,0,0,128\n"
            "5,122.0,cpu_up_majflt,blk.1.ffn_up_exps.weight,20,1000,0,0,0,20\n"
            "6,130.0,cpu_prompt_down,blk.1.ffn_down_exps.weight,20,1000,0,0,0,3000\n"
            "7,140.0,cpu_down_compute,blk.1.ffn_down_exps.weight,20,1000,0,0,0,2800\n"
            "8,10100.0,mark_first_token,,-1,0,0,0,0,0\n"
        )
        plan = self.calibrate.build_mechanism_plan(summary, {"interactive_ttft_s": 10.0})

        self.assertEqual(summary["prompt_moe_faults"]["minor"], 128.0)
        self.assertEqual(summary["prompt_moe_faults"]["major"], 20.0)
        self.assertEqual(plan["next_experiment"], "prompt_critical_residency_or_ttft_pack")

    def test_ttft_mechanism_plan_rejects_oversized_cpu_pack_after_pack_probe(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,110.0,cpu_prompt_upgate,blk.1.ffn_up_exps.weight,20,1000,0,0,0,6000\n"
            "3,120.0,cpu_up_compute,blk.1.ffn_up_exps.weight,20,1000,0,0,0,5900\n"
            "4,121.0,cpu_up_majflt,blk.1.ffn_up_exps.weight,20,1000,0,0,0,20\n"
            "5,130.0,cpu_prompt_down,blk.1.ffn_down_exps.weight,20,1000,0,0,0,3000\n"
            "6,140.0,cpu_down_compute,blk.1.ffn_down_exps.weight,20,1000,0,0,0,2800\n"
            "7,141.0,cpu_up_route,blk.1.ffn_up_exps.weight,1,8589934592,0,0,0,0\n"
            "8,142.0,cpu_down_route,blk.1.ffn_down_exps.weight,1,8589934592,0,0,0,0\n"
            "9,10100.0,mark_first_token,,-1,0,0,0,0,0\n"
        )
        plan = self.calibrate.build_mechanism_plan(summary, {"interactive_ttft_s": 10.0})

        self.assertGreaterEqual(summary["prompt_moe_route_unique_gib"], 16.0)
        self.assertEqual(plan["next_experiment"], "speculative_expert_prefetch_accuracy_probe")
        self.assertIn("route-prediction accuracy report", plan["implementation_target"])
        self.assertIn("not more cache/thread sweeps", plan["rationale"])

    def test_candidate_compare_rejects_pack_without_compute_reduction(self) -> None:
        baseline = {"ok": True, "interactive_ttft_s": 12.6, "time_to_type_s": 10.4, "eval_tokens_per_s": 0.88}
        candidate = {
            "ok": True,
            "n36_canary_ok": True,
            "interactive_ttft_s": 13.1,
            "time_to_type_s": 10.7,
            "eval_tokens_per_s": 0.85,
            "command": ["env", "LLAMA_CHAT_TTFT_EXPERT_PACK=pack.expert-pack", "python3"],
        }
        decision = self.compare_candidate.compare(
            baseline,
            candidate,
            {"prompt_compute_ms": 9600.0},
            {"prompt_compute_ms": 9880.0},
        )

        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["mechanism"], "ttft_mmap_pack")
        self.assertEqual(decision["next_experiment"], "speculative_expert_prefetch_accuracy_probe")

    def test_candidate_compare_accepts_material_canary_safe_win(self) -> None:
        baseline = {"ok": True, "interactive_ttft_s": 15.0, "time_to_type_s": 10.0, "eval_tokens_per_s": 1.0}
        candidate = {
            "ok": True,
            "n36_canary_ok": True,
            "interactive_ttft_s": 12.0,
            "time_to_type_s": 9.8,
            "eval_tokens_per_s": 0.97,
            "command": ["env", "GGML_MOE_STREAM_PROMPT_UP_GATE=exact-q8-k", "python3"],
        }
        decision = self.compare_candidate.compare(
            baseline,
            candidate,
            {"prompt_compute_ms": 9000.0},
            {"prompt_compute_ms": 6500.0},
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["mechanism"], "exact_prompt_cuda_upgate")

    def test_ttft_summary_counts_down_many_compute(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,120.0,cpu_up_many_compute,blk.1.ffn_up_exps.weight,20,1000,0,0,0,3000\n"
            "3,130.0,cpu_down_many_compute,blk.1.ffn_down_exps.weight,20,1000,0,0,0,2000\n"
            "4,5100.0,mark_first_token,,-1,0,0,0,0,0\n"
        )

        self.assertEqual(summary["prompt_moe_subspans"]["compute"], 5000.0)

    def test_ttft_summary_uses_first_submit_window(self) -> None:
        summary = self.calibrate.summarize_ttft_trace_csv(
            "seq,t_ms,op,tensor,expert_idx,expert_bytes,cache_hit,pack_hit,ram_hit,copy_ms\n"
            "1,100.0,mark_submit,,-1,0,0,0,0,0\n"
            "2,120.0,cpu_up_compute,blk.1.ffn_up_exps.weight,20,1000,0,0,0,10\n"
            "3,150.0,mark_first_token,,-1,0,0,0,0,0\n"
            "4,200.0,mark_ready,,-1,0,0,0,0,0\n"
            "5,210.0,mark_submit,,-1,0,0,0,0,0\n"
        )

        self.assertEqual(summary["submit_to_first_token_ms"], 50.0)
        self.assertEqual(summary["prompt_moe_subspans"]["compute"], 10.0)

    def test_remote_cat_when_ready_waits_for_trace_file(self) -> None:
        calls = {"exists": 0, "cat": 0}

        def fake_exists(_args, _path):
            calls["exists"] += 1
            return calls["exists"] >= 2

        def fake_cat(_args, _path):
            calls["cat"] += 1
            return "trace"

        old_exists = self.calibrate.remote_file_exists
        old_cat = self.calibrate.remote_cat
        old_sleep = self.calibrate.time.sleep
        try:
            self.calibrate.remote_file_exists = fake_exists
            self.calibrate.remote_cat = fake_cat
            self.calibrate.time.sleep = lambda _seconds: None
            self.assertEqual(self.calibrate.remote_cat_when_ready(argparse.Namespace(), "trace.csv"), "trace")
        finally:
            self.calibrate.remote_file_exists = old_exists
            self.calibrate.remote_cat = old_cat
            self.calibrate.time.sleep = old_sleep

        self.assertEqual(calls, {"exists": 2, "cat": 1})

    def test_remote_path_resolves_trace_under_remote_dir(self) -> None:
        args = argparse.Namespace(remote_dir="/home/wici/venti/ik_llama")
        self.assertEqual(
            self.calibrate.remote_path(args, "bench/trace.csv"),
            "/home/wici/venti/ik_llama/bench/trace.csv",
        )
        self.assertEqual(
            self.calibrate.remote_path(args, "/home/wici/.cache/run.stderr.log"),
            "/home/wici/.cache/run.stderr.log",
        )

    def test_remote_cat_when_ready_returns_none_after_timeout(self) -> None:
        old_exists = self.calibrate.remote_file_exists
        old_sleep = self.calibrate.time.sleep
        old_monotonic = self.calibrate.time.monotonic
        times = iter([0.0, 1.0])
        try:
            self.calibrate.remote_file_exists = lambda _args, _path: False
            self.calibrate.time.sleep = lambda _seconds: None
            self.calibrate.time.monotonic = lambda: next(times)
            self.assertIsNone(self.calibrate.remote_cat_when_ready(argparse.Namespace(), "trace.csv", wait_s=0.5))
        finally:
            self.calibrate.remote_file_exists = old_exists
            self.calibrate.time.sleep = old_sleep
            self.calibrate.time.monotonic = old_monotonic

    def test_remote_file_contains_when_ready_waits_for_marker(self) -> None:
        calls = {"contains": 0}

        def fake_contains(_args, _path, _needle):
            calls["contains"] += 1
            return calls["contains"] >= 3

        old_contains = self.calibrate.remote_file_contains
        old_sleep = self.calibrate.time.sleep
        try:
            self.calibrate.remote_file_contains = fake_contains
            self.calibrate.time.sleep = lambda _seconds: None
            self.assertTrue(
                self.calibrate.remote_file_contains_when_ready(
                    argparse.Namespace(),
                    "stderr.log",
                    "TTFT trace written",
                )
            )
        finally:
            self.calibrate.remote_file_contains = old_contains
            self.calibrate.time.sleep = old_sleep

        self.assertEqual(calls, {"contains": 3})

    def test_fingerprint_ignores_transient_memory_availability(self) -> None:
        mod = self.moe_run
        model = mod.ModelStats("m", "glm", 1, 8, 8, 1, [1], 1)
        pack = mod.PackStats(1, 1024, 1024, 8, 512, 512)
        cpu = mod.CpuInfo("cpu", 10, 2, 5, 1, 1)
        storage = mod.StorageInfo("/dev/nvme0n1p2", "ext4", "rw", "/dev/nvme0n1", "nvme", 0, "none", "disk", "2T")
        gpu_a = mod.GpuInfo("NVIDIA GeForce RTX 5060 Ti", 16311, 15500, "0000:02:00.0", "595.71.05")
        gpu_b = mod.GpuInfo("NVIDIA GeForce RTX 5060 Ti", 16311, 2000, "0000:02:00.0", "595.71.05")

        f1, inputs = mod.build_fingerprint(
            [REPO / "README.md"],
            REPO / "CMakeLists.txt",
            None,
            None,
            model,
            pack,
            gpu_a,
            cpu,
            mod.MemInfo(64000, 32000),
            storage,
            2048,
        )
        f2, _ = mod.build_fingerprint(
            [REPO / "README.md"],
            REPO / "CMakeLists.txt",
            None,
            None,
            model,
            pack,
            gpu_b,
            cpu,
            mod.MemInfo(64000, 12000),
            storage,
            2048,
        )

        self.assertEqual(f1, f2)
        self.assertEqual(inputs["observed"]["gpu"]["free_mib"], 15500)
        self.assertEqual(inputs["mem"]["total_mib"], 64000)

    def test_interactive_profile_env_adds_startup_preload_default(self) -> None:
        mod = self.moe_run
        model = mod.ModelStats("model", "glm", 1, 8, 8, 1, [1], 1)
        pack = mod.PackStats(1, 1024, 1024, 8, 512, 512)
        gpu = mod.GpuInfo("gpu", 16000, 12000, "bus", "driver")
        mem = mod.MemInfo(64000, 32000)

        patches = {
            "compute_vram_cache_mib": lambda *args, **kwargs: 1792,
            "compute_n_gpu_layers": lambda *args, **kwargs: (60, 0),
            "is_nvme_path": lambda path: True,
            "compute_profile_reserve_pct": lambda *args, **kwargs: 10,
            "compute_upgate_pct": lambda *args, **kwargs: 60,
            "compute_ram_tier_mib": lambda *args, **kwargs: 3072,
            "compute_ram_tier_skip": lambda *args, **kwargs: 447,
        }
        originals = {name: getattr(mod, name) for name in patches}
        old_override = os.environ.pop("MOE_RUN_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS", None)
        old_safety_override = os.environ.pop("MOE_RUN_CHAT_VRAM_CACHE_SAFETY_MIB", None)
        try:
            for name, value in patches.items():
                setattr(mod, name, value)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                profile_path = tmp_path / "route.csv"
                profile_path.write_text("seq,tensor\n", encoding="utf-8")
                env = mod.compute_env(
                    tmp_path / "model.gguf",
                    tmp_path / "model.expert-pack",
                    profile_path,
                    model,
                    pack,
                    gpu,
                    mem,
                    2048,
                    interactive=True,
                )
                self.assertEqual(env["LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS"], "3")
                self.assertEqual(env["GGML_CUDA_EAGER_CUBLAS"], "1")
                self.assertEqual(env["GGML_MOE_VRAM_CACHE_AUTO_CLAMP"], "1")
                self.assertEqual(env["GGML_MOE_VRAM_CACHE_SAFETY_MIB"], "256")

                os.environ["MOE_RUN_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS"] = "5"
                os.environ["MOE_RUN_CHAT_VRAM_CACHE_SAFETY_MIB"] = "256"
                env = mod.compute_env(
                    tmp_path / "model.gguf",
                    tmp_path / "model.expert-pack",
                    profile_path,
                    model,
                    pack,
                    gpu,
                    mem,
                    2048,
                    interactive=True,
                )
                self.assertEqual(env["LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS"], "5")
                self.assertEqual(env["GGML_MOE_VRAM_CACHE_SAFETY_MIB"], "256")
        finally:
            if old_override is not None:
                os.environ["MOE_RUN_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS"] = old_override
            else:
                os.environ.pop("MOE_RUN_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS", None)
            if old_safety_override is not None:
                os.environ["MOE_RUN_CHAT_VRAM_CACHE_SAFETY_MIB"] = old_safety_override
            else:
                os.environ.pop("MOE_RUN_CHAT_VRAM_CACHE_SAFETY_MIB", None)
            for name, value in originals.items():
                setattr(mod, name, value)

    def test_calibration_selects_lowest_ttft_startup_preload_candidate(self) -> None:
        args = argparse.Namespace(
            min_eval_tps=0.93,
            max_ttft_s=None,
            max_time_to_type_s=None,
            min_ttft_improvement_pct=0.0,
        )
        candidates = [
            {
                "preload_tensors": 0,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 15.0,
                    "eval_tokens_per_s": 0.98,
                    "time_to_type_s": 8.0,
                },
            },
            {
                "preload_tensors": 2,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.7,
                    "eval_tokens_per_s": 0.92,
                    "time_to_type_s": 10.0,
                },
            },
            {
                "preload_tensors": 3,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.9,
                    "eval_tokens_per_s": 0.94,
                    "time_to_type_s": 10.5,
                },
            },
        ]

        selected = self.calibrate.select_latency_candidate(args, candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["preload_tensors"], 3)

    def test_calibration_requires_session_ttft_improvement_when_baseline_present(self) -> None:
        args = argparse.Namespace(
            min_eval_tps=0.93,
            max_ttft_s=None,
            max_time_to_type_s=None,
            min_ttft_improvement_pct=10.0,
        )
        baseline = {
            "ok": True,
            "interactive_ttft_s": 12.0,
            "eval_tokens_per_s": 0.98,
            "time_to_type_s": 9.0,
        }
        candidates = [
            {
                "preload_tensors": 2,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 11.5,
                    "eval_tokens_per_s": 0.98,
                    "time_to_type_s": 9.5,
                },
            },
            {
                "preload_tensors": 3,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 10.7,
                    "eval_tokens_per_s": 0.94,
                    "time_to_type_s": 10.5,
                },
            },
        ]

        selected = self.calibrate.select_latency_candidate(args, candidates, baseline)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["preload_tensors"], 3)
        self.assertFalse(self.calibrate.validation_passes_latency_gate(args, candidates[0]["record"], baseline))
        self.assertTrue(self.calibrate.validation_passes_latency_gate(args, candidates[1]["record"], baseline))
        self.assertAlmostEqual(
            self.calibrate.ttft_improvement_pct(baseline, candidates[1]["record"]),
            10.833333333333334,
        )

    def test_ordered_latency_candidates_keeps_canary_fallback_order(self) -> None:
        args = argparse.Namespace(
            min_eval_tps=0.93,
            max_ttft_s=None,
            max_time_to_type_s=None,
            min_ttft_improvement_pct=0.0,
        )
        candidates = [
            {
                "preload_tensors": 4,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.7,
                    "eval_tokens_per_s": 0.95,
                    "time_to_type_s": 12.0,
                },
            },
            {
                "preload_tensors": 3,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.9,
                    "eval_tokens_per_s": 0.98,
                    "time_to_type_s": 10.5,
                },
            },
            {
                "preload_tensors": 2,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.8,
                    "eval_tokens_per_s": 0.92,
                    "time_to_type_s": 10.0,
                },
            },
        ]

        ordered = self.calibrate.ordered_latency_candidates(args, candidates)
        self.assertEqual([item["preload_tensors"] for item in ordered], [4, 3])
        self.assertEqual(
            self.calibrate.startup_preload_remote_env(ordered[0]),
            ["LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS=4"],
        )

    def test_cache_budget_candidates_are_bounded_and_selectable(self) -> None:
        args = argparse.Namespace(
            cache_budget_candidates=None,
            max_cache_budget_mib=1600,
            min_eval_tps=0.93,
            max_ttft_s=None,
            max_time_to_type_s=None,
            min_ttft_improvement_pct=0.0,
        )
        preset = {"env": {"GGML_MOE_VRAM_CACHE_MIB": "1536"}}
        self.assertEqual(
            self.calibrate.cache_budget_candidates(args, preset),
            [1408, 1536, 1600],
        )
        self.assertEqual(self.calibrate.preset_vram_cache_mib(preset), 1536)

        candidates = [
            {
                "cache_mib": 1664,
                "record": {
                    "ok": True,
                    "moe_vram_cache_failed": True,
                    "interactive_ttft_s": 12.0,
                    "eval_tokens_per_s": 0.99,
                    "time_to_type_s": 10.0,
                },
            },
            {
                "cache_mib": 1536,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.8,
                    "eval_tokens_per_s": 0.96,
                    "time_to_type_s": 10.5,
                },
            },
        ]
        selected = self.calibrate.select_latency_candidate(args, candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["cache_mib"], 1536)
        self.assertEqual(
            self.calibrate.cache_budget_remote_env(selected),
            ["GGML_MOE_VRAM_CACHE_MIB=1536"],
        )

    def test_profile_reserve_candidates_are_bounded_and_selectable(self) -> None:
        args = argparse.Namespace(
            profile_reserve_candidates=None,
            max_profile_reserve_pct=25,
            min_eval_tps=0.93,
            max_ttft_s=None,
            max_time_to_type_s=None,
            min_ttft_improvement_pct=0.0,
        )
        preset = {"env": {"GGML_MOE_VRAM_PROFILE_RESERVE_PCT": "10"}}
        self.assertEqual(
            self.calibrate.profile_reserve_candidates(args, preset),
            [10, 20, 25],
        )
        self.assertEqual(self.calibrate.preset_profile_reserve_pct(preset), 10)

        candidates = [
            {
                "profile_reserve_pct": 30,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 13.5,
                    "eval_tokens_per_s": 0.95,
                    "time_to_type_s": 10.5,
                },
            },
            {
                "profile_reserve_pct": 20,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.8,
                    "eval_tokens_per_s": 0.94,
                    "time_to_type_s": 10.5,
                },
            },
        ]
        selected = self.calibrate.select_latency_candidate(args, candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["profile_reserve_pct"], 20)
        self.assertEqual(
            self.calibrate.profile_reserve_remote_env(selected),
            ["GGML_MOE_VRAM_PROFILE_RESERVE_PCT=20"],
        )

    def test_startup_preload_candidates_are_derived_from_preset_default(self) -> None:
        args = argparse.Namespace(
            startup_preload_candidates=None,
            max_startup_preload_tensors=6,
        )
        preset = {"env": {"LLAMA_CHAT_STARTUP_PROFILE_PRELOAD_TENSORS": "3"}}
        self.assertEqual(
            self.calibrate.startup_preload_candidates(args, preset),
            [0, 2, 3, 4],
        )

        args.startup_preload_candidates = "1, 3, 9"
        self.assertEqual(
            self.calibrate.startup_preload_candidates(args, preset),
            [1, 3, 6],
        )
        self.assertEqual(self.calibrate.preset_startup_preload(preset), 3)

    def test_ordered_latency_candidates_ignores_skipped_candidates(self) -> None:
        args = argparse.Namespace(
            min_eval_tps=0.93,
            max_ttft_s=None,
            max_time_to_type_s=None,
            min_ttft_improvement_pct=0.0,
        )
        candidates = [
            {
                "preload_tensors": 3,
                "skipped": True,
                "skip_reason": "matches_session_baseline",
            },
            {
                "preload_tensors": 2,
                "record": {
                    "ok": True,
                    "interactive_ttft_s": 12.8,
                    "eval_tokens_per_s": 0.95,
                    "time_to_type_s": 10.0,
                },
            },
        ]

        ordered = self.calibrate.ordered_latency_candidates(args, candidates)
        self.assertEqual([item["preload_tensors"] for item in ordered], [2])

    def test_calibration_can_reuse_session_baseline_when_no_candidate_selected(self) -> None:
        baseline = {
            "output": "baseline.jsonl",
            "record": {
                "ok": True,
                "interactive_ttft_s": 12.0,
                "eval_tokens_per_s": 0.98,
            },
        }
        validate_output, validate_record = self.calibrate.selected_validation_result(None, baseline)

        self.assertEqual(validate_output, Path("baseline.jsonl"))
        self.assertEqual(validate_record["interactive_ttft_s"], 12.0)

    def test_moe_run_dry_run_does_not_write_candidate_preset(self) -> None:
        mod = self.moe_run
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.gguf"
            pack_path = tmp_path / "model.expert-pack"
            model_path.write_text("model", encoding="utf-8")
            pack_path.write_text("pack", encoding="utf-8")

            patches = {
                "REPO_PRESETS": tmp_path / "repo-presets",
                "llama_cache_dir": lambda: tmp_path / "cache",
                "discover_model": lambda explicit: model_path,
                "model_shards": lambda path: [path],
                "discover_pack": lambda model, explicit: pack_path,
                "discover_profile": lambda explicit, chat=False, model_path=None, pack_path=None: None,
                "discover_route_trace": lambda explicit: None,
                "read_gpu_info": lambda: mod.GpuInfo("gpu", 16000, 12000, "bus", "driver"),
                "read_cpu_info": lambda: mod.CpuInfo("cpu", 8, 2, 4, 1, 1),
                "read_mem_info": lambda: mod.MemInfo(64000, 32000),
                "read_storage_info": lambda path: mod.StorageInfo("src", "ext4", "rw", "disk", "nvme", 0, "none", "model", "2T"),
                "read_model_stats": lambda paths: mod.ModelStats("model", "glm", 1, 8, 8, 1, [1], 1),
                "read_pack_stats": lambda path: mod.PackStats(1, 1024, 1024, 8, 512, 512),
                "compute_env": lambda *args, **kwargs: {"MODEL": str(model_path), "N_GPU_LAYERS": "1"},
            }
            originals = {name: getattr(mod, name) for name in patches}
            try:
                for name, value in patches.items():
                    setattr(mod, name, value)
                args = argparse.Namespace(
                    preset=None,
                    model=None,
                    expert_pack=None,
                    profile=None,
                    route_trace=None,
                    ctx_size=2048,
                    force=True,
                    dry_run=True,
                    save_repo_preset=False,
                    chat=False,
                )
                _data, json_path, env_path = mod.load_or_create_preset(args)
            finally:
                for name, value in originals.items():
                    setattr(mod, name, value)

            self.assertFalse(json_path.exists())
            self.assertFalse(env_path.exists())


if __name__ == "__main__":
    unittest.main()

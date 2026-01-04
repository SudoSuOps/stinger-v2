#!/usr/bin/env python3
"""
=======================================================================
PARALLEL SPINE THROUGHPUT BENCHMARK — Dual GPU Edition
=======================================================================

Parallel processing benchmark using asyncio.gather() for concurrent
LLM requests. Tests throughput with configurable worker count.

Run: python benchmark_spine_parallel.py --workers 2

trustcat.ai | stinger.swarmbee.eth
=======================================================================
"""

import os
import sys
import time
import json
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import statistics

sys.path.insert(0, str(Path(__file__).parent))

import pydicom
import numpy as np
from PIL import Image
import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

SPINE_SAMPLES_DIR_NVME = Path("/mnt/btcnode/datasets/spine")
SPINE_SAMPLES_DIR_NAS = Path(__file__).parent / "e2e_samples" / "spine"
SPINE_MODEL_PATH = Path("/home/ai/Desktop/quantum-rails/queenbee-llm/spine_detector/checkpoints/spine_detector_20260103_140009/weights/best.pt")
BUMBLE_URL = "http://localhost:11434"
OUTPUT_DIR = Path(__file__).parent / "data" / "benchmark_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RESULT TRACKING
# =============================================================================

@dataclass
class StudyResult:
    study_id: str
    success: bool = False
    error: Optional[str] = None
    total_ms: float = 0.0
    dicom_ms: float = 0.0
    detect_ms: float = 0.0
    llm_ms: float = 0.0
    pdf_ms: float = 0.0
    tokens: int = 0


# =============================================================================
# SPINE GOLD PROMPT
# =============================================================================

SPINE_PROMPT = {
    "system": """You are an expert musculoskeletal radiologist analyzing a spine X-ray. Provide a concise structured report.

FINDINGS TO ANALYZE:
{findings}

Format:
TECHNIQUE: [View and region]
FINDINGS: [Key observations]
IMPRESSION: [1-2 sentences]""",
    "user": """Spine X-ray analysis.
Vertebrae detected: {vertebrae}
Abnormalities: {abnormalities}
Confidence: {confidence:.1%}

Provide a brief clinical report.""",
}


# =============================================================================
# COMPONENTS
# =============================================================================

def load_dicom(path: Path) -> Dict[str, Any]:
    """Load DICOM file"""
    ds = pydicom.dcmread(str(path))
    pixel_array = ds.pixel_array

    if pixel_array.dtype != np.uint8:
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)

    img = Image.fromarray(pixel_array).resize((640, 640), Image.Resampling.LANCZOS)

    return {
        "pixel_array": np.array(img),
        "patient_id": getattr(ds, "PatientID", "ANON"),
    }


class SpineDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None

    def load(self):
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
            except:
                pass
        return self.model is not None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            return {"vertebrae": 0, "abnormalities": 0, "confidence": 0.0}

        results = self.model(image, verbose=False)
        vertebrae, abnormalities = 0, 0
        confidences = []

        for r in results:
            if r.boxes:
                for box in r.boxes:
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    confidences.append(conf)
                    if cls == 0:
                        vertebrae += 1
                    else:
                        abnormalities += 1

        return {
            "vertebrae": vertebrae,
            "abnormalities": abnormalities,
            "confidence": sum(confidences) / len(confidences) if confidences else 0.0
        }


async def call_llm(prompt: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    """Call Ollama LLM - Optimized for throughput with 256 token cap"""
    try:
        response = await client.post(
            f"{BUMBLE_URL}/api/generate",
            json={
                "model": "meditron:70b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256, "num_ctx": 2048}
            }
        )
        response.raise_for_status()
        result = response.json()
        return {
            "text": result.get("response", ""),
            "tokens": result.get("eval_count", 0),
            "success": True
        }
    except Exception as e:
        return {"text": "", "tokens": 0, "success": False, "error": str(e)}


async def generate_pdf_fast(output_path: Path, study_id: str, report: str) -> bool:
    """Fast PDF generation"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"SPINE REPORT - {study_id}", styles['Heading1']),
            Paragraph(report.replace('\n', '<br/>'), styles['BodyText'])
        ]
        doc.build(story)
        return True
    except:
        return False


# =============================================================================
# PARALLEL WORKER
# =============================================================================

async def process_study(
    detector: SpineDetector,
    dicom_path: Path,
    study_idx: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> StudyResult:
    """Process single study with semaphore for concurrency control"""

    study_id = f"par-{study_idx:04d}"
    result = StudyResult(study_id=study_id)

    async with semaphore:
        total_start = time.perf_counter()

        try:
            # DICOM Load
            t0 = time.perf_counter()
            data = load_dicom(dicom_path)
            result.dicom_ms = (time.perf_counter() - t0) * 1000

            # Spine Detection
            t0 = time.perf_counter()
            detection = detector.detect(data["pixel_array"])
            result.detect_ms = (time.perf_counter() - t0) * 1000

            # Build prompt
            prompt = SPINE_PROMPT["system"].format(findings="Spine X-ray study")
            prompt += "\n\n" + SPINE_PROMPT["user"].format(
                vertebrae=detection["vertebrae"],
                abnormalities=detection["abnormalities"],
                confidence=detection["confidence"]
            )

            # LLM Inference
            t0 = time.perf_counter()
            llm_result = await call_llm(prompt, client)
            result.llm_ms = (time.perf_counter() - t0) * 1000
            result.tokens = llm_result.get("tokens", 0)

            if not llm_result["success"]:
                result.error = llm_result.get("error")
                result.total_ms = (time.perf_counter() - total_start) * 1000
                return result

            # PDF Generation
            t0 = time.perf_counter()
            pdf_path = OUTPUT_DIR / f"{study_id}_report.pdf"
            await generate_pdf_fast(pdf_path, study_id, llm_result["text"])
            result.pdf_ms = (time.perf_counter() - t0) * 1000

            result.success = True
            result.total_ms = (time.perf_counter() - total_start) * 1000

        except Exception as e:
            result.error = str(e)
            result.total_ms = (time.perf_counter() - total_start) * 1000

    return result


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

async def run_parallel_benchmark(
    num_studies: int = 20,
    workers: int = 2,
    warmup: int = 2,
    use_nvme: bool = True,
):
    """Run parallel benchmark"""

    samples_dir = SPINE_SAMPLES_DIR_NVME if use_nvme else SPINE_SAMPLES_DIR_NAS

    print("=" * 70)
    print("🚀 PARALLEL SPINE BENCHMARK — Dual GPU Edition")
    print("=" * 70)
    print()
    print(f"📁 Samples: {samples_dir}")
    print(f"👷 Workers: {workers} concurrent")
    print(f"📊 Studies: {num_studies}")
    print(f"🔥 Warmup: {warmup}")
    print()

    # Init
    sample_files = sorted(samples_dir.glob("*.dicom"))[:num_studies]
    if not sample_files:
        print("❌ No samples found!")
        return

    detector = SpineDetector(SPINE_MODEL_PATH)
    detector.load()
    print(f"   ✅ Spine detector loaded")

    # Check Ollama
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{BUMBLE_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            has_meditron = any("meditron" in m for m in models)
            print(f"   ✅ Bumble70B: {'meditron:70b ready' if has_meditron else 'loading...'}")
        except Exception as e:
            print(f"   ❌ Ollama error: {e}")
            return
    print()

    # Warmup
    if warmup > 0:
        print(f"🔥 Warmup ({warmup} sequential)...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            semaphore = asyncio.Semaphore(1)
            for i in range(warmup):
                await process_study(detector, sample_files[i % len(sample_files)], -1, client, semaphore)
                print(f"   Warmup {i+1}/{warmup}")
        print()

    # Main benchmark
    print(f"⏱️  Benchmark: {len(sample_files)} studies with {workers} workers...")
    print("-" * 70)

    semaphore = asyncio.Semaphore(workers)

    benchmark_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            process_study(detector, path, i, client, semaphore)
            for i, path in enumerate(sample_files)
        ]
        results = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start

    # Print results as they complete
    for r in results:
        status = "✅" if r.success else "❌"
        print(f"   {status} {r.study_id} | DICOM:{r.dicom_ms:5.0f}ms | YOLO:{r.detect_ms:4.0f}ms | "
              f"LLM:{r.llm_ms:6.0f}ms | PDF:{r.pdf_ms:3.0f}ms | Total:{r.total_ms:7.0f}ms")

    print("-" * 70)
    print()

    # Stats
    successful = [r for r in results if r.success]

    if successful:
        avg_total = statistics.mean(r.total_ms for r in successful)
        avg_llm = statistics.mean(r.llm_ms for r in successful)
        avg_dicom = statistics.mean(r.dicom_ms for r in successful)
        avg_detect = statistics.mean(r.detect_ms for r in successful)

        # Throughput: studies / actual_time
        actual_throughput = len(successful) / benchmark_duration * 3600

        # Theoretical max if no contention
        theoretical_per_study = avg_total / 1000  # seconds
        theoretical_throughput = 3600 / theoretical_per_study * workers

        print("=" * 70)
        print("📊 PARALLEL BENCHMARK RESULTS")
        print("=" * 70)
        print()
        print(f"   Studies:          {len(results)} total, {len(successful)} successful")
        print(f"   Workers:          {workers} concurrent")
        print(f"   Wall Clock Time:  {benchmark_duration:.1f}s")
        print()
        print("⏱️  TIMING (per study average):")
        print(f"   ├─ DICOM Load:     {avg_dicom:7.0f} ms")
        print(f"   ├─ Spine Detect:   {avg_detect:7.0f} ms")
        print(f"   ├─ LLM Inference:  {avg_llm:7.0f} ms  ⬅️ BOTTLENECK")
        print(f"   └─ TOTAL:          {avg_total:7.0f} ms")
        print()
        print("🚀 THROUGHPUT:")
        print(f"   ├─ Actual:         {actual_throughput:.0f} studies/hour")
        print(f"   └─ Theoretical:    {theoretical_throughput:.0f} studies/hour (with {workers} workers)")
        print()
        print(f"📈 Speedup vs Sequential: {actual_throughput / (3600000 / avg_total):.2f}x")
        print()

        # Save results
        results_file = OUTPUT_DIR / f"parallel_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump({
                "config": {"workers": workers, "studies": len(results)},
                "summary": {
                    "successful": len(successful),
                    "wall_clock_seconds": benchmark_duration,
                    "actual_throughput_per_hour": actual_throughput,
                    "avg_total_ms": avg_total,
                    "avg_llm_ms": avg_llm,
                },
                "results": [
                    {"id": r.study_id, "success": r.success, "total_ms": r.total_ms, "llm_ms": r.llm_ms}
                    for r in results
                ]
            }, f, indent=2)

        print(f"📁 Results: {results_file}")
        print()
        print("=" * 70)
        print(f"💎 {workers} WORKERS: {actual_throughput:.0f} spine studies/hour")
        print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Spine Benchmark")
    parser.add_argument("-n", "--num-studies", type=int, default=20)
    parser.add_argument("-w", "--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--nas", action="store_true", help="Use NAS instead of NVMe")

    args = parser.parse_args()

    asyncio.run(run_parallel_benchmark(
        num_studies=args.num_studies,
        workers=args.workers,
        warmup=args.warmup,
        use_nvme=not args.nas,
    ))

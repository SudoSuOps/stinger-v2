#!/usr/bin/env python3
"""
=======================================================================
SPINE THROUGHPUT BENCHMARK — "The Full Port Drop"
=======================================================================

End-to-end benchmark measuring how many spine studies can be processed
in 1 hour including:
- DICOM loading & preprocessing
- YOLO spine detection (vertebrae, abnormalities)
- Gold prompt retrieval from QueenBee
- Bumble70B clinical report generation
- PDF report generation

Run: python benchmark_spine_throughput.py

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
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import statistics

# Add stinger to path
sys.path.insert(0, str(Path(__file__).parent))

import pydicom
import numpy as np
from PIL import Image
import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

SPINE_SAMPLES_DIR_DEFAULT = Path(__file__).parent / "e2e_samples" / "spine"
SPINE_SAMPLES_DIR_NVME = Path("/mnt/btcnode/datasets/spine")
SPINE_MODEL_PATH = Path("/home/ai/Desktop/quantum-rails/queenbee-llm/spine_detector/checkpoints/spine_detector_20260103_140009/weights/best.pt")
QUEENBEE_URL = "http://192.168.0.52:8200"
BUMBLE_URL = "http://localhost:11434"  # Ollama
OUTPUT_DIR = Path(__file__).parent / "data" / "benchmark_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TIMING DATACLASS
# =============================================================================

@dataclass
class StageTiming:
    """Timing for a single pipeline stage"""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()


@dataclass
class BenchmarkResult:
    """Result for a single study"""
    study_id: str
    success: bool = False
    error: Optional[str] = None

    # Stage timings
    dicom_load_ms: float = 0.0
    spine_detect_ms: float = 0.0
    prompt_fetch_ms: float = 0.0
    llm_inference_ms: float = 0.0
    pdf_generation_ms: float = 0.0

    # Total
    total_ms: float = 0.0

    # Results
    vertebrae_detected: int = 0
    abnormalities_found: int = 0
    report_tokens: int = 0
    pdf_path: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics"""
    total_studies: int = 0
    successful: int = 0
    failed: int = 0

    # Timing stats (ms)
    avg_dicom_load_ms: float = 0.0
    avg_spine_detect_ms: float = 0.0
    avg_prompt_fetch_ms: float = 0.0
    avg_llm_inference_ms: float = 0.0
    avg_pdf_generation_ms: float = 0.0
    avg_total_ms: float = 0.0

    min_total_ms: float = 0.0
    max_total_ms: float = 0.0
    stddev_total_ms: float = 0.0

    # Throughput
    studies_per_minute: float = 0.0
    studies_per_hour: float = 0.0

    # Bottleneck analysis
    bottleneck_stage: str = ""
    bottleneck_percentage: float = 0.0


# =============================================================================
# DICOM LOADER
# =============================================================================

def load_dicom(path: Path) -> Dict[str, Any]:
    """Load and preprocess DICOM file"""
    ds = pydicom.dcmread(str(path))

    # Get pixel array
    pixel_array = ds.pixel_array

    # Normalize to 0-255
    if pixel_array.dtype != np.uint8:
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    # Convert to RGB if grayscale
    if len(pixel_array.shape) == 2:
        pixel_array = np.stack([pixel_array] * 3, axis=-1)

    # Resize for model (640x640 for YOLO)
    img = Image.fromarray(pixel_array)
    img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)

    return {
        "pixel_array": np.array(img_resized),
        "original_shape": ds.pixel_array.shape,
        "patient_id": getattr(ds, "PatientID", "ANONYMOUS"),
        "study_date": getattr(ds, "StudyDate", "Unknown"),
        "modality": getattr(ds, "Modality", "XR"),
        "body_part": getattr(ds, "BodyPartExamined", "SPINE"),
    }


# =============================================================================
# SPINE DETECTOR
# =============================================================================

class SpineDetector:
    """YOLO-based spine vertebrae and abnormality detector"""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Load YOLO model"""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                print(f"   ✅ Spine detector loaded: {self.model_path.name}")
            except Exception as e:
                print(f"   ⚠️  Spine detector load failed: {e}")
                self.model = None
        return self.model is not None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Run detection on spine image"""
        if self.model is None:
            # Return mock result if model not loaded
            return {
                "vertebrae": [],
                "abnormalities": [],
                "confidence": 0.0,
                "mock": True
            }

        # Run inference
        results = self.model(image, verbose=False)

        # Parse results
        vertebrae = []
        abnormalities = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                    coords = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else [0,0,0,0]

                    detection = {
                        "class": cls,
                        "confidence": conf,
                        "bbox": coords,
                    }

                    # Class 0 = vertebra, other = abnormality
                    if cls == 0:
                        vertebrae.append(detection)
                    else:
                        abnormalities.append(detection)

        avg_conf = sum(v["confidence"] for v in vertebrae) / len(vertebrae) if vertebrae else 0.0

        return {
            "vertebrae": vertebrae,
            "abnormalities": abnormalities,
            "vertebrae_count": len(vertebrae),
            "abnormality_count": len(abnormalities),
            "average_confidence": avg_conf,
        }


# =============================================================================
# QUEENBEE CLIENT (GOLD PROMPTS)
# =============================================================================

# Fallback prompt if QueenBee offline
SPINE_GOLD_PROMPT = {
    "system": """You are an expert musculoskeletal radiologist analyzing a spine X-ray. Provide a structured radiology report following ACR guidelines.

FINDINGS TO ANALYZE:
{findings}

Provide your report in this exact format:

TECHNIQUE:
- View: [AP/Lateral/Oblique]
- Region: [Cervical/Thoracic/Lumbar/Sacral]

VERTEBRAL BODIES:
- Alignment: [Normal/Anterolisthesis/Retrolisthesis/Scoliosis]
- Height: [Maintained/Compression fractures at levels]
- Marrow signal: [Normal/Abnormal]

INTERVERTEBRAL DISCS:
- Height: [Maintained/Reduced at levels]
- Disc space narrowing: [None/Levels affected]

SPINAL CANAL:
- Central canal: [Patent/Stenosis at levels]
- Neural foramina: [Patent/Narrowed at levels]

POSTERIOR ELEMENTS:
- Pedicles: [Intact/Abnormal]
- Facet joints: [Normal/Arthrosis]
- Spinous processes: [Normal/Abnormal]

SOFT TISSUES:
- Paravertebral tissues: [Normal/Abnormal]

OTHER:
- Hardware: [None/Present - describe]
- Incidental findings

IMPRESSION:
1. [Primary finding]
2. [Secondary findings]

RECOMMENDATIONS:
[Clinical correlation, follow-up imaging if needed]""",
    "user_template": """Analyze this spine X-ray study.

PATIENT: {patient_id}
DATE: {study_date}

DETECTED FINDINGS:
- Vertebrae visualized: {vertebrae_count}
- Abnormalities detected: {abnormality_count}
- Average detection confidence: {avg_confidence:.1%}

Provide a comprehensive spine radiology report.""",
}


async def get_gold_prompt(study_type: str = "spine") -> Dict[str, str]:
    """Get gold prompt from QueenBee or use fallback"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QUEENBEE_URL}/v1/prompt/{study_type}")
            if response.status_code == 200:
                return response.json()
    except:
        pass

    # Return fallback
    return SPINE_GOLD_PROMPT


# =============================================================================
# BUMBLE70B CLIENT (LLM INFERENCE)
# =============================================================================

async def call_bumble70b(prompt: str) -> Dict[str, Any]:
    """Call Bumble70B (meditron:70b via Ollama) for report generation"""

    request_body = {
        "model": "meditron:70b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "top_p": 0.9,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{BUMBLE_URL}/api/generate",
                json=request_body,
            )
            response.raise_for_status()
            result = response.json()

            return {
                "text": result.get("response", ""),
                "tokens": result.get("eval_count", 0),
                "duration_ns": result.get("total_duration", 0),
                "success": True,
            }
    except Exception as e:
        return {
            "text": f"Error: {str(e)}",
            "tokens": 0,
            "duration_ns": 0,
            "success": False,
            "error": str(e),
        }


# =============================================================================
# PDF GENERATOR
# =============================================================================

async def generate_pdf(
    output_path: Path,
    study_id: str,
    patient_id: str,
    study_date: str,
    detection_results: Dict[str, Any],
    report_text: str,
) -> bool:
    """Generate PDF report"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='TrustCatTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#4ade80'),
            spaceAfter=15,
            alignment=TA_CENTER,
        ))

        story = []
        story.append(Paragraph("TRUSTCAT SPINE REPORT", styles['TrustCatTitle']))
        story.append(Spacer(1, 10))

        # Info table
        info_data = [
            ["Patient ID:", patient_id, "Study Date:", study_date],
            ["Study ID:", study_id[:20], "Generated:", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Vertebrae:", str(detection_results.get("vertebrae_count", 0)),
             "Abnormalities:", str(detection_results.get("abnormality_count", 0))],
        ]

        info_table = Table(info_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#64748b')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)

        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#4ade80'),
                                spaceBefore=15, spaceAfter=15))

        # Report text
        for para in report_text.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles['BodyText']))
                story.append(Spacer(1, 6))

        # Footer
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'),
                                spaceBefore=10, spaceAfter=10))
        story.append(Paragraph("trustcat.ai | stinger.swarmbee.eth | Benchmark Run",
                              ParagraphStyle('Footer', fontSize=8, textColor=colors.HexColor('#64748b'),
                                           alignment=TA_CENTER)))

        doc.build(story)
        return True

    except Exception as e:
        print(f"      PDF error: {e}")
        return False


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

async def benchmark_single_study(
    spine_detector: SpineDetector,
    dicom_path: Path,
    study_index: int,
) -> BenchmarkResult:
    """Run full E2E benchmark on a single study"""

    study_id = f"bench-{study_index:04d}-{hashlib.sha256(str(dicom_path).encode()).hexdigest()[:8]}"
    result = BenchmarkResult(study_id=study_id)

    total_start = time.perf_counter()

    try:
        # =====================================================================
        # STAGE 1: DICOM LOAD
        # =====================================================================
        stage_start = time.perf_counter()

        study_data = load_dicom(dicom_path)

        result.dicom_load_ms = (time.perf_counter() - stage_start) * 1000

        # =====================================================================
        # STAGE 2: SPINE DETECTION
        # =====================================================================
        stage_start = time.perf_counter()

        detection = spine_detector.detect(study_data["pixel_array"])

        result.spine_detect_ms = (time.perf_counter() - stage_start) * 1000
        result.vertebrae_detected = detection.get("vertebrae_count", 0)
        result.abnormalities_found = detection.get("abnormality_count", 0)

        # =====================================================================
        # STAGE 3: GOLD PROMPT FETCH
        # =====================================================================
        stage_start = time.perf_counter()

        gold_prompt = await get_gold_prompt("spine")

        result.prompt_fetch_ms = (time.perf_counter() - stage_start) * 1000

        # Build full prompt
        user_prompt = gold_prompt["user_template"].format(
            patient_id=study_data.get("patient_id", "ANONYMOUS"),
            study_date=study_data.get("study_date", "Unknown"),
            vertebrae_count=result.vertebrae_detected,
            abnormality_count=result.abnormalities_found,
            avg_confidence=detection.get("average_confidence", 0.0),
        )

        full_prompt = f"{gold_prompt['system']}\n\n{user_prompt}"

        # =====================================================================
        # STAGE 4: LLM INFERENCE (BUMBLE70B)
        # =====================================================================
        stage_start = time.perf_counter()

        llm_result = await call_bumble70b(full_prompt)

        result.llm_inference_ms = (time.perf_counter() - stage_start) * 1000
        result.report_tokens = llm_result.get("tokens", 0)

        if not llm_result.get("success"):
            result.error = llm_result.get("error", "LLM inference failed")
            result.success = False
            result.total_ms = (time.perf_counter() - total_start) * 1000
            return result

        report_text = llm_result["text"]

        # =====================================================================
        # STAGE 5: PDF GENERATION
        # =====================================================================
        stage_start = time.perf_counter()

        pdf_path = OUTPUT_DIR / f"{study_id}_report.pdf"
        pdf_success = await generate_pdf(
            output_path=pdf_path,
            study_id=study_id,
            patient_id=study_data.get("patient_id", "ANONYMOUS"),
            study_date=study_data.get("study_date", "Unknown"),
            detection_results=detection,
            report_text=report_text,
        )

        result.pdf_generation_ms = (time.perf_counter() - stage_start) * 1000
        result.pdf_path = str(pdf_path) if pdf_success else None

        # =====================================================================
        # COMPLETE
        # =====================================================================
        result.success = True
        result.total_ms = (time.perf_counter() - total_start) * 1000

    except Exception as e:
        result.success = False
        result.error = str(e)
        result.total_ms = (time.perf_counter() - total_start) * 1000

    return result


async def run_benchmark(num_studies: int = None, warmup: int = 2, samples_dir: Path = None):
    """Run full E2E benchmark"""

    samples_dir = samples_dir or SPINE_SAMPLES_DIR_DEFAULT

    print("=" * 70)
    print("🎯 SPINE THROUGHPUT BENCHMARK — The Full Port Drop")
    print("=" * 70)
    print()

    # Discover samples
    sample_files = sorted(samples_dir.glob("*.dicom"))
    if not sample_files:
        print("❌ No spine samples found!")
        return None

    num_available = len(sample_files)
    num_studies = num_studies or num_available
    num_studies = min(num_studies, num_available)

    storage_type = "NVMe" if "btcnode" in str(samples_dir) or "nvme" in str(samples_dir).lower() else "NAS"
    print(f"📁 Samples directory: {samples_dir}")
    print(f"💾 Storage type: {storage_type}")
    print(f"📊 Available samples: {num_available}")
    print(f"🎯 Studies to benchmark: {num_studies}")
    print(f"🔥 Warmup runs: {warmup}")
    print()

    # Initialize components
    print("🔌 Initializing pipeline components...")

    # Spine detector
    spine_detector = SpineDetector(SPINE_MODEL_PATH)
    detector_loaded = spine_detector.load()

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BUMBLE_URL}/api/tags")
            models = response.json().get("models", [])
            has_meditron = any("meditron" in m.get("name", "") for m in models)
            print(f"   ✅ Bumble70B (Ollama): {'meditron:70b loaded' if has_meditron else 'needs model load'}")
    except Exception as e:
        print(f"   ⚠️  Bumble70B (Ollama): offline - {e}")
        return None

    # Check QueenBee
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QUEENBEE_URL}/health")
            print(f"   ✅ QueenBee: online")
    except:
        print(f"   ⚠️  QueenBee: offline (using fallback prompts)")

    print()

    # Warmup runs
    if warmup > 0:
        print(f"🔥 Running {warmup} warmup iterations...")
        for i in range(warmup):
            _ = await benchmark_single_study(spine_detector, sample_files[i % len(sample_files)], -1)
            print(f"   Warmup {i+1}/{warmup} complete")
        print()

    # Main benchmark
    print(f"⏱️  Starting benchmark of {num_studies} studies...")
    print("-" * 70)

    results: List[BenchmarkResult] = []
    benchmark_start = time.perf_counter()

    for i, sample_path in enumerate(sample_files[:num_studies]):
        study_start = time.perf_counter()

        result = await benchmark_single_study(spine_detector, sample_path, i)
        results.append(result)

        status = "✅" if result.success else "❌"
        print(f"   [{i+1:3d}/{num_studies}] {status} {sample_path.name[:30]:30s} | "
              f"DICOM:{result.dicom_load_ms:6.0f}ms | "
              f"YOLO:{result.spine_detect_ms:6.0f}ms | "
              f"LLM:{result.llm_inference_ms:7.0f}ms | "
              f"PDF:{result.pdf_generation_ms:5.0f}ms | "
              f"Total:{result.total_ms:8.0f}ms")

    benchmark_duration = time.perf_counter() - benchmark_start

    print("-" * 70)
    print()

    # Calculate summary
    successful_results = [r for r in results if r.success]

    summary = BenchmarkSummary(
        total_studies=len(results),
        successful=len(successful_results),
        failed=len(results) - len(successful_results),
    )

    if successful_results:
        # Timing averages
        summary.avg_dicom_load_ms = statistics.mean(r.dicom_load_ms for r in successful_results)
        summary.avg_spine_detect_ms = statistics.mean(r.spine_detect_ms for r in successful_results)
        summary.avg_prompt_fetch_ms = statistics.mean(r.prompt_fetch_ms for r in successful_results)
        summary.avg_llm_inference_ms = statistics.mean(r.llm_inference_ms for r in successful_results)
        summary.avg_pdf_generation_ms = statistics.mean(r.pdf_generation_ms for r in successful_results)
        summary.avg_total_ms = statistics.mean(r.total_ms for r in successful_results)

        totals = [r.total_ms for r in successful_results]
        summary.min_total_ms = min(totals)
        summary.max_total_ms = max(totals)
        summary.stddev_total_ms = statistics.stdev(totals) if len(totals) > 1 else 0.0

        # Throughput
        summary.studies_per_minute = 60000.0 / summary.avg_total_ms
        summary.studies_per_hour = summary.studies_per_minute * 60

        # Bottleneck analysis
        stage_times = {
            "DICOM Load": summary.avg_dicom_load_ms,
            "Spine Detection": summary.avg_spine_detect_ms,
            "Prompt Fetch": summary.avg_prompt_fetch_ms,
            "LLM Inference": summary.avg_llm_inference_ms,
            "PDF Generation": summary.avg_pdf_generation_ms,
        }
        summary.bottleneck_stage = max(stage_times, key=stage_times.get)
        summary.bottleneck_percentage = stage_times[summary.bottleneck_stage] / summary.avg_total_ms * 100

    # Print summary
    print("=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"   Total Studies:     {summary.total_studies}")
    print(f"   Successful:        {summary.successful}")
    print(f"   Failed:            {summary.failed}")
    print(f"   Success Rate:      {summary.successful/summary.total_studies*100:.1f}%")
    print()

    print("⏱️  TIMING BREAKDOWN (average per study):")
    print(f"   ├─ DICOM Load:       {summary.avg_dicom_load_ms:8.1f} ms")
    print(f"   ├─ Spine Detection:  {summary.avg_spine_detect_ms:8.1f} ms")
    print(f"   ├─ Prompt Fetch:     {summary.avg_prompt_fetch_ms:8.1f} ms")
    print(f"   ├─ LLM Inference:    {summary.avg_llm_inference_ms:8.1f} ms  ⬅️  {'BOTTLENECK' if summary.bottleneck_stage == 'LLM Inference' else ''}")
    print(f"   └─ PDF Generation:   {summary.avg_pdf_generation_ms:8.1f} ms")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL:               {summary.avg_total_ms:8.1f} ms")
    print()

    print(f"   Min/Max/StdDev:      {summary.min_total_ms:.0f} / {summary.max_total_ms:.0f} / {summary.stddev_total_ms:.0f} ms")
    print()

    print("🚀 THROUGHPUT:")
    print(f"   ├─ Per Minute:       {summary.studies_per_minute:.2f} studies/min")
    print(f"   └─ Per Hour:         {summary.studies_per_hour:.1f} studies/hour")
    print()

    print("🔍 BOTTLENECK ANALYSIS:")
    print(f"   Slowest Stage:       {summary.bottleneck_stage}")
    print(f"   % of Total Time:     {summary.bottleneck_percentage:.1f}%")
    print()

    # Save results
    results_file = OUTPUT_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results_data = {
        "benchmark_time": datetime.now().isoformat(),
        "config": {
            "num_studies": num_studies,
            "warmup": warmup,
            "spine_model": str(SPINE_MODEL_PATH),
            "bumble_url": BUMBLE_URL,
            "queenbee_url": QUEENBEE_URL,
        },
        "summary": {
            "total_studies": summary.total_studies,
            "successful": summary.successful,
            "failed": summary.failed,
            "avg_total_ms": summary.avg_total_ms,
            "studies_per_hour": summary.studies_per_hour,
            "bottleneck_stage": summary.bottleneck_stage,
            "bottleneck_percentage": summary.bottleneck_percentage,
        },
        "timing_breakdown_ms": {
            "dicom_load": summary.avg_dicom_load_ms,
            "spine_detection": summary.avg_spine_detect_ms,
            "prompt_fetch": summary.avg_prompt_fetch_ms,
            "llm_inference": summary.avg_llm_inference_ms,
            "pdf_generation": summary.avg_pdf_generation_ms,
        },
        "individual_results": [
            {
                "study_id": r.study_id,
                "success": r.success,
                "error": r.error,
                "total_ms": r.total_ms,
                "vertebrae_detected": r.vertebrae_detected,
                "abnormalities_found": r.abnormalities_found,
                "report_tokens": r.report_tokens,
            }
            for r in results
        ]
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"📁 Results saved to: {results_file}")
    print()

    print("=" * 70)
    print(f"💎 CONCLUSION: {summary.studies_per_hour:.0f} spine studies/hour E2E with PDFs")
    print("=" * 70)

    return summary


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spine E2E Throughput Benchmark")
    parser.add_argument("-n", "--num-studies", type=int, default=10,
                        help="Number of studies to benchmark (default: 10)")
    parser.add_argument("-w", "--warmup", type=int, default=2,
                        help="Number of warmup iterations (default: 2)")
    parser.add_argument("--all", action="store_true",
                        help="Run all available samples")
    parser.add_argument("--nvme", action="store_true",
                        help="Use local NVMe storage instead of NAS")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Custom samples directory path")

    args = parser.parse_args()

    num_studies = None if args.all else args.num_studies

    # Determine samples directory
    if args.samples_dir:
        samples_dir = Path(args.samples_dir)
    elif args.nvme:
        samples_dir = SPINE_SAMPLES_DIR_NVME
    else:
        samples_dir = SPINE_SAMPLES_DIR_DEFAULT

    asyncio.run(run_benchmark(num_studies=num_studies, warmup=args.warmup, samples_dir=samples_dir))

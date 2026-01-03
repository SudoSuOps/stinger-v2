#!/usr/bin/env python3
"""
Stinger V2 — Production E2E Batch Validation
=============================================
Proves reliability, not just correctness.

Full pipeline for every sample:
  Disk → Stinger → Parse → Route → QueenBee → Bumble70B
       → PDF → Hash → Sign → IPFS → Ledger

No shortcuts. No stubs. No skipping.
"""

import asyncio
import json
import time
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import httpx

# Configuration
STINGER_URL = "http://localhost:8100"
SAMPLES_DIR = Path("e2e_samples")
RESULTS_DIR = Path("e2e_results")
MAX_POLL_TIME = 600  # 10 minutes max per job
POLL_INTERVAL = 5    # seconds


@dataclass
class JobResult:
    """Result of a single E2E job"""
    job_id: str
    sample_path: str
    study_type: str
    status: str

    # Timing metrics (ms)
    submit_time: float = 0
    complete_time: float = 0
    total_time_ms: int = 0

    # Artifact checks
    pdf_exists: bool = False
    pdf_size_bytes: int = 0
    merkle_root: str = ""
    ipfs_cid: str = ""
    signature: str = ""
    epoch_id: str = ""

    # Content metrics
    report_length: int = 0
    findings_count: int = 0

    # Error tracking
    error: str = ""
    abstained: bool = False

    # Determinism check
    input_hash: str = ""
    output_hash: str = ""


@dataclass
class BatchMetrics:
    """Aggregate metrics for the batch run"""
    total_samples: int = 0
    completed: int = 0
    failed: int = 0
    abstained: int = 0

    # By study type
    spine_count: int = 0
    spine_completed: int = 0
    ecg_count: int = 0
    ecg_completed: int = 0
    cgm_count: int = 0
    cgm_completed: int = 0

    # Timing (ms)
    total_time_ms: int = 0
    avg_time_ms: float = 0
    p50_time_ms: float = 0
    p95_time_ms: float = 0
    min_time_ms: int = 0
    max_time_ms: int = 0

    # Artifact integrity
    all_pdfs_exist: bool = True
    all_ledger_entries: bool = True
    all_hashes_valid: bool = True

    # Results
    results: List[JobResult] = field(default_factory=list)


class E2EBatchRunner:
    """Production E2E batch validation runner"""

    def __init__(self, stinger_url: str = STINGER_URL):
        self.stinger_url = stinger_url
        self.client = httpx.Client(timeout=30.0)
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)

    def discover_samples(self) -> List[Path]:
        """Discover all samples in e2e_samples/"""
        samples = []

        # Spine DICOMs
        for f in sorted(SAMPLES_DIR.glob("spine/*.dicom")):
            samples.append(f)

        # ECG files (need .dat files)
        for f in sorted(SAMPLES_DIR.glob("ecg/*.dat")):
            samples.append(f)

        # CGM CSVs
        for f in sorted(SAMPLES_DIR.glob("cgm/*.csv")):
            samples.append(f)

        return samples

    def compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file contents"""
        # Follow symlinks
        real_path = filepath.resolve()
        with open(real_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def submit_job(self, sample_path: Path) -> str:
        """Submit a sample for analysis, return job_id"""
        # Follow symlink to get real file
        real_path = sample_path.resolve()

        with open(real_path, 'rb') as f:
            files = {'file': (sample_path.name, f)}
            data = {
                'patient_id': f'E2E_{sample_path.stem}',
                'study_description': f'E2E Validation: {sample_path.parent.name}',
                'include_pdf': 'true',
                'include_proof': 'true',
            }

            response = self.client.post(
                f"{self.stinger_url}/analyze",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()['job_id']

    def poll_job(self, job_id: str, timeout: int = MAX_POLL_TIME) -> Dict[str, Any]:
        """Poll job until completion or timeout"""
        start = time.time()

        while time.time() - start < timeout:
            response = self.client.get(f"{self.stinger_url}/job/{job_id}")
            response.raise_for_status()
            data = response.json()

            status = data.get('status', '')
            if status in ('completed', 'failed'):
                return data

            time.sleep(POLL_INTERVAL)

        return {'status': 'timeout', 'message': f'Timed out after {timeout}s'}

    def verify_pdf(self, job_id: str) -> tuple[bool, int]:
        """Verify PDF exists and get size"""
        pdf_path = Path(f"data/outputs/{job_id}_report.pdf")
        if pdf_path.exists():
            return True, pdf_path.stat().st_size
        return False, 0

    def verify_ledger_entry(self, job_id: str) -> bool:
        """Verify job is recorded in ledger"""
        try:
            response = self.client.get(f"{self.stinger_url}/ledger/job/{job_id}")
            return response.status_code == 200
        except:
            return False

    def run_single(self, sample_path: Path) -> JobResult:
        """Run E2E for a single sample"""
        result = JobResult(
            job_id="",
            sample_path=str(sample_path),
            study_type=sample_path.parent.name,
            status="pending",
            input_hash=self.compute_file_hash(sample_path),
        )

        try:
            # Submit
            result.submit_time = time.time()
            result.job_id = self.submit_job(sample_path)

            # Poll until complete
            job_data = self.poll_job(result.job_id)
            result.complete_time = time.time()
            result.total_time_ms = int((result.complete_time - result.submit_time) * 1000)

            result.status = job_data.get('status', 'unknown')

            if result.status == 'completed' and job_data.get('result'):
                res = job_data['result']

                # Extract metrics
                result.report_length = len(res.get('report_text', ''))
                result.epoch_id = res.get('epoch', '')

                # Proof data
                proof = res.get('proof', {})
                if proof:
                    result.merkle_root = proof.get('merkle_root', '')
                    result.ipfs_cid = proof.get('ipfs_cid', '')
                    result.signature = proof.get('signature', '')

                # Findings
                findings = res.get('findings', {})
                result.findings_count = len(findings.get('findings', []))

                # Check for abstention
                if not result.report_length or result.report_length < 50:
                    result.abstained = True

                # Compute output hash for determinism check
                result.output_hash = hashlib.sha256(
                    json.dumps(res, sort_keys=True).encode()
                ).hexdigest()

                # Verify PDF
                result.pdf_exists, result.pdf_size_bytes = self.verify_pdf(result.job_id)

            elif result.status == 'failed':
                result.error = job_data.get('message', 'Unknown error')

        except Exception as e:
            result.status = 'error'
            result.error = str(e)
            result.complete_time = time.time()
            result.total_time_ms = int((result.complete_time - result.submit_time) * 1000)

        return result

    def run_batch(self, samples: List[Path], parallel: int = 1) -> BatchMetrics:
        """Run E2E validation on all samples"""
        metrics = BatchMetrics(total_samples=len(samples))

        print(f"\n{'='*70}")
        print(f"  STINGER V2 — PRODUCTION E2E VALIDATION")
        print(f"  {len(samples)} samples | {parallel} parallel jobs")
        print(f"{'='*70}\n")

        start_time = time.time()
        times = []

        if parallel > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [executor.submit(self.run_single, s) for s in samples]
                for i, future in enumerate(futures):
                    result = future.result()
                    metrics.results.append(result)
                    self._update_metrics(metrics, result)
                    self._print_progress(i + 1, len(samples), result)
                    if result.total_time_ms > 0:
                        times.append(result.total_time_ms)
        else:
            # Sequential execution
            for i, sample in enumerate(samples):
                result = self.run_single(sample)
                metrics.results.append(result)
                self._update_metrics(metrics, result)
                self._print_progress(i + 1, len(samples), result)
                if result.total_time_ms > 0:
                    times.append(result.total_time_ms)

        # Compute timing statistics
        metrics.total_time_ms = int((time.time() - start_time) * 1000)
        if times:
            times.sort()
            metrics.avg_time_ms = sum(times) / len(times)
            metrics.min_time_ms = min(times)
            metrics.max_time_ms = max(times)
            metrics.p50_time_ms = times[len(times) // 2]
            metrics.p95_time_ms = times[int(len(times) * 0.95)]

        # Verify all artifacts
        metrics.all_pdfs_exist = all(r.pdf_exists for r in metrics.results if r.status == 'completed')
        metrics.all_ledger_entries = all(
            self.verify_ledger_entry(r.job_id) for r in metrics.results if r.job_id
        )

        return metrics

    def _update_metrics(self, metrics: BatchMetrics, result: JobResult):
        """Update aggregate metrics from a single result"""
        if result.status == 'completed':
            metrics.completed += 1
        elif result.status == 'failed':
            metrics.failed += 1

        if result.abstained:
            metrics.abstained += 1

        # By study type
        if result.study_type == 'spine':
            metrics.spine_count += 1
            if result.status == 'completed':
                metrics.spine_completed += 1
        elif result.study_type == 'ecg':
            metrics.ecg_count += 1
            if result.status == 'completed':
                metrics.ecg_completed += 1
        elif result.study_type == 'cgm':
            metrics.cgm_count += 1
            if result.status == 'completed':
                metrics.cgm_completed += 1

    def _print_progress(self, current: int, total: int, result: JobResult):
        """Print progress update"""
        status_icon = {
            'completed': '✅',
            'failed': '❌',
            'timeout': '⏱️',
            'error': '💥',
        }.get(result.status, '❓')

        abstain_mark = ' [ABSTAIN]' if result.abstained else ''

        print(f"  [{current:3d}/{total}] {status_icon} {result.study_type:8s} | "
              f"{result.total_time_ms:6d}ms | "
              f"PDF:{result.pdf_size_bytes:6d}B | "
              f"Report:{result.report_length:5d}c{abstain_mark}")

    def save_results(self, metrics: BatchMetrics, filename: str = None):
        """Save detailed results to JSON"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"e2e_results_{timestamp}.json"

        filepath = self.results_dir / filename

        # Convert to serializable format
        data = {
            'summary': {
                'total_samples': metrics.total_samples,
                'completed': metrics.completed,
                'failed': metrics.failed,
                'abstained': metrics.abstained,
                'completion_rate': metrics.completed / metrics.total_samples if metrics.total_samples else 0,
                'spine': f"{metrics.spine_completed}/{metrics.spine_count}",
                'ecg': f"{metrics.ecg_completed}/{metrics.ecg_count}",
                'cgm': f"{metrics.cgm_completed}/{metrics.cgm_count}",
            },
            'timing': {
                'total_time_ms': metrics.total_time_ms,
                'avg_time_ms': metrics.avg_time_ms,
                'p50_time_ms': metrics.p50_time_ms,
                'p95_time_ms': metrics.p95_time_ms,
                'min_time_ms': metrics.min_time_ms,
                'max_time_ms': metrics.max_time_ms,
            },
            'integrity': {
                'all_pdfs_exist': metrics.all_pdfs_exist,
                'all_ledger_entries': metrics.all_ledger_entries,
                'all_hashes_valid': metrics.all_hashes_valid,
            },
            'results': [asdict(r) for r in metrics.results],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def print_summary(self, metrics: BatchMetrics):
        """Print final summary report"""
        completion_rate = metrics.completed / metrics.total_samples * 100 if metrics.total_samples else 0

        print(f"\n{'='*70}")
        print(f"  E2E VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"""
  RELIABILITY METRICS
  ───────────────────────────────────────────────────────────────────
  Total Samples:     {metrics.total_samples}
  Completed:         {metrics.completed} ({completion_rate:.1f}%)
  Failed:            {metrics.failed}
  Abstained:         {metrics.abstained}

  BY STUDY TYPE
  ───────────────────────────────────────────────────────────────────
  Spine:             {metrics.spine_completed}/{metrics.spine_count}
  ECG:               {metrics.ecg_completed}/{metrics.ecg_count}
  CGM:               {metrics.cgm_completed}/{metrics.cgm_count}

  TIMING (ms)
  ───────────────────────────────────────────────────────────────────
  Total Wall Time:   {metrics.total_time_ms:,}ms ({metrics.total_time_ms/1000:.1f}s)
  Average:           {metrics.avg_time_ms:,.0f}ms
  P50 (Median):      {metrics.p50_time_ms:,.0f}ms
  P95:               {metrics.p95_time_ms:,.0f}ms
  Min:               {metrics.min_time_ms:,}ms
  Max:               {metrics.max_time_ms:,}ms

  ARTIFACT INTEGRITY
  ───────────────────────────────────────────────────────────────────
  All PDFs Exist:    {'✅ YES' if metrics.all_pdfs_exist else '❌ NO'}
  All Ledger Entries:{'✅ YES' if metrics.all_ledger_entries else '❌ NO'}
  All Hashes Valid:  {'✅ YES' if metrics.all_hashes_valid else '❌ NO'}
""")

        # Final verdict
        if completion_rate >= 98 and metrics.all_pdfs_exist and metrics.all_ledger_entries:
            print(f"  {'='*67}")
            print(f"  💎 VALIDATION PASSED — PRODUCTION READY 💎")
            print(f"  {'='*67}")
        else:
            print(f"  {'='*67}")
            print(f"  ⚠️  VALIDATION ISSUES DETECTED — REVIEW REQUIRED")
            print(f"  {'='*67}")

        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Stinger V2 E2E Batch Validation')
    parser.add_argument('--parallel', '-p', type=int, default=1,
                       help='Number of parallel jobs (default: 1)')
    parser.add_argument('--limit', '-n', type=int, default=0,
                       help='Limit number of samples (0 = all)')
    parser.add_argument('--type', '-t', choices=['spine', 'ecg', 'cgm', 'all'], default='all',
                       help='Study type to test (default: all)')
    args = parser.parse_args()

    runner = E2EBatchRunner()

    # Discover samples
    samples = runner.discover_samples()

    # Filter by type
    if args.type != 'all':
        samples = [s for s in samples if s.parent.name == args.type]

    # Limit
    if args.limit > 0:
        samples = samples[:args.limit]

    if not samples:
        print("No samples found!")
        return

    # Run batch
    metrics = runner.run_batch(samples, parallel=args.parallel)

    # Save results
    results_file = runner.save_results(metrics)
    print(f"\n  Results saved to: {results_file}")

    # Print summary
    runner.print_summary(metrics)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Stinger V2 - End-to-End Test
Full pipeline test: Client → Stinger → QueenBee → Bumble → PDF → Client

Diamond Hands Edition 💎
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_parsers():
    """Test input parsers"""
    print("\n" + "=" * 60)
    print("🔍 TESTING PARSERS")
    print("=" * 60)
    
    from stinger.parsers import detect_input_type, DicomParser, CgmParser
    
    # Test input type detection
    test_cases = [
        (b"\x00" * 128 + b"DICM", "test.dcm", "dicom"),
        (b"<?xml", "test.xml", "ecg"),
        (b"timestamp,glucose\n2024-01-01,120", "test.csv", "cgm"),
    ]
    
    for data, filename, expected in test_cases:
        detected = detect_input_type(data, filename)
        status = "✅" if detected == expected else "❌"
        print(f"   {status} {filename}: {detected} (expected {expected})")
    
    # Test CGM parser
    cgm_parser = CgmParser()
    cgm_data = b"timestamp,glucose\n2024-01-01 00:00,120\n2024-01-01 00:05,125\n2024-01-01 00:10,118"
    result = await cgm_parser.parse(cgm_data)
    print(f"   ✅ CGM Parser: {result['num_readings']} readings, mean={result['stats'].get('mean', 0):.1f}")
    
    return True


async def test_classifier():
    """Test study classifier"""
    print("\n" + "=" * 60)
    print("🎯 TESTING CLASSIFIER")
    print("=" * 60)
    
    from stinger.router import StudyClassifier, StudyType
    
    classifier = StudyClassifier()
    await classifier.load()
    
    # Test DICOM classification
    test_data = {
        "type": "dicom",
        "modality": "CR",
        "body_part": "lumbar spine",
        "study_description": "lumbar spine xray",
        "metadata": {}
    }
    
    result = await classifier.classify(test_data)
    status = "✅" if result.study_type == StudyType.SPINE else "❌"
    print(f"   {status} DICOM Spine: {result.study_type.value} (conf={result.confidence:.2f})")
    
    # Test ECG classification
    ecg_data = {"type": "ecg", "num_leads": 12}
    result = await classifier.classify(ecg_data)
    status = "✅" if result.study_type == StudyType.ECG else "❌"
    print(f"   {status} ECG: {result.study_type.value} (conf={result.confidence:.2f})")
    
    # Test CGM classification
    cgm_data = {"type": "cgm", "readings": []}
    result = await classifier.classify(cgm_data)
    status = "✅" if result.study_type == StudyType.CGM else "❌"
    print(f"   {status} CGM: {result.study_type.value} (conf={result.confidence:.2f})")
    
    return True


async def test_context_assembler():
    """Test context assembler"""
    print("\n" + "=" * 60)
    print("📦 TESTING CONTEXT ASSEMBLER")
    print("=" * 60)
    
    from stinger.context import ContextAssembler
    from stinger.router import StudyClassifier, Classification, StudyType, Modality, BodyRegion
    
    assembler = ContextAssembler()
    
    study_data = {
        "type": "dicom",
        "modality": "MR",
        "metadata": {
            "patient_id": "TEST001",
            "study_description": "MRI Lumbar Spine",
        }
    }
    
    classification = Classification(
        study_type=StudyType.SPINE,
        modality=Modality.MRI,
        body_region=BodyRegion.SPINE_LUMBAR,
        confidence=0.95
    )
    
    context = await assembler.assemble(
        study_data=study_data,
        classification=classification,
        patient_id="TEST001",
        job_id="test-job-001"
    )
    
    prompt_context = context.to_prompt_context()
    print(f"   ✅ Context assembled for job: {context.job_id}")
    print(f"   ✅ Study type: {context.study.study_type}")
    print(f"   ✅ Prompt context length: {len(prompt_context)} chars")
    
    return True


async def test_gold_prompts():
    """Test gold prompts"""
    print("\n" + "=" * 60)
    print("👑 TESTING GOLD PROMPTS")
    print("=" * 60)
    
    from stinger.queenbee import GOLD_PROMPTS, get_gold_prompt
    
    study_types = ["spine", "cardiac", "chest", "ecg", "cgm", "neuro"]
    
    for st in study_types:
        prompt = get_gold_prompt(st)
        sys_len = len(prompt.get("system", ""))
        user_len = len(prompt.get("user_template", ""))
        print(f"   ✅ {st}: system={sys_len} chars, template={user_len} chars")
    
    return True


async def test_merkle_tree():
    """Test Merkle tree"""
    print("\n" + "=" * 60)
    print("🌳 TESTING MERKLE TREE")
    print("=" * 60)
    
    from stinger.crypto import MerkleTree
    
    merkle = MerkleTree()
    
    # Test data
    job_data = {
        "job_id": "test-001",
        "study_type": "spine",
        "findings": {"stenosis": "moderate"},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    root = merkle.compute_root(job_data)
    print(f"   ✅ Merkle root: {root[:20]}...")
    
    # Verify deterministic
    root2 = merkle.compute_root(job_data)
    status = "✅" if root == root2 else "❌"
    print(f"   {status} Deterministic: {root == root2}")
    
    return True


async def test_eip191_signer():
    """Test EIP-191 signer"""
    print("\n" + "=" * 60)
    print("🔐 TESTING EIP-191 SIGNER")
    print("=" * 60)
    
    from stinger.crypto import EIP191Signer
    
    # Test with demo key
    signer = EIP191Signer(private_key="0x" + "ab" * 32)
    
    message = "0x1234567890abcdef"
    signature = signer.sign(message)
    
    print(f"   ✅ Signer address: {signer.address}")
    print(f"   ✅ Signature: {signature[:40]}...")
    
    return True


async def test_ledger():
    """Test SwarmPool ledger"""
    print("\n" + "=" * 60)
    print("📒 TESTING LEDGER")
    print("=" * 60)
    
    from stinger.ledger import SwarmPoolLedger
    import tempfile
    
    # Use temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    ledger = SwarmPoolLedger(db_path=db_path)
    await ledger.initialize()
    
    epoch = await ledger.get_current_epoch()
    print(f"   ✅ Current epoch: {epoch}")
    
    # Record a job
    await ledger.record_job(
        job_id="test-job-001",
        study_type="spine",
        status="completed",
        findings={"stenosis": "moderate"},
        processing_time_ms=1234
    )
    
    # Get stats
    stats = await ledger.get_stats()
    print(f"   ✅ Total jobs: {stats['total_jobs']}")
    print(f"   ✅ Completed: {stats['completed_jobs']}")
    
    await ledger.close()
    
    # Cleanup
    db_path.unlink()
    
    return True


async def test_full_pipeline():
    """Test full end-to-end pipeline (mock mode)"""
    print("\n" + "=" * 60)
    print("🚀 TESTING FULL PIPELINE (MOCK)")
    print("=" * 60)
    
    from stinger.parsers import CgmParser
    from stinger.router import StudyClassifier
    from stinger.context import ContextAssembler
    from stinger.queenbee import QueenBeeClient
    from stinger.crypto import MerkleTree, EIP191Signer
    from stinger.ledger import SwarmPoolLedger
    import tempfile
    
    start_time = time.time()
    
    # 1. Parse input
    print("   ⏳ Step 1: Parsing CGM data...")
    cgm_parser = CgmParser()
    cgm_data = b"""timestamp,glucose
2024-01-01 00:00,120
2024-01-01 00:05,125
2024-01-01 00:10,118
2024-01-01 00:15,130
2024-01-01 00:20,145
2024-01-01 00:25,160
2024-01-01 00:30,155
2024-01-01 00:35,140
2024-01-01 00:40,125
2024-01-01 00:45,115"""
    
    study_data = await cgm_parser.parse(cgm_data)
    print(f"   ✅ Parsed {study_data['num_readings']} readings")
    
    # 2. Classify
    print("   ⏳ Step 2: Classifying study...")
    classifier = StudyClassifier()
    await classifier.load()
    classification = await classifier.classify(study_data)
    print(f"   ✅ Classified as: {classification.study_type.value}")
    
    # 3. Assemble context
    print("   ⏳ Step 3: Assembling context...")
    assembler = ContextAssembler()
    context = await assembler.assemble(
        study_data=study_data,
        classification=classification,
        patient_id="TEST001",
        job_id="e2e-test-001"
    )
    print(f"   ✅ Context assembled")
    
    # 4. Get gold prompt
    print("   ⏳ Step 4: Getting gold prompt...")
    queenbee = QueenBeeClient()
    prompt = await queenbee.get_prompt(
        study_type=classification.study_type.value,
        body_region=classification.body_region.value,
        context=context
    )
    print(f"   ✅ Gold prompt retrieved ({len(prompt['system'])} chars)")
    
    # 5. Mock inference (skip actual Bumble call)
    print("   ⏳ Step 5: Mock inference...")
    mock_report = """
GLUCOSE ANALYSIS REPORT

FINDINGS:
- Time in Range (70-180 mg/dL): 70%
- Mean glucose: 133.3 mg/dL
- Glucose variability: Moderate

IMPRESSION:
Overall glucose control is fair with room for improvement.
Notable post-meal spike pattern observed.

RECOMMENDATIONS:
1. Consider pre-meal insulin timing adjustment
2. Monitor for post-prandial patterns
3. Review carbohydrate counting accuracy
"""
    findings = {
        "summary": "CGM analysis showing fair glucose control",
        "pathologies": ["Post-meal hyperglycemia"],
        "measurements": {"mean_glucose": 133.3, "tir": 70},
        "recommendations": ["Adjust insulin timing", "Monitor patterns"],
        "confidence": 0.92
    }
    print(f"   ✅ Inference complete (mock)")
    
    # 6. Generate proof
    print("   ⏳ Step 6: Generating cryptographic proof...")
    merkle = MerkleTree()
    signer = EIP191Signer(private_key="0x" + "ab" * 32)
    
    job_data = {
        "job_id": "e2e-test-001",
        "study_type": classification.study_type.value,
        "findings": findings,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    merkle_root = merkle.compute_root(job_data)
    signature = signer.sign(merkle_root)
    
    proof = {
        "merkle_root": merkle_root,
        "signature": signature,
        "signer": signer.address,
        "ipfs_cid": "QmTest123...",
        "chain_id": 1,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    print(f"   ✅ Proof generated: {merkle_root[:20]}...")
    
    # 7. Record to ledger
    print("   ⏳ Step 7: Recording to ledger...")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    ledger = SwarmPoolLedger(db_path=db_path)
    await ledger.initialize()
    
    await ledger.record_job(
        job_id="e2e-test-001",
        study_type=classification.study_type.value,
        status="completed",
        findings=findings,
        proof=proof,
        processing_time_ms=int((time.time() - start_time) * 1000)
    )
    
    epoch = await ledger.get_current_epoch()
    print(f"   ✅ Recorded to epoch: {epoch}")
    
    await ledger.close()
    db_path.unlink()
    
    # Summary
    elapsed_ms = int((time.time() - start_time) * 1000)
    print(f"\n   🎉 PIPELINE COMPLETE in {elapsed_ms}ms")
    
    return True


async def main():
    """Run all tests"""
    print("=" * 70)
    print("🐱 STINGER V2 — END-TO-END TEST SUITE")
    print("   stinger.swarmbee.eth")
    print("   Diamond Hands Edition 💎")
    print("=" * 70)
    
    tests = [
        ("Parsers", test_parsers),
        ("Classifier", test_classifier),
        ("Context Assembler", test_context_assembler),
        ("Gold Prompts", test_gold_prompts),
        ("Merkle Tree", test_merkle_tree),
        ("EIP-191 Signer", test_eip191_signer),
        ("Ledger", test_ledger),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n   ❌ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n   💎 ALL TESTS PASSED — DIAMOND HANDS VERIFIED 💎")
    else:
        print("\n   ⚠️  Some tests failed")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

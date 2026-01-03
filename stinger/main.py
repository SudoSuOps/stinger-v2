"""
STINGER V2 — The Intelligent Medical AI Gateway
stinger.swarmbee.eth

End-to-end sovereign medical AI pipeline:
Client → Stinger → QueenBee → Bumble70B → PDF → Client

Diamond Hands Edition. No shortcuts. No jeets.
"""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx

from stinger.config import Settings
from stinger.parsers import DicomParser, EcgParser, CgmParser, detect_input_type
from stinger.router import StudyClassifier, ModelOrchestrator
from stinger.context import ContextAssembler
from stinger.queenbee import QueenBeeClient, get_gold_prompt
from stinger.bumble import BumbleClient
from stinger.reports import PDFReportGenerator
from stinger.crypto import MerkleTree, EIP191Signer, IPFSClient
from stinger.ledger import SwarmPoolLedger, Job, JobStatus
from stinger.models import Study, StudyType, Report

# =============================================================================
# CONFIGURATION
# =============================================================================

settings = Settings()

app = FastAPI(
    title="Stinger V2",
    description="Intelligent Medical AI Gateway — stinger.swarmbee.eth",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SERVICE CLIENTS
# =============================================================================

# Parsers
dicom_parser = DicomParser()
ecg_parser = EcgParser()
cgm_parser = CgmParser()

# Router
classifier = StudyClassifier(model_path=settings.classifier_model_path)
orchestrator = ModelOrchestrator(settings)

# Context
context_assembler = ContextAssembler()

# QueenBee & Bumble
queenbee = QueenBeeClient(base_url=settings.queenbee_url)
bumble = BumbleClient(base_url=settings.bumble_url)

# Reports
pdf_generator = PDFReportGenerator(
    template_dir=Path(__file__).parent / "reports" / "templates",
    assets_dir=Path(__file__).parent / "reports" / "assets"
)

# Crypto
merkle = MerkleTree()
signer = EIP191Signer(private_key=settings.signer_private_key)
ipfs = IPFSClient(gateway_url=settings.ipfs_gateway)

# Ledger
ledger = SwarmPoolLedger(db_path=settings.ledger_db_path)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request to analyze medical data"""
    patient_id: Optional[str] = None
    study_description: Optional[str] = None
    priority: str = Field(default="routine", pattern="^(stat|urgent|routine)$")
    include_pdf: bool = True
    include_proof: bool = True


class AnalyzeResponse(BaseModel):
    """Response from analysis"""
    job_id: str
    status: str
    study_type: str
    findings: Dict[str, Any]
    report_text: str
    pdf_url: Optional[str] = None
    proof: Optional[Dict[str, Any]] = None
    processing_time_ms: int
    timestamp: str
    epoch: str


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: int
    message: str
    result: Optional[AnalyzeResponse] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    ens: str
    services: Dict[str, str]
    gpu_fleet: Dict[str, Any]
    uptime_seconds: float


# =============================================================================
# STATE
# =============================================================================

START_TIME = time.time()
JOBS: Dict[str, Job] = {}
OUTPUT_DIR = Path(settings.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CORE PIPELINE
# =============================================================================

async def process_study(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    request: AnalyzeRequest
) -> AnalyzeResponse:
    """
    Full end-to-end pipeline:
    Parse → Classify → Route → QueenBee → Bumble → Report → Sign → Ledger
    """
    start_time = time.time()
    job = JOBS[job_id]
    
    try:
        # =====================================================================
        # STEP 1: PARSE INPUT
        # =====================================================================
        job.update(status=JobStatus.PARSING, progress=10, message="Parsing input...")
        
        input_type = detect_input_type(file_bytes, filename)
        
        if input_type == "dicom":
            study_data = await dicom_parser.parse(file_bytes)
        elif input_type == "ecg":
            study_data = await ecg_parser.parse(file_bytes)
        elif input_type == "cgm":
            study_data = await cgm_parser.parse(file_bytes)
        else:
            raise HTTPException(400, f"Unsupported input type: {input_type}")
        
        # =====================================================================
        # STEP 2: CLASSIFY STUDY
        # =====================================================================
        job.update(status=JobStatus.CLASSIFYING, progress=20, message="Classifying study...")
        
        classification = await classifier.classify(study_data)
        study_type = classification.study_type
        body_region = classification.body_region
        modality = classification.modality
        
        # =====================================================================
        # STEP 3: ASSEMBLE CONTEXT
        # =====================================================================
        job.update(status=JobStatus.ASSEMBLING, progress=30, message="Assembling context...")
        
        context = await context_assembler.assemble(
            study_data=study_data,
            classification=classification,
            patient_id=request.patient_id,
            study_description=request.study_description
        )
        
        # =====================================================================
        # STEP 4: GET GOLD PROMPT FROM QUEENBEE
        # =====================================================================
        job.update(status=JobStatus.PROMPTING, progress=40, message="Getting gold prompt...")
        
        gold_prompt = await queenbee.get_prompt(
            study_type=study_type,
            body_region=body_region,
            context=context
        )
        
        # =====================================================================
        # STEP 5: INFERENCE WITH BUMBLE70B
        # =====================================================================
        job.update(status=JobStatus.INFERRING, progress=50, message="Running Bumble70B inference...")
        
        # Determine which models to run
        models_to_run = await orchestrator.determine_models(classification)
        
        # Run inference (may be multiple models)
        inference_results = {}
        for model_name, model_config in models_to_run.items():
            result = await bumble.infer(
                prompt=gold_prompt,
                image_data=study_data.get("pixel_array"),
                signal_data=study_data.get("signal_data"),
                model=model_config["model"],
                endpoint=model_config["endpoint"]
            )
            inference_results[model_name] = result
        
        # =====================================================================
        # STEP 6: EXTRACT FINDINGS
        # =====================================================================
        job.update(status=JobStatus.EXTRACTING, progress=70, message="Extracting findings...")
        
        # Parse structured findings from Bumble response
        primary_result = inference_results.get("primary", list(inference_results.values())[0])
        findings = extract_findings(primary_result, study_type)
        report_text = primary_result.get("report", primary_result.get("text", ""))
        
        # =====================================================================
        # STEP 7: GENERATE PDF REPORT
        # =====================================================================
        pdf_url = None
        if request.include_pdf:
            job.update(status=JobStatus.GENERATING, progress=80, message="Generating PDF report...")
            
            pdf_path = OUTPUT_DIR / f"{job_id}_report.pdf"
            await pdf_generator.generate(
                output_path=pdf_path,
                study_type=study_type,
                patient_id=request.patient_id or "ANONYMOUS",
                study_date=datetime.now(timezone.utc),
                findings=findings,
                report_text=report_text,
                images=study_data.get("images", []),
                job_id=job_id
            )
            pdf_url = f"/reports/{job_id}_report.pdf"
        
        # =====================================================================
        # STEP 8: CRYPTOGRAPHIC PROOF
        # =====================================================================
        proof = None
        if request.include_proof:
            job.update(status=JobStatus.SIGNING, progress=90, message="Generating cryptographic proof...")
            
            # Build Merkle tree of job data
            job_data = {
                "job_id": job_id,
                "study_type": study_type,
                "findings": findings,
                "report_hash": hashlib.sha256(report_text.encode()).hexdigest(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_hash": hashlib.sha256(file_bytes).hexdigest()
            }
            
            merkle_root = merkle.compute_root(job_data)
            
            # Sign with EIP-191
            signature = signer.sign(merkle_root)
            
            # Pin to IPFS
            ipfs_cid = await ipfs.pin(json.dumps(job_data))
            
            proof = {
                "merkle_root": merkle_root,
                "signature": signature,
                "signer": signer.address,
                "ipfs_cid": ipfs_cid,
                "chain_id": 1,  # Ethereum mainnet
                "timestamp": job_data["timestamp"]
            }
        
        # =====================================================================
        # STEP 9: RECORD IN LEDGER
        # =====================================================================
        job.update(status=JobStatus.RECORDING, progress=95, message="Recording to ledger...")
        
        await ledger.record_job(
            job_id=job_id,
            study_type=study_type,
            status="completed",
            findings=findings,
            proof=proof,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        current_epoch = await ledger.get_current_epoch()
        
        # =====================================================================
        # STEP 10: COMPLETE
        # =====================================================================
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response = AnalyzeResponse(
            job_id=job_id,
            status="completed",
            study_type=study_type,
            findings=findings,
            report_text=report_text,
            pdf_url=pdf_url,
            proof=proof,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            epoch=current_epoch
        )
        
        job.update(
            status=JobStatus.COMPLETED,
            progress=100,
            message="Analysis complete",
            result=response
        )
        
        return response
        
    except Exception as e:
        job.update(
            status=JobStatus.FAILED,
            progress=0,
            message=f"Error: {str(e)}"
        )
        raise


def extract_findings(result: Dict[str, Any], study_type: str) -> Dict[str, Any]:
    """Extract structured findings from Bumble response"""
    findings = {
        "summary": result.get("summary", ""),
        "pathologies": result.get("pathologies", []),
        "measurements": result.get("measurements", {}),
        "recommendations": result.get("recommendations", []),
        "confidence": result.get("confidence", 0.0),
        "study_type": study_type
    }
    
    # Study-type specific extractions
    if study_type == "spine":
        findings["levels"] = result.get("levels", {})
        findings["stenosis"] = result.get("stenosis", {})
        findings["disc_findings"] = result.get("disc_findings", {})
    elif study_type == "cardiac":
        findings["ejection_fraction"] = result.get("ejection_fraction")
        findings["wall_motion"] = result.get("wall_motion", {})
        findings["valve_findings"] = result.get("valve_findings", {})
    elif study_type == "ecg":
        findings["rhythm"] = result.get("rhythm", "")
        findings["rate"] = result.get("rate")
        findings["intervals"] = result.get("intervals", {})
        findings["axis"] = result.get("axis", {})
    
    return findings


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def health():
    """Health check and system status"""
    services = {}
    
    # Check QueenBee
    try:
        await queenbee.health()
        services["queenbee"] = "online"
    except:
        services["queenbee"] = "offline"
    
    # Check Bumble
    try:
        await bumble.health()
        services["bumble"] = "online"
    except:
        services["bumble"] = "offline"
    
    # Check IPFS
    try:
        await ipfs.health()
        services["ipfs"] = "online"
    except:
        services["ipfs"] = "offline"
    
    # Check Ledger
    try:
        await ledger.health()
        services["ledger"] = "online"
    except:
        services["ledger"] = "offline"
    
    return HealthResponse(
        status="online",
        version="2.0.0",
        ens="stinger.swarmbee.eth",
        services=services,
        gpu_fleet=await orchestrator.get_fleet_status(),
        uptime_seconds=time.time() - START_TIME
    )


@app.post("/analyze", response_model=JobStatusResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    study_description: Optional[str] = None,
    priority: str = "routine",
    include_pdf: bool = True,
    include_proof: bool = True
):
    """
    Submit a study for analysis.
    
    Accepts:
    - DICOM files (.dcm, .dicom)
    - ECG files (.xml, .scp, .dat)
    - CGM data (.csv, .json)
    
    Returns job ID for status polling.
    """
    # Generate job ID
    job_id = f"stinger-{uuid.uuid4().hex[:12]}"
    
    # Read file
    file_bytes = await file.read()
    
    # Create job
    job = Job(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0,
        message="Job queued",
        created_at=datetime.now(timezone.utc)
    )
    JOBS[job_id] = job
    
    # Create request
    request = AnalyzeRequest(
        patient_id=patient_id,
        study_description=study_description,
        priority=priority,
        include_pdf=include_pdf,
        include_proof=include_proof
    )
    
    # Process in background
    background_tasks.add_task(
        process_study,
        job_id,
        file_bytes,
        file.filename,
        request
    )
    
    return JobStatusResponse(
        job_id=job_id,
        status="queued",
        progress=0,
        message="Job queued for processing"
    )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a job"""
    if job_id not in JOBS:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    job = JOBS[job_id]
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        result=job.result
    )


# =============================================================================
# CGM GUARDIAN ENDPOINTS - The 3AM Guardian
# =============================================================================

class CGMPredictRequest(BaseModel):
    """Request for CGM prediction"""
    glucose_history: List[float] = Field(..., description="Glucose values in mg/dL (5-min intervals, at least 24 values)")
    timestamps: Optional[List[str]] = Field(None, description="ISO timestamps for each reading")
    patient_id: Optional[str] = None

class CGMPredictResponse(BaseModel):
    """Response from CGM prediction"""
    current_glucose: float
    predicted_30min: float
    predicted_60min: float
    predicted_sequence: List[float]
    anomaly: Dict[str, Any]
    trend: Dict[str, Any]
    alert: bool
    alert_level: str
    alert_message: Optional[str]
    time_in_range: Dict[str, float]

@app.post("/cgm/predict", response_model=CGMPredictResponse)
async def predict_glucose(request: CGMPredictRequest):
    """
    🐝 THE 3AM GUARDIAN - CGM Glucose Prediction

    Predicts glucose 30 and 60 minutes ahead with anomaly detection.

    Input:
    - glucose_history: List of glucose values in mg/dL (5-min intervals)
    - timestamps: Optional ISO timestamps

    Returns:
    - Predictions for 30min and 60min ahead
    - Anomaly classification (SEVERE_HYPO, HYPO, NORMAL, HYPER, SEVERE_HYPER)
    - Trend (FALLING, STABLE, RISING)
    - Alerts for dangerous glucose levels
    """
    from stinger.models.cgm_guardian import get_cgm_guardian

    guardian = get_cgm_guardian()

    if len(request.glucose_history) < 6:
        raise HTTPException(400, "Need at least 6 glucose readings (30 minutes)")

    prediction = guardian.predict(
        glucose_history=request.glucose_history,
        timestamps=request.timestamps,
    )

    if "error" in prediction:
        raise HTTPException(500, prediction["error"])

    return CGMPredictResponse(**prediction)


@app.get("/cgm/health")
async def cgm_health():
    """Check CGM Guardian health"""
    from stinger.models.cgm_guardian import get_cgm_guardian

    guardian = get_cgm_guardian()
    loaded = guardian.load()

    return {
        "service": "cgm_guardian",
        "name": "The 3AM Guardian",
        "status": "healthy" if loaded else "model_not_loaded",
        "model_path": str(guardian.model_path),
        "device": str(guardian.device),
        "capabilities": [
            "glucose_forecasting_30min",
            "glucose_forecasting_60min",
            "anomaly_detection",
            "trend_classification",
            "hypoglycemia_alerts",
        ]
    }


# =============================================================================
# ECG TRANSFORMER ENDPOINTS
# =============================================================================

class ECGAnalyzeRequest(BaseModel):
    """Request for ECG analysis"""
    signal: List[List[float]] = Field(..., description="12-lead ECG signal [12, samples] or [samples, 12]")
    sampling_rate: int = Field(500, description="Sampling rate in Hz")
    patient_id: Optional[str] = None

class ECGAnalyzeResponse(BaseModel):
    """Response from ECG analysis"""
    primary_diagnosis: Dict[str, Any]
    all_diagnoses: List[Dict[str, Any]]
    superclass_probabilities: Dict[str, float]
    is_normal: bool
    requires_attention: bool
    urgency: str

@app.post("/ecg/analyze", response_model=ECGAnalyzeResponse)
async def analyze_ecg(request: ECGAnalyzeRequest):
    """
    🐝 ECG-TRANSFORMER - 12-Lead ECG Analysis

    Analyzes 12-lead ECG for diagnostic classification.

    Input:
    - signal: 12-lead ECG signal [12, samples] or [samples, 12]
    - sampling_rate: Sample rate in Hz (default 500)

    Returns:
    - Primary diagnosis with confidence
    - All detected findings
    - Urgency level
    """
    from stinger.models.ecg_transformer import get_ecg_analyzer
    import numpy as np

    analyzer = get_ecg_analyzer()

    signal = np.array(request.signal)
    if signal.size == 0:
        raise HTTPException(400, "Empty signal provided")

    analysis = analyzer.analyze(
        signal=signal,
        sampling_rate=request.sampling_rate,
    )

    if "error" in analysis:
        raise HTTPException(500, analysis["error"])

    return ECGAnalyzeResponse(**analysis)


@app.get("/ecg/health")
async def ecg_health():
    """Check ECG Transformer health"""
    from stinger.models.ecg_transformer import get_ecg_analyzer

    analyzer = get_ecg_analyzer()
    loaded = analyzer.load()

    return {
        "service": "ecg_transformer",
        "name": "QueenBee ECG-Transformer",
        "status": "healthy" if loaded else "model_not_loaded",
        "model_path": str(analyzer.model_path),
        "device": str(analyzer.device),
        "capabilities": [
            "12_lead_analysis",
            "superclass_classification",
            "mi_detection",
            "arrhythmia_detection",
        ]
    }


# =============================================================================
# REPORT ENDPOINTS
# =============================================================================

@app.get("/reports/{filename}")
async def get_report(filename: str):
    """Download a generated report"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(404, f"Report not found: {filename}")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.get("/ledger/epoch")
async def get_current_epoch():
    """Get current epoch information"""
    return await ledger.get_epoch_info()


@app.get("/ledger/job/{job_id}")
async def get_job_record(job_id: str):
    """Get job record from ledger"""
    record = await ledger.get_job(job_id)
    if not record:
        raise HTTPException(404, f"Job not found in ledger: {job_id}")
    return record


@app.get("/ledger/merkle/{epoch_id}")
async def get_epoch_merkle(epoch_id: str):
    """Get Merkle tree for an epoch"""
    return await ledger.get_epoch_merkle(epoch_id)


# =============================================================================
# BATCH ENDPOINTS
# =============================================================================

@app.post("/batch/analyze")
async def batch_analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    patient_id: Optional[str] = None,
    priority: str = "routine"
):
    """Submit multiple studies for batch analysis"""
    job_ids = []
    
    for file in files:
        job_id = f"stinger-{uuid.uuid4().hex[:12]}"
        file_bytes = await file.read()
        
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            progress=0,
            message="Job queued",
            created_at=datetime.now(timezone.utc)
        )
        JOBS[job_id] = job
        
        request = AnalyzeRequest(
            patient_id=patient_id,
            priority=priority
        )
        
        background_tasks.add_task(
            process_study,
            job_id,
            file_bytes,
            file.filename,
            request
        )
        
        job_ids.append(job_id)
    
    return {"job_ids": job_ids, "count": len(job_ids)}


# =============================================================================
# STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    print("=" * 60)
    print("🎯 STINGER V2 — Intelligent Medical AI Gateway")
    print("   stinger.swarmbee.eth")
    print("=" * 60)
    
    # Initialize ledger
    await ledger.initialize()
    
    # Load classifier model
    await classifier.load()
    
    # Verify services
    print("\n🔌 Checking services...")
    
    try:
        await queenbee.health()
        print("   ✅ QueenBee: ONLINE")
    except Exception as e:
        print(f"   ⚠️  QueenBee: OFFLINE ({e})")
    
    try:
        await bumble.health()
        print("   ✅ Bumble70B: ONLINE")
    except Exception as e:
        print(f"   ⚠️  Bumble70B: OFFLINE ({e})")
    
    print("\n💎 Diamond Hands Mode: ENGAGED")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("\n🛑 Stinger V2 shutting down...")
    await ledger.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "stinger.main:app",
        host="0.0.0.0",
        port=8100,
        reload=True
    )

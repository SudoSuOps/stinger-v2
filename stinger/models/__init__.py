"""
Stinger V2 - Data Models
Pydantic models for API and internal use

stinger.swarmbee.eth
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# STUDY MODELS
# =============================================================================

class StudyType(str, Enum):
    SPINE = "spine"
    CARDIAC = "cardiac"
    CHEST = "chest"
    NEURO = "neuro"
    ABDOMEN = "abdomen"
    MSK = "msk"
    MAMMO = "mammo"
    ECG = "ecg"
    CGM = "cgm"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    XR = "xr"
    CT = "ct"
    MRI = "mri"
    US = "us"
    NM = "nm"
    PET = "pet"
    MAMMO = "mammo"
    ECG = "ecg"
    SIGNAL = "signal"
    TIMESERIES = "timeseries"
    UNKNOWN = "unknown"


class Study(BaseModel):
    """Medical study data model"""
    study_id: str
    study_type: StudyType
    modality: Modality
    patient_id: Optional[str] = None
    study_date: Optional[datetime] = None
    study_description: Optional[str] = None
    body_region: Optional[str] = None
    num_images: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


# =============================================================================
# JOB MODELS
# =============================================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreate(BaseModel):
    """Request to create a job"""
    patient_id: Optional[str] = None
    study_description: Optional[str] = None
    priority: str = Field(default="routine", pattern="^(stat|urgent|routine)$")
    include_pdf: bool = True
    include_proof: bool = True


class JobResponse(BaseModel):
    """Job response"""
    job_id: str
    status: JobStatus
    progress: int = 0
    message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None


# =============================================================================
# REPORT MODELS
# =============================================================================

class Finding(BaseModel):
    """Individual finding"""
    name: str
    location: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = 0.0
    description: Optional[str] = None


class Measurement(BaseModel):
    """Measurement value"""
    name: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    is_abnormal: bool = False


class Report(BaseModel):
    """Medical report"""
    report_id: str
    job_id: str
    study_type: StudyType
    
    # Content
    summary: str = ""
    findings: List[Finding] = Field(default_factory=list)
    measurements: List[Measurement] = Field(default_factory=list)
    impression: str = ""
    recommendations: List[str] = Field(default_factory=list)
    full_text: str = ""
    
    # Metadata
    generated_at: datetime
    model_used: str = ""
    confidence: float = 0.0
    processing_time_ms: int = 0
    
    # Proof
    merkle_root: Optional[str] = None
    ipfs_cid: Optional[str] = None
    signature: Optional[str] = None
    
    class Config:
        use_enum_values = True


# =============================================================================
# PROOF MODELS
# =============================================================================

class CryptoProof(BaseModel):
    """Cryptographic proof"""
    merkle_root: str
    signature: str
    signer: str
    ipfs_cid: str
    chain_id: int = 1
    timestamp: datetime


class EpochAttestation(BaseModel):
    """Epoch attestation"""
    epoch_id: str
    status: str
    started_at: datetime
    sealed_at: Optional[datetime] = None
    job_count: int
    merkle_root: Optional[str] = None
    ipfs_cid: Optional[str] = None
    signature: Optional[str] = None
    signer: Optional[str] = None


# =============================================================================
# API MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    ens: str = "stinger.swarmbee.eth"
    services: Dict[str, str] = Field(default_factory=dict)
    gpu_fleet: Dict[str, Any] = Field(default_factory=dict)
    uptime_seconds: float = 0


class AnalyzeRequest(BaseModel):
    """Request to analyze a study"""
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
    proof: Optional[CryptoProof] = None
    processing_time_ms: int
    timestamp: datetime
    epoch: str


class BatchAnalyzeRequest(BaseModel):
    """Request for batch analysis"""
    patient_id: Optional[str] = None
    priority: str = "routine"


class BatchAnalyzeResponse(BaseModel):
    """Response from batch analysis"""
    job_ids: List[str]
    count: int


# =============================================================================
# LEDGER MODELS
# =============================================================================

class LedgerStats(BaseModel):
    """Ledger statistics"""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_processing_time_ms: Optional[float] = None
    total_epochs: int
    sealed_epochs: int
    current_epoch: Optional[str] = None


class JobRecord(BaseModel):
    """Job record from ledger"""
    job_id: str
    status: str
    study_type: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None
    findings: Optional[Dict[str, Any]] = None
    proof: Optional[Dict[str, Any]] = None
    epoch_id: Optional[str] = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'StudyType',
    'Modality',
    'JobStatus',
    # Study models
    'Study',
    # Job models
    'JobCreate',
    'JobResponse',
    # Report models
    'Finding',
    'Measurement',
    'Report',
    # Proof models
    'CryptoProof',
    'EpochAttestation',
    # API models
    'HealthResponse',
    'AnalyzeRequest',
    'AnalyzeResponse',
    'BatchAnalyzeRequest',
    'BatchAnalyzeResponse',
    # Ledger models
    'LedgerStats',
    'JobRecord',
]

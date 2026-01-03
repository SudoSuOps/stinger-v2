"""
Stinger V2 - Context Assembler
Builds unified context for Bumble70B inference

stinger.swarmbee.eth
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import json


@dataclass
class PatientContext:
    """Patient-level context"""
    patient_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    clinical_history: List[str] = field(default_factory=list)
    prior_studies: List[Dict[str, Any]] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)


@dataclass 
class StudyContext:
    """Study-level context"""
    study_type: str
    modality: str
    body_region: str
    study_description: str
    series_descriptions: List[str] = field(default_factory=list)
    study_date: Optional[datetime] = None
    referring_physician: Optional[str] = None
    clinical_indication: Optional[str] = None
    technique: Optional[str] = None


@dataclass
class ImageContext:
    """Image-level context"""
    num_images: int = 0
    image_dimensions: Optional[tuple] = None
    pixel_spacing: Optional[tuple] = None
    slice_thickness: Optional[float] = None
    window_settings: Optional[Dict[str, float]] = None
    image_quality: str = "adequate"


@dataclass
class SignalContext:
    """Signal data context (ECG, CGM)"""
    signal_type: str = ""
    sampling_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    num_channels: int = 0
    data_quality: str = "adequate"
    statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class UnifiedContext:
    """Complete unified context for inference"""
    job_id: str
    timestamp: datetime
    patient: PatientContext
    study: StudyContext
    images: ImageContext
    signals: SignalContext
    classification_confidence: float
    priority: str = "routine"
    
    def to_prompt_context(self) -> str:
        """Convert to text context for prompt injection"""
        lines = []
        
        # Study info
        lines.append(f"STUDY TYPE: {self.study.study_type}")
        lines.append(f"MODALITY: {self.study.modality}")
        lines.append(f"BODY REGION: {self.study.body_region}")
        
        if self.study.study_description:
            lines.append(f"STUDY DESCRIPTION: {self.study.study_description}")
        
        if self.study.clinical_indication:
            lines.append(f"CLINICAL INDICATION: {self.study.clinical_indication}")
        
        # Patient info (if available)
        if self.patient.patient_id:
            lines.append(f"PATIENT ID: {self.patient.patient_id}")
        if self.patient.age:
            lines.append(f"AGE: {self.patient.age}")
        if self.patient.sex:
            lines.append(f"SEX: {self.patient.sex}")
        
        if self.patient.clinical_history:
            lines.append(f"CLINICAL HISTORY: {', '.join(self.patient.clinical_history)}")
        
        # Technical info
        if self.images.num_images > 0:
            lines.append(f"IMAGES: {self.images.num_images}")
            if self.images.image_dimensions:
                lines.append(f"DIMENSIONS: {self.images.image_dimensions}")
        
        # Signal info
        if self.signals.signal_type:
            lines.append(f"SIGNAL TYPE: {self.signals.signal_type}")
            if self.signals.sampling_rate:
                lines.append(f"SAMPLING RATE: {self.signals.sampling_rate} Hz")
            if self.signals.statistics:
                for key, val in self.signals.statistics.items():
                    lines.append(f"{key.upper()}: {val:.1f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "patient": {
                "patient_id": self.patient.patient_id,
                "age": self.patient.age,
                "sex": self.patient.sex,
                "clinical_history": self.patient.clinical_history,
            },
            "study": {
                "study_type": self.study.study_type,
                "modality": self.study.modality,
                "body_region": self.study.body_region,
                "study_description": self.study.study_description,
                "clinical_indication": self.study.clinical_indication,
            },
            "images": {
                "num_images": self.images.num_images,
                "dimensions": self.images.image_dimensions,
            },
            "signals": {
                "signal_type": self.signals.signal_type,
                "sampling_rate": self.signals.sampling_rate,
                "statistics": self.signals.statistics,
            },
            "classification_confidence": self.classification_confidence,
            "priority": self.priority,
        }


class ContextAssembler:
    """
    Assembles unified context from study data and classification.
    """
    
    def __init__(self):
        pass
    
    async def assemble(
        self,
        study_data: Dict[str, Any],
        classification,  # Classification object from router
        patient_id: Optional[str] = None,
        study_description: Optional[str] = None,
        job_id: str = "",
    ) -> UnifiedContext:
        """
        Assemble unified context from all available data.
        """
        # Build patient context
        patient = self._build_patient_context(study_data, patient_id)
        
        # Build study context
        study = self._build_study_context(study_data, classification, study_description)
        
        # Build image context
        images = self._build_image_context(study_data)
        
        # Build signal context
        signals = self._build_signal_context(study_data)
        
        return UnifiedContext(
            job_id=job_id,
            timestamp=datetime.now(timezone.utc),
            patient=patient,
            study=study,
            images=images,
            signals=signals,
            classification_confidence=classification.confidence,
            priority=classification.urgency,
        )
    
    def _build_patient_context(
        self, 
        study_data: Dict[str, Any],
        patient_id: Optional[str]
    ) -> PatientContext:
        """Build patient context from available data"""
        metadata = study_data.get("metadata", {})
        
        # Extract age from birth date if available
        age = None
        birth_date = metadata.get("patient_birth_date")
        if birth_date:
            try:
                # DICOM format: YYYYMMDD
                birth_year = int(str(birth_date)[:4])
                age = datetime.now().year - birth_year
            except:
                pass
        
        return PatientContext(
            patient_id=patient_id or metadata.get("patient_id"),
            age=age,
            sex=metadata.get("patient_sex"),
            clinical_history=[],  # Would come from EHR integration
            prior_studies=[],
            current_medications=[],
            allergies=[],
        )
    
    def _build_study_context(
        self,
        study_data: Dict[str, Any],
        classification,
        study_description: Optional[str]
    ) -> StudyContext:
        """Build study context"""
        metadata = study_data.get("metadata", {})
        
        # Parse study date
        study_date = None
        date_str = metadata.get("study_date")
        if date_str:
            try:
                study_date = datetime.strptime(str(date_str), "%Y%m%d")
            except:
                pass
        
        return StudyContext(
            study_type=classification.study_type.value,
            modality=classification.modality.value,
            body_region=classification.body_region.value,
            study_description=study_description or metadata.get("study_description", ""),
            series_descriptions=[metadata.get("series_description", "")],
            study_date=study_date,
            referring_physician=metadata.get("referring_physician"),
            clinical_indication=metadata.get("clinical_indication"),
            technique=metadata.get("protocol_name"),
        )
    
    def _build_image_context(self, study_data: Dict[str, Any]) -> ImageContext:
        """Build image context"""
        metadata = study_data.get("metadata", {})
        images = study_data.get("images", [])
        pixel_array = study_data.get("pixel_array")
        
        # Get dimensions
        dimensions = None
        if pixel_array is not None:
            dimensions = pixel_array.shape
        elif metadata.get("rows") and metadata.get("columns"):
            dimensions = (metadata["rows"], metadata["columns"])
        
        # Pixel spacing
        pixel_spacing = metadata.get("pixel_spacing")
        if pixel_spacing and not isinstance(pixel_spacing, tuple):
            try:
                pixel_spacing = tuple(pixel_spacing)
            except:
                pixel_spacing = None
        
        # Window settings
        window_settings = None
        if metadata.get("window_center") and metadata.get("window_width"):
            window_settings = {
                "center": float(metadata["window_center"]) if metadata["window_center"] else None,
                "width": float(metadata["window_width"]) if metadata["window_width"] else None,
            }
        
        return ImageContext(
            num_images=len(images) if images else (1 if pixel_array is not None else 0),
            image_dimensions=dimensions,
            pixel_spacing=pixel_spacing,
            slice_thickness=metadata.get("slice_thickness"),
            window_settings=window_settings,
            image_quality="adequate",
        )
    
    def _build_signal_context(self, study_data: Dict[str, Any]) -> SignalContext:
        """Build signal context for ECG/CGM data"""
        data_type = study_data.get("type", "")
        
        if data_type == "ecg":
            signal_data = study_data.get("signal_data", {})
            return SignalContext(
                signal_type="ecg",
                sampling_rate=signal_data.get("sampling_rate", 500),
                duration_seconds=signal_data.get("duration_ms", 10000) / 1000,
                num_channels=study_data.get("num_leads", 12),
                data_quality="adequate",
            )
        
        if data_type == "cgm":
            stats = study_data.get("stats", {})
            return SignalContext(
                signal_type="cgm",
                sampling_rate=None,  # CGM is typically every 5 minutes
                duration_seconds=None,
                num_channels=1,
                data_quality="adequate",
                statistics={
                    "mean_glucose": stats.get("mean", 0),
                    "std_glucose": stats.get("std", 0),
                    "time_in_range": stats.get("time_in_range", 0),
                    "below_range": stats.get("below_range", 0),
                    "above_range": stats.get("above_range", 0),
                    "num_readings": stats.get("count", 0),
                },
            )
        
        return SignalContext()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PatientContext',
    'StudyContext',
    'ImageContext',
    'SignalContext',
    'UnifiedContext',
    'ContextAssembler',
]

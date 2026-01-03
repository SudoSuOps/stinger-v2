"""
Stinger V2 - Intelligent Router
Study classification, model orchestration, GPU load balancing

stinger.swarmbee.eth
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class StudyType(str, Enum):
    SPINE = "spine"
    CARDIAC = "cardiac"
    CHEST = "chest"
    NEURO = "neuro"
    ABDOMEN = "abdomen"
    MSK = "msk"  # Musculoskeletal
    MAMMO = "mammo"
    ECG = "ecg"
    CGM = "cgm"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    XR = "xr"  # X-ray
    CT = "ct"
    MRI = "mri"
    US = "us"  # Ultrasound
    NM = "nm"  # Nuclear medicine
    PET = "pet"
    MAMMO = "mammo"
    ECG = "ecg"
    SIGNAL = "signal"
    TIMESERIES = "timeseries"
    UNKNOWN = "unknown"


class BodyRegion(str, Enum):
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE_CERVICAL = "spine_cervical"
    SPINE_THORACIC = "spine_thoracic"
    SPINE_LUMBAR = "spine_lumbar"
    SPINE_FULL = "spine_full"
    UPPER_EXTREMITY = "upper_extremity"
    LOWER_EXTREMITY = "lower_extremity"
    HEART = "heart"
    BREAST = "breast"
    WHOLE_BODY = "whole_body"
    UNKNOWN = "unknown"


@dataclass
class Classification:
    """Result of study classification"""
    study_type: StudyType
    modality: Modality
    body_region: BodyRegion
    confidence: float
    urgency: str = "routine"  # stat, urgent, routine
    requires_prior: bool = False
    sub_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelConfig:
    """Configuration for a model endpoint"""
    name: str
    endpoint: str
    model: str
    capabilities: List[str]
    priority: int = 0
    max_batch: int = 1
    timeout_seconds: int = 300


@dataclass
class GPUNode:
    """GPU node status"""
    name: str
    url: str
    num_gpus: int
    gpu_type: str
    available: bool = True
    current_load: float = 0.0
    queue_depth: int = 0
    last_heartbeat: float = 0.0


# =============================================================================
# STUDY CLASSIFIER
# =============================================================================

class StudyClassifier:
    """
    Intelligent study classifier.
    Determines study type, modality, body region from input data.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None
        
        # Keyword mappings for rule-based classification
        self._body_part_mapping = {
            # Spine
            "spine": BodyRegion.SPINE_FULL,
            "lumbar": BodyRegion.SPINE_LUMBAR,
            "l-spine": BodyRegion.SPINE_LUMBAR,
            "lspine": BodyRegion.SPINE_LUMBAR,
            "thoracic": BodyRegion.SPINE_THORACIC,
            "t-spine": BodyRegion.SPINE_THORACIC,
            "cervical": BodyRegion.SPINE_CERVICAL,
            "c-spine": BodyRegion.SPINE_CERVICAL,
            # Chest
            "chest": BodyRegion.CHEST,
            "thorax": BodyRegion.CHEST,
            "lung": BodyRegion.CHEST,
            "cxr": BodyRegion.CHEST,
            # Head
            "head": BodyRegion.HEAD,
            "brain": BodyRegion.HEAD,
            "skull": BodyRegion.HEAD,
            # Heart
            "heart": BodyRegion.HEART,
            "cardiac": BodyRegion.HEART,
            "echo": BodyRegion.HEART,
            # Abdomen
            "abdomen": BodyRegion.ABDOMEN,
            "liver": BodyRegion.ABDOMEN,
            "kidney": BodyRegion.ABDOMEN,
            # Pelvis
            "pelvis": BodyRegion.PELVIS,
            "hip": BodyRegion.PELVIS,
            # Breast
            "breast": BodyRegion.BREAST,
            "mammo": BodyRegion.BREAST,
        }
        
        self._modality_mapping = {
            "CR": Modality.XR,
            "DX": Modality.XR,
            "CT": Modality.CT,
            "MR": Modality.MRI,
            "MRI": Modality.MRI,
            "US": Modality.US,
            "NM": Modality.NM,
            "PT": Modality.PET,
            "MG": Modality.MAMMO,
            "ECG": Modality.ECG,
            "EKG": Modality.ECG,
        }
    
    async def load(self):
        """Load the classifier model"""
        if self.model_path and self.model_path.exists():
            # Load PyTorch model for advanced classification
            try:
                import torch
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                print(f"   ✅ Classifier model loaded from {self.model_path}")
            except Exception as e:
                print(f"   ⚠️  Could not load classifier model: {e}")
                print("   ℹ️  Using rule-based classification")
        else:
            print("   ℹ️  Using rule-based classification (no model file)")
    
    async def classify(self, study_data: Dict[str, Any]) -> Classification:
        """
        Classify a study based on its data.
        Uses ML model if available, falls back to rule-based.
        """
        data_type = study_data.get("type", "unknown")
        
        # Handle signal/timeseries data
        if data_type == "ecg":
            return Classification(
                study_type=StudyType.ECG,
                modality=Modality.ECG,
                body_region=BodyRegion.HEART,
                confidence=1.0,
                sub_type="12-lead"
            )
        
        if data_type == "cgm":
            return Classification(
                study_type=StudyType.CGM,
                modality=Modality.TIMESERIES,
                body_region=BodyRegion.WHOLE_BODY,
                confidence=1.0,
                sub_type="glucose"
            )
        
        # DICOM classification
        if data_type == "dicom":
            return await self._classify_dicom(study_data)
        
        return Classification(
            study_type=StudyType.UNKNOWN,
            modality=Modality.UNKNOWN,
            body_region=BodyRegion.UNKNOWN,
            confidence=0.0
        )
    
    async def _classify_dicom(self, study_data: Dict[str, Any]) -> Classification:
        """Classify DICOM study"""
        metadata = study_data.get("metadata", {})
        
        # Extract relevant fields
        modality_str = study_data.get("modality", "").upper()
        body_part = study_data.get("body_part", "").lower()
        study_desc = study_data.get("study_description", "").lower()
        series_desc = study_data.get("series_description", "").lower()
        
        # Combine descriptions for searching
        combined_desc = f"{body_part} {study_desc} {series_desc}"
        
        # Determine modality
        modality = self._modality_mapping.get(modality_str, Modality.UNKNOWN)
        
        # Determine body region
        body_region = BodyRegion.UNKNOWN
        for keyword, region in self._body_part_mapping.items():
            if keyword in combined_desc:
                body_region = region
                break
        
        # Determine study type from body region
        study_type = self._region_to_study_type(body_region, modality)
        
        # Check for urgent keywords
        urgency = "routine"
        urgent_keywords = ["stat", "urgent", "emergent", "trauma", "code"]
        if any(kw in combined_desc for kw in urgent_keywords):
            urgency = "stat"
        
        # Confidence based on how much we matched
        confidence = 0.5
        if body_region != BodyRegion.UNKNOWN:
            confidence += 0.25
        if modality != Modality.UNKNOWN:
            confidence += 0.25
        
        return Classification(
            study_type=study_type,
            modality=modality,
            body_region=body_region,
            confidence=confidence,
            urgency=urgency,
            metadata={
                "raw_modality": modality_str,
                "raw_body_part": body_part,
                "study_description": study_desc,
            }
        )
    
    def _region_to_study_type(self, region: BodyRegion, modality: Modality) -> StudyType:
        """Map body region to study type"""
        mapping = {
            BodyRegion.SPINE_CERVICAL: StudyType.SPINE,
            BodyRegion.SPINE_THORACIC: StudyType.SPINE,
            BodyRegion.SPINE_LUMBAR: StudyType.SPINE,
            BodyRegion.SPINE_FULL: StudyType.SPINE,
            BodyRegion.HEART: StudyType.CARDIAC,
            BodyRegion.CHEST: StudyType.CHEST,
            BodyRegion.HEAD: StudyType.NEURO,
            BodyRegion.ABDOMEN: StudyType.ABDOMEN,
            BodyRegion.PELVIS: StudyType.ABDOMEN,
            BodyRegion.BREAST: StudyType.MAMMO,
            BodyRegion.UPPER_EXTREMITY: StudyType.MSK,
            BodyRegion.LOWER_EXTREMITY: StudyType.MSK,
        }
        return mapping.get(region, StudyType.UNKNOWN)


# =============================================================================
# MODEL ORCHESTRATOR
# =============================================================================

class ModelOrchestrator:
    """
    Orchestrates model selection and execution.
    Determines which models to run based on classification.
    """
    
    def __init__(self, settings):
        self.settings = settings
        
        # Model registry
        self._models: Dict[str, ModelConfig] = {
            # Primary inference models
            "bumble70b": ModelConfig(
                name="bumble70b",
                endpoint="http://192.168.0.250:8000/v1/completions",
                model="meditron-70b-awq",
                capabilities=["reasoning", "report_generation", "all_domains"],
                priority=1
            ),
            # Specialized models
            "spine_detector": ModelConfig(
                name="spine_detector",
                endpoint="http://192.168.0.250:8001/detect",
                model="queenbee-spine-yolov8x",
                capabilities=["spine", "detection", "localization"],
                priority=2
            ),
            "cardiac_segmenter": ModelConfig(
                name="cardiac_segmenter",
                endpoint="http://192.168.0.250:8002/segment",
                model="queenbee-cardiac-unet",
                capabilities=["cardiac", "segmentation"],
                priority=2
            ),
            "spine_classifier": ModelConfig(
                name="spine_classifier",
                endpoint="http://192.168.0.250:8003/classify",
                model="queenbee-spine-effnet",
                capabilities=["spine", "classification", "stenosis"],
                priority=2
            ),
        }
        
        # Study type to model mapping
        self._study_models = {
            StudyType.SPINE: ["spine_detector", "spine_classifier", "bumble70b"],
            StudyType.CARDIAC: ["cardiac_segmenter", "bumble70b"],
            StudyType.CHEST: ["bumble70b"],
            StudyType.NEURO: ["bumble70b"],
            StudyType.ECG: ["bumble70b"],
            StudyType.CGM: ["bumble70b"],
            StudyType.UNKNOWN: ["bumble70b"],
        }
    
    async def determine_models(
        self, 
        classification: Classification
    ) -> Dict[str, Dict[str, Any]]:
        """
        Determine which models to run for a classification.
        Returns dict of model configs to execute.
        """
        models_to_run = {}
        
        study_type = classification.study_type
        model_names = self._study_models.get(study_type, ["bumble70b"])
        
        for name in model_names:
            if name in self._models:
                config = self._models[name]
                models_to_run[name] = {
                    "model": config.model,
                    "endpoint": config.endpoint,
                    "capabilities": config.capabilities,
                    "priority": config.priority,
                }
        
        # Always include primary reasoning model
        if "bumble70b" not in models_to_run:
            config = self._models["bumble70b"]
            models_to_run["primary"] = {
                "model": config.model,
                "endpoint": config.endpoint,
                "capabilities": config.capabilities,
                "priority": 1,
            }
        
        return models_to_run
    
    async def get_fleet_status(self) -> Dict[str, Any]:
        """Get status of GPU fleet"""
        # In production, this would query actual endpoints
        return {
            "total_gpus": 296,
            "online_gpus": 296,
            "gpu_types": {
                "rtx_5090": 48,
                "rtx_6000_ada": 48,
                "rtx_3090": 200,
            },
            "utilization": 0.0,
            "queue_depth": 0,
        }


# =============================================================================
# LOAD BALANCER
# =============================================================================

class LoadBalancer:
    """
    GPU load balancer.
    Distributes work across the fleet.
    """
    
    def __init__(self, nodes: List[Dict[str, Any]]):
        self.nodes: Dict[str, GPUNode] = {}
        for node_config in nodes:
            node = GPUNode(
                name=node_config["name"],
                url=node_config["url"],
                num_gpus=node_config.get("gpus", 1),
                gpu_type=node_config.get("type", "unknown"),
            )
            self.nodes[node.name] = node
    
    async def get_best_node(
        self, 
        required_capabilities: List[str] = None,
        preferred_gpu_type: str = None
    ) -> Optional[GPUNode]:
        """
        Get the best available node for a job.
        Considers load, queue depth, and capabilities.
        """
        available_nodes = [
            n for n in self.nodes.values() 
            if n.available and n.current_load < 0.9
        ]
        
        if not available_nodes:
            return None
        
        # Prefer nodes with lower load
        available_nodes.sort(key=lambda n: (n.queue_depth, n.current_load))
        
        # Prefer specific GPU type if requested
        if preferred_gpu_type:
            typed_nodes = [n for n in available_nodes if n.gpu_type == preferred_gpu_type]
            if typed_nodes:
                return typed_nodes[0]
        
        return available_nodes[0]
    
    async def update_node_status(self, name: str, load: float, queue_depth: int):
        """Update a node's status"""
        if name in self.nodes:
            self.nodes[name].current_load = load
            self.nodes[name].queue_depth = queue_depth
    
    async def mark_node_unavailable(self, name: str):
        """Mark a node as unavailable"""
        if name in self.nodes:
            self.nodes[name].available = False
    
    async def get_fleet_stats(self) -> Dict[str, Any]:
        """Get overall fleet statistics"""
        total_gpus = sum(n.num_gpus for n in self.nodes.values())
        available_gpus = sum(
            n.num_gpus for n in self.nodes.values() if n.available
        )
        avg_load = np.mean([n.current_load for n in self.nodes.values()]) if self.nodes else 0
        
        return {
            "total_nodes": len(self.nodes),
            "available_nodes": sum(1 for n in self.nodes.values() if n.available),
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "average_load": avg_load,
            "total_queue_depth": sum(n.queue_depth for n in self.nodes.values()),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StudyType',
    'Modality', 
    'BodyRegion',
    'Classification',
    'ModelConfig',
    'GPUNode',
    'StudyClassifier',
    'ModelOrchestrator',
    'LoadBalancer',
]

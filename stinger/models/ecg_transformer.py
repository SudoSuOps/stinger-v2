"""
ECG-Transformer: 12-Lead ECG Analysis
Diagnostic classification with QueenBee ECG Transformer

stinger.swarmbee.eth
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

import torch
import torch.nn.functional as F

# ECG Transformer model path
QUEENBEE_PATH = Path(__file__).parent.parent.parent.parent / "queenbee-llm" / "ecg_transformer"
ECG_MODEL_FILE = QUEENBEE_PATH / "model.py"


def load_ecg_transformer_class():
    """Load ECGTransformer from specific file path to avoid module conflicts"""
    spec = importlib.util.spec_from_file_location("ecg_model", ECG_MODEL_FILE)
    ecg_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ecg_model_module)
    return ecg_model_module.ECGTransformer


class ECGAnalyzer:
    """
    ECG-Transformer for 12-lead ECG analysis
    
    Multi-label classification for 5 superclasses + 44 SCP codes
    """
    
    SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
    SUPERCLASS_DESCRIPTIONS = {
        "NORM": "Normal ECG",
        "MI": "Myocardial Infarction",
        "STTC": "ST/T Change",
        "CD": "Conduction Disturbance",
        "HYP": "Hypertrophy",
    }
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loaded = False
        
        if model_path is None:
            model_path = QUEENBEE_PATH / "checkpoints" / "ecg_transformer_best.pt"
        self.model_path = Path(model_path)
    
    def load(self) -> bool:
        """Load the ECG transformer model"""
        if self.loaded:
            return True
        
        if not self.model_path.exists():
            print(f"   ⚠️  ECG model not found at {self.model_path}")
            return False
        
        try:
            ECGTransformer = load_ecg_transformer_class()

            self.model = ECGTransformer(
                num_leads=12,
                signal_length=5000,
                patch_size=50,
                embed_dim=256,
                depth=6,
                num_heads=8,
                num_superclasses=5,
                num_scp_codes=44,
            ).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.loaded = True
            print(f"   ✅ ECG Transformer loaded (AUC: {checkpoint.get('best_auc', 'N/A'):.3f})")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load ECG model: {e}")
            return False
    
    def analyze(
        self,
        signal: np.ndarray,
        sampling_rate: int = 500,
    ) -> Dict[str, Any]:
        """
        Analyze 12-lead ECG signal
        
        Args:
            signal: ECG signal [samples, 12] or [12, samples]
            sampling_rate: Sample rate in Hz
            
        Returns:
            Analysis results with classifications
        """
        if not self.loaded:
            if not self.load():
                return {"error": "Model not loaded"}
        
        # Ensure shape is [samples, 12]
        if signal.shape[0] == 12:
            signal = signal.T
        
        # Resample to 500Hz if needed
        if sampling_rate != 500:
            from scipy import signal as scipy_signal
            num_samples = int(signal.shape[0] * 500 / sampling_rate)
            signal = scipy_signal.resample(signal, num_samples, axis=0)
        
        # Normalize per-lead
        signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)
        
        # Pad/truncate to 5000 samples (10 seconds)
        if signal.shape[0] < 5000:
            signal = np.pad(signal, ((0, 5000 - signal.shape[0]), (0, 0)))
        else:
            signal = signal[:5000]
        
        # To tensor [1, 12, 5000]
        x = torch.tensor(signal.T, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            sc_logits, scp_logits = self.model(x)
            sc_probs = torch.sigmoid(sc_logits).cpu().numpy()[0]
            scp_probs = torch.sigmoid(scp_logits).cpu().numpy()[0]
        
        # Get predictions
        predictions = []
        for cls, prob in zip(self.SUPERCLASSES, sc_probs):
            if prob > 0.3:  # Lower threshold for multi-label
                predictions.append({
                    "class": cls,
                    "description": self.SUPERCLASS_DESCRIPTIONS[cls],
                    "confidence": round(float(prob), 3),
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Primary diagnosis
        primary_idx = np.argmax(sc_probs)
        primary_class = self.SUPERCLASSES[primary_idx]
        
        return {
            "primary_diagnosis": {
                "class": primary_class,
                "description": self.SUPERCLASS_DESCRIPTIONS[primary_class],
                "confidence": round(float(sc_probs[primary_idx]), 3),
            },
            "all_diagnoses": predictions,
            "superclass_probabilities": {
                cls: round(float(p), 3) 
                for cls, p in zip(self.SUPERCLASSES, sc_probs)
            },
            "is_normal": primary_class == "NORM",
            "requires_attention": primary_class in ["MI", "STTC", "CD", "HYP"] and sc_probs[primary_idx] > 0.7,
            "urgency": "stat" if primary_class == "MI" and sc_probs[primary_idx] > 0.8 else "routine",
        }
    
    def get_clinical_context(self, analysis: Dict[str, Any]) -> str:
        """Generate clinical context for LLM prompting"""
        diagnoses_str = "\n".join([
            f"  - {d['class']}: {d['description']} ({d['confidence']:.0%})"
            for d in analysis['all_diagnoses']
        ])
        
        return f"""12-LEAD ECG ANALYSIS (QueenBee ECG-Transformer)

Primary Diagnosis: {analysis['primary_diagnosis']['class']} - {analysis['primary_diagnosis']['description']}
Confidence: {analysis['primary_diagnosis']['confidence']:.0%}

All Detected Findings:
{diagnoses_str}

Status: {"NORMAL" if analysis['is_normal'] else "ABNORMAL"}
Urgency: {analysis['urgency'].upper()}
{"⚠️ REQUIRES IMMEDIATE ATTENTION" if analysis['requires_attention'] else ""}
"""


# Singleton
_analyzer: Optional[ECGAnalyzer] = None

def get_ecg_analyzer() -> ECGAnalyzer:
    """Get ECG Analyzer singleton"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ECGAnalyzer()
    return _analyzer


__all__ = ['ECGAnalyzer', 'get_ecg_analyzer']

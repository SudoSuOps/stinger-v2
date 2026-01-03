"""
CGM-Guardian: The 3AM Guardian
Glucose Forecasting & Anomaly Detection for Stinger

stinger.swarmbee.eth
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

import torch
import torch.nn.functional as F

# Add queenbee-llm to path for model import
QUEENBEE_PATH = Path(__file__).parent.parent.parent.parent / "queenbee-llm" / "cgm_transformer"
sys.path.insert(0, str(QUEENBEE_PATH))


class CGMGuardian:
    """
    The 3AM Guardian - CGM Transformer for glucose forecasting
    
    Predicts glucose 30 and 60 minutes ahead with anomaly detection.
    Watches over diabetics while they sleep.
    """
    
    # Clinical thresholds (mg/dL)
    THRESHOLDS = {
        "severe_hypo": 54,
        "hypo": 70,
        "target_low": 70,
        "target_high": 180,
        "hyper": 180,
        "severe_hyper": 250,
    }
    
    ANOMALY_CLASSES = ["SEVERE_HYPO", "HYPO", "NORMAL", "HYPER", "SEVERE_HYPER"]
    TREND_CLASSES = ["FALLING", "STABLE", "RISING"]
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loaded = False
        
        # Default model path
        if model_path is None:
            model_path = QUEENBEE_PATH / "checkpoints" / "cgm_transformer_best.pt"
        self.model_path = Path(model_path)
        
        # Normalization parameters
        self.glucose_mean = 120.0
        self.glucose_std = 40.0
    
    def load(self) -> bool:
        """Load the CGM transformer model"""
        if self.loaded:
            return True
        
        if not self.model_path.exists():
            print(f"   ⚠️  CGM model not found at {self.model_path}")
            return False
        
        try:
            from model import CGMTransformer
            
            self.model = CGMTransformer(
                input_dim=3,
                d_model=128,
                n_heads=8,
                n_layers=4,
                d_ff=256,
                dropout=0.1,
                forecast_horizon=12,
            ).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.loaded = True
            print(f"   ✅ CGM Guardian loaded (RMSE: {checkpoint.get('best_rmse', 'N/A'):.1f} mg/dL)")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load CGM model: {e}")
            return False
    
    def _prepare_input(self, glucose_history: List[float], timestamps: Optional[List[str]] = None) -> torch.Tensor:
        """
        Prepare input tensor from glucose history
        
        Args:
            glucose_history: List of glucose values (mg/dL), at least 24 values (2 hours @ 5min)
            timestamps: Optional list of ISO timestamps for time features
            
        Returns:
            Tensor of shape [1, T, 3] (glucose, hour_sin, hour_cos)
        """
        # Ensure we have enough history
        if len(glucose_history) < 24:
            # Pad with repetition of first value
            pad_len = 24 - len(glucose_history)
            glucose_history = [glucose_history[0]] * pad_len + list(glucose_history)
        
        # Take last 24 readings (2 hours)
        glucose_history = glucose_history[-24:]
        
        # Normalize glucose
        glucose_norm = [(g - self.glucose_mean) / self.glucose_std for g in glucose_history]
        
        # Generate time features (default to current time if not provided)
        if timestamps:
            from datetime import datetime
            hours = []
            for ts in timestamps[-24:]:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    hours.append(dt.hour + dt.minute / 60)
                except:
                    hours.append(12.0)  # Default to noon
        else:
            # Generate synthetic hours (assume 5-min intervals ending now)
            from datetime import datetime
            now = datetime.now()
            hours = []
            for i in range(24):
                offset_minutes = (23 - i) * 5
                hour = (now.hour + now.minute / 60 - offset_minutes / 60) % 24
                hours.append(hour)
        
        # Cyclical encoding
        hour_sin = [np.sin(2 * np.pi * h / 24) for h in hours]
        hour_cos = [np.cos(2 * np.pi * h / 24) for h in hours]
        
        # Combine features [T, 3]
        features = np.array([glucose_norm, hour_sin, hour_cos]).T
        
        # To tensor [1, T, 3]
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return x
    
    def predict(
        self,
        glucose_history: List[float],
        timestamps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict glucose and detect anomalies
        
        Args:
            glucose_history: List of glucose values in mg/dL (5-min intervals)
            timestamps: Optional ISO timestamps
            
        Returns:
            Dictionary with predictions and alerts
        """
        if not self.loaded:
            if not self.load():
                return {"error": "Model not loaded"}
        
        # Prepare input
        x = self._prepare_input(glucose_history, timestamps)
        
        # Inference
        with torch.no_grad():
            pred_30, pred_60, pred_seq, anomaly_logits, trend_logits = self.model(x)
        
        # Denormalize predictions
        pred_30_mgdl = pred_30.item() * self.glucose_std + self.glucose_mean
        pred_60_mgdl = pred_60.item() * self.glucose_std + self.glucose_mean
        pred_seq_mgdl = (pred_seq.squeeze().cpu().numpy() * self.glucose_std + self.glucose_mean).tolist()
        
        # Get class predictions
        anomaly_probs = F.softmax(anomaly_logits, dim=-1).squeeze().cpu().numpy()
        anomaly_pred = anomaly_probs.argmax()
        
        trend_probs = F.softmax(trend_logits, dim=-1).squeeze().cpu().numpy()
        trend_pred = trend_probs.argmax()
        
        # Current glucose (last reading)
        current_glucose = glucose_history[-1]
        
        # Determine if alert needed
        alert = False
        alert_level = "none"
        alert_message = None
        
        # Check predicted glucose
        if pred_30_mgdl < self.THRESHOLDS["severe_hypo"]:
            alert = True
            alert_level = "critical"
            alert_message = f"⚠️ SEVERE HYPOGLYCEMIA PREDICTED: {pred_30_mgdl:.0f} mg/dL in 30 minutes"
        elif pred_30_mgdl < self.THRESHOLDS["hypo"]:
            alert = True
            alert_level = "warning"
            alert_message = f"⚠️ Hypoglycemia predicted: {pred_30_mgdl:.0f} mg/dL in 30 minutes"
        elif pred_30_mgdl > self.THRESHOLDS["severe_hyper"]:
            alert = True
            alert_level = "warning"
            alert_message = f"⚠️ Severe hyperglycemia predicted: {pred_30_mgdl:.0f} mg/dL in 30 minutes"
        elif pred_30_mgdl > self.THRESHOLDS["hyper"]:
            alert = True
            alert_level = "info"
            alert_message = f"ℹ️ Hyperglycemia predicted: {pred_30_mgdl:.0f} mg/dL in 30 minutes"
        
        # Build response
        return {
            "current_glucose": round(current_glucose, 1),
            "predicted_30min": round(pred_30_mgdl, 1),
            "predicted_60min": round(pred_60_mgdl, 1),
            "predicted_sequence": [round(v, 1) for v in pred_seq_mgdl],
            "anomaly": {
                "class": self.ANOMALY_CLASSES[anomaly_pred],
                "confidence": round(float(anomaly_probs[anomaly_pred]), 3),
                "probabilities": {c: round(float(p), 3) for c, p in zip(self.ANOMALY_CLASSES, anomaly_probs)},
            },
            "trend": {
                "class": self.TREND_CLASSES[trend_pred],
                "confidence": round(float(trend_probs[trend_pred]), 3),
            },
            "alert": alert,
            "alert_level": alert_level,
            "alert_message": alert_message,
            "time_in_range": self._calculate_tir(glucose_history),
            "thresholds": self.THRESHOLDS,
        }
    
    def _calculate_tir(self, values: List[float]) -> Dict[str, float]:
        """Calculate Time in Range statistics"""
        if not values:
            return {"in_range": 0, "below": 0, "above": 0}
        
        below = sum(1 for v in values if v < self.THRESHOLDS["target_low"])
        above = sum(1 for v in values if v > self.THRESHOLDS["target_high"])
        in_range = len(values) - below - above
        
        return {
            "in_range": round(in_range / len(values) * 100, 1),
            "below": round(below / len(values) * 100, 1),
            "above": round(above / len(values) * 100, 1),
        }
    
    def get_clinical_context(self, prediction: Dict[str, Any]) -> str:
        """Generate clinical context string for LLM prompting"""
        return f"""CGM ANALYSIS (The 3AM Guardian)
        
Current Glucose: {prediction['current_glucose']} mg/dL
30-Minute Prediction: {prediction['predicted_30min']} mg/dL
60-Minute Prediction: {prediction['predicted_60min']} mg/dL

Trend: {prediction['trend']['class']} ({prediction['trend']['confidence']:.0%} confidence)
Anomaly Classification: {prediction['anomaly']['class']} ({prediction['anomaly']['confidence']:.0%})

Time in Range (last 2 hours):
  - In Range (70-180): {prediction['time_in_range']['in_range']}%
  - Below Range (<70): {prediction['time_in_range']['below']}%
  - Above Range (>180): {prediction['time_in_range']['above']}%

{"ALERT: " + prediction['alert_message'] if prediction['alert'] else "No immediate alerts."}
"""


# Singleton instance
_guardian: Optional[CGMGuardian] = None


def get_cgm_guardian() -> CGMGuardian:
    """Get the CGM Guardian singleton"""
    global _guardian
    if _guardian is None:
        _guardian = CGMGuardian()
    return _guardian


__all__ = ['CGMGuardian', 'get_cgm_guardian']

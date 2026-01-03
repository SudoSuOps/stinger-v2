"""
Stinger V2 - Input Parsers
DICOM, ECG, CGM intelligence

stinger.swarmbee.eth
"""

import io
import json
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np


# =============================================================================
# INPUT TYPE DETECTION
# =============================================================================

def detect_input_type(file_bytes: bytes, filename: str) -> str:
    """
    Detect the type of medical data input.
    
    Returns: 'dicom', 'ecg', 'cgm', or 'unknown'
    """
    filename_lower = filename.lower()
    
    # Check by extension first
    if filename_lower.endswith(('.dcm', '.dicom')):
        return 'dicom'
    
    if filename_lower.endswith(('.xml',)) and b'<AnnotatedECG' in file_bytes[:2000]:
        return 'ecg'
    
    if filename_lower.endswith('.scp'):
        return 'ecg'
    
    if filename_lower.endswith('.dat') and len(file_bytes) > 0:
        # Check for WFDB format
        return 'ecg'
    
    if filename_lower.endswith('.csv'):
        # Check if it's CGM data by looking for glucose-related headers
        header = file_bytes[:500].decode('utf-8', errors='ignore').lower()
        if any(kw in header for kw in ['glucose', 'cgm', 'bg', 'sugar', 'mmol', 'mg/dl']):
            return 'cgm'
    
    if filename_lower.endswith('.json'):
        try:
            data = json.loads(file_bytes.decode('utf-8'))
            if any(k in str(data).lower() for k in ['glucose', 'cgm', 'readings']):
                return 'cgm'
        except:
            pass
    
    # Check DICOM magic bytes
    if len(file_bytes) > 132:
        # DICOM files have 'DICM' at byte 128
        if file_bytes[128:132] == b'DICM':
            return 'dicom'
    
    return 'unknown'


# =============================================================================
# BASE PARSER
# =============================================================================

class BaseParser(ABC):
    """Abstract base parser for medical data"""
    
    @abstractmethod
    async def parse(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse input bytes and return structured data"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats"""
        pass


# =============================================================================
# DICOM PARSER
# =============================================================================

class DicomParser(BaseParser):
    """
    DICOM file parser with intelligence.
    Extracts metadata, pixel data, and study information.
    """
    
    def __init__(self):
        self._pydicom = None
    
    def _ensure_pydicom(self):
        """Lazy load pydicom"""
        if self._pydicom is None:
            try:
                import pydicom
                self._pydicom = pydicom
            except ImportError:
                raise ImportError("pydicom required: pip install pydicom")
    
    async def parse(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse DICOM file"""
        self._ensure_pydicom()
        
        # Read DICOM from bytes
        ds = self._pydicom.dcmread(io.BytesIO(file_bytes))
        
        # Extract metadata
        metadata = self._extract_metadata(ds)
        
        # Extract pixel array if present
        pixel_array = None
        images = []
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            # Normalize to uint8 for display
            if pixel_array is not None:
                normalized = self._normalize_pixels(pixel_array)
                images = [normalized] if len(normalized.shape) == 2 else list(normalized)
        
        return {
            "type": "dicom",
            "metadata": metadata,
            "pixel_array": pixel_array,
            "images": images,
            "modality": metadata.get("modality", "unknown"),
            "body_part": metadata.get("body_part_examined", "unknown"),
            "study_description": metadata.get("study_description", ""),
            "series_description": metadata.get("series_description", ""),
            "patient_id": metadata.get("patient_id"),
            "study_date": metadata.get("study_date"),
            "study_time": metadata.get("study_time"),
            "rows": metadata.get("rows"),
            "columns": metadata.get("columns"),
            "bits_stored": metadata.get("bits_stored"),
            "photometric_interpretation": metadata.get("photometric_interpretation"),
        }
    
    def _extract_metadata(self, ds) -> Dict[str, Any]:
        """Extract relevant DICOM metadata"""
        metadata = {}
        
        # Standard tags to extract
        tags = {
            "patient_id": (0x0010, 0x0020),
            "patient_name": (0x0010, 0x0010),
            "patient_birth_date": (0x0010, 0x0030),
            "patient_sex": (0x0010, 0x0040),
            "study_date": (0x0008, 0x0020),
            "study_time": (0x0008, 0x0030),
            "study_description": (0x0008, 0x1030),
            "series_description": (0x0008, 0x103E),
            "modality": (0x0008, 0x0060),
            "body_part_examined": (0x0018, 0x0015),
            "manufacturer": (0x0008, 0x0070),
            "institution_name": (0x0008, 0x0080),
            "rows": (0x0028, 0x0010),
            "columns": (0x0028, 0x0011),
            "bits_stored": (0x0028, 0x0101),
            "photometric_interpretation": (0x0028, 0x0004),
            "pixel_spacing": (0x0028, 0x0030),
            "slice_thickness": (0x0018, 0x0050),
            "window_center": (0x0028, 0x1050),
            "window_width": (0x0028, 0x1051),
        }
        
        for name, tag in tags.items():
            try:
                value = ds[tag].value
                if hasattr(value, 'original_string'):
                    value = str(value)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    value = list(value)
                metadata[name] = value
            except (KeyError, AttributeError):
                metadata[name] = None
        
        return metadata
    
    def _normalize_pixels(self, pixel_array: np.ndarray) -> np.ndarray:
        """Normalize pixel array to 0-255 uint8"""
        arr = pixel_array.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max() * 255
        return arr.astype(np.uint8)
    
    def get_supported_formats(self) -> List[str]:
        return ['.dcm', '.dicom', '.DCM', '.DICOM']


# =============================================================================
# ECG PARSER
# =============================================================================

class EcgParser(BaseParser):
    """
    ECG file parser.
    Supports HL7 aECG XML, SCP-ECG, WFDB, and raw formats.
    """
    
    async def parse(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse ECG file"""
        
        # Try to detect format
        if file_bytes[:5] == b'<?xml':
            return await self._parse_xml_ecg(file_bytes)
        elif file_bytes[:2] == b'\x00\x00':
            return await self._parse_scp_ecg(file_bytes)
        else:
            return await self._parse_raw_ecg(file_bytes)
    
    async def _parse_xml_ecg(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse HL7 aECG XML format"""
        import xml.etree.ElementTree as ET
        
        root = ET.fromstring(file_bytes.decode('utf-8'))
        
        # Extract basic info (simplified)
        signal_data = {
            "format": "hl7_aecg",
            "leads": {},
            "sampling_rate": 500,  # Default
            "duration_ms": 10000,  # Default 10 seconds
        }
        
        # Find sequence data
        for sequence in root.iter():
            if 'sequence' in sequence.tag.lower():
                # Extract lead data
                pass
        
        return {
            "type": "ecg",
            "signal_data": signal_data,
            "format": "hl7_aecg",
            "num_leads": 12,
            "sampling_rate": signal_data["sampling_rate"],
            "duration_ms": signal_data["duration_ms"],
        }
    
    async def _parse_scp_ecg(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse SCP-ECG format"""
        # SCP-ECG parsing (simplified)
        return {
            "type": "ecg",
            "signal_data": {"format": "scp_ecg", "raw": file_bytes},
            "format": "scp_ecg",
            "num_leads": 12,
        }
    
    async def _parse_raw_ecg(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse raw ECG data (WFDB or similar)"""
        return {
            "type": "ecg",
            "signal_data": {"format": "raw", "raw": file_bytes},
            "format": "raw",
        }
    
    def get_supported_formats(self) -> List[str]:
        return ['.xml', '.scp', '.dat', '.hea', '.ecg']


# =============================================================================
# CGM PARSER
# =============================================================================

class CgmParser(BaseParser):
    """
    CGM (Continuous Glucose Monitor) data parser.
    Supports Dexcom, Libre, and generic CSV formats.
    """
    
    async def parse(self, file_bytes: bytes) -> Dict[str, Any]:
        """Parse CGM data"""
        content = file_bytes.decode('utf-8', errors='ignore')
        
        # Try JSON first
        if content.strip().startswith('{') or content.strip().startswith('['):
            return await self._parse_json_cgm(content)
        
        # Try CSV
        return await self._parse_csv_cgm(content)
    
    async def _parse_json_cgm(self, content: str) -> Dict[str, Any]:
        """Parse JSON CGM data"""
        data = json.loads(content)
        
        readings = []
        if isinstance(data, list):
            readings = data
        elif 'readings' in data:
            readings = data['readings']
        elif 'egvs' in data:  # Dexcom format
            readings = data['egvs']
        
        return self._process_readings(readings, "json")
    
    async def _parse_csv_cgm(self, content: str) -> Dict[str, Any]:
        """Parse CSV CGM data"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return {"type": "cgm", "readings": [], "error": "No data"}
        
        header = lines[0].lower().split(',')
        
        # Find glucose and timestamp columns
        glucose_col = None
        time_col = None
        
        for i, col in enumerate(header):
            if any(kw in col for kw in ['glucose', 'bg', 'value', 'reading', 'mmol', 'mg']):
                glucose_col = i
            if any(kw in col for kw in ['time', 'date', 'timestamp']):
                time_col = i
        
        if glucose_col is None:
            glucose_col = 1  # Default assumption
        if time_col is None:
            time_col = 0
        
        readings = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) > max(glucose_col, time_col):
                try:
                    glucose = float(parts[glucose_col].strip())
                    timestamp = parts[time_col].strip()
                    readings.append({
                        "timestamp": timestamp,
                        "glucose": glucose,
                    })
                except (ValueError, IndexError):
                    continue
        
        return self._process_readings(readings, "csv")
    
    def _process_readings(self, readings: List[Dict], source_format: str) -> Dict[str, Any]:
        """Process CGM readings into structured output"""
        if not readings:
            return {
                "type": "cgm",
                "readings": [],
                "stats": {},
                "format": source_format,
            }
        
        # Extract glucose values
        glucose_values = []
        for r in readings:
            val = r.get('glucose') or r.get('value') or r.get('sgv') or r.get('bg')
            if val is not None:
                try:
                    glucose_values.append(float(val))
                except:
                    pass
        
        # Calculate statistics
        if glucose_values:
            stats = {
                "count": len(glucose_values),
                "mean": np.mean(glucose_values),
                "std": np.std(glucose_values),
                "min": np.min(glucose_values),
                "max": np.max(glucose_values),
                "time_in_range": self._calculate_tir(glucose_values),
                "below_range": sum(1 for v in glucose_values if v < 70) / len(glucose_values) * 100,
                "above_range": sum(1 for v in glucose_values if v > 180) / len(glucose_values) * 100,
            }
        else:
            stats = {}
        
        return {
            "type": "cgm",
            "readings": readings,
            "glucose_values": glucose_values,
            "stats": stats,
            "format": source_format,
            "num_readings": len(readings),
        }
    
    def _calculate_tir(self, values: List[float], low: float = 70, high: float = 180) -> float:
        """Calculate Time in Range percentage"""
        if not values:
            return 0.0
        in_range = sum(1 for v in values if low <= v <= high)
        return in_range / len(values) * 100
    
    def get_supported_formats(self) -> List[str]:
        return ['.csv', '.json', '.txt']


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'detect_input_type',
    'BaseParser',
    'DicomParser',
    'EcgParser',
    'CgmParser',
]

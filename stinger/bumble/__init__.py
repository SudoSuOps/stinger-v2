"""
Stinger V2 - Bumble70B Client
Interface to the 70B medical reasoning model

bumble.swarmbee.eth
"""

import base64
from typing import Dict, Any, Optional, List
import httpx
import numpy as np


class BumbleClient:
    """
    Client for Bumble70B inference service.
    bumble.swarmbee.eth
    
    Supports:
    - Text-only inference (clinical reasoning)
    - Image + text inference (radiology)
    - Signal + text inference (ECG)
    - Multi-modal inference (combined)
    """
    
    def __init__(
        self, 
        base_url: str = "http://192.168.0.250:8000",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def health(self) -> Dict[str, Any]:
        """Check Bumble health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def infer(
        self,
        prompt: Dict[str, str],  # {"system": ..., "user": ...}
        image_data: Optional[np.ndarray] = None,
        signal_data: Optional[Dict[str, Any]] = None,
        model: str = "meditron-70b-awq",
        endpoint: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Run inference on Bumble70B.
        
        Args:
            prompt: Dict with "system" and "user" prompts
            image_data: Optional numpy array of image pixels
            signal_data: Optional dict of signal data (ECG, CGM)
            model: Model name to use
            endpoint: Override endpoint URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dict with inference results
        """
        url = endpoint or f"{self.base_url}/v1/completions"
        
        # Build the request
        messages = []
        
        # System message
        if prompt.get("system"):
            messages.append({
                "role": "system",
                "content": prompt["system"]
            })
        
        # User message with optional multimodal content
        user_content = []
        
        # Add image if present
        if image_data is not None:
            image_b64 = self._encode_image(image_data)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                }
            })
        
        # Add signal data as text if present
        if signal_data:
            signal_text = self._format_signal_data(signal_data)
            user_content.append({
                "type": "text",
                "text": signal_text
            })
        
        # Add main user prompt
        user_content.append({
            "type": "text",
            "text": prompt.get("user", "Please analyze this study.")
        })
        
        # If only text, simplify content
        if len(user_content) == 1:
            messages.append({
                "role": "user",
                "content": user_content[0]["text"]
            })
        else:
            messages.append({
                "role": "user", 
                "content": user_content
            })
        
        # Make request
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        try:
            response = await self.client.post(url, json=request_body)
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            if "choices" in result:
                text = result["choices"][0]["message"]["content"]
            elif "text" in result:
                text = result["text"]
            elif "response" in result:
                text = result["response"]
            else:
                text = str(result)
            
            # Parse structured output
            parsed = self._parse_response(text)
            
            return {
                "text": text,
                "report": text,
                "parsed": parsed,
                "model": model,
                "tokens_used": result.get("usage", {}),
            }
            
        except httpx.HTTPStatusError as e:
            return {
                "error": f"HTTP error: {e.response.status_code}",
                "text": "",
                "report": "",
            }
        except Exception as e:
            return {
                "error": str(e),
                "text": "",
                "report": "",
            }
    
    async def infer_batch(
        self,
        prompts: List[Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run batch inference"""
        results = []
        images = images or [None] * len(prompts)
        
        for prompt, image in zip(prompts, images):
            result = await self.infer(prompt, image_data=image, **kwargs)
            results.append(result)
        
        return results
    
    def _encode_image(self, image_data: np.ndarray) -> str:
        """Encode numpy array to base64 PNG"""
        try:
            from PIL import Image
            import io
            
            # Ensure uint8
            if image_data.dtype != np.uint8:
                img_normalized = image_data.astype(np.float32)
                img_normalized = img_normalized - img_normalized.min()
                if img_normalized.max() > 0:
                    img_normalized = img_normalized / img_normalized.max() * 255
                image_data = img_normalized.astype(np.uint8)
            
            # Handle different shapes
            if len(image_data.shape) == 2:
                # Grayscale
                img = Image.fromarray(image_data, mode='L')
            elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                # RGB
                img = Image.fromarray(image_data, mode='RGB')
            else:
                # Take first slice if 3D volume
                img = Image.fromarray(image_data[0] if len(image_data.shape) == 3 else image_data, mode='L')
            
            # Encode to PNG
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except ImportError:
            # Fallback without PIL
            return base64.b64encode(image_data.tobytes()).decode('utf-8')
    
    def _format_signal_data(self, signal_data: Dict[str, Any]) -> str:
        """Format signal data as text for the prompt"""
        lines = ["SIGNAL DATA:"]
        
        if signal_data.get("format"):
            lines.append(f"Format: {signal_data['format']}")
        
        if signal_data.get("sampling_rate"):
            lines.append(f"Sampling Rate: {signal_data['sampling_rate']} Hz")
        
        # CGM statistics
        if "statistics" in signal_data:
            stats = signal_data["statistics"]
            lines.append("GLUCOSE STATISTICS:")
            for key, value in stats.items():
                lines.append(f"  {key}: {value}")
        
        # ECG leads summary
        if "leads" in signal_data:
            lines.append("ECG LEADS: " + ", ".join(signal_data["leads"].keys()))
        
        return "\n".join(lines)
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse structured data from response text"""
        parsed = {
            "summary": "",
            "findings": [],
            "impression": "",
            "recommendations": [],
        }
        
        # Simple section extraction
        sections = {
            "IMPRESSION": "impression",
            "FINDINGS": "findings_text",
            "RECOMMENDATION": "recommendations_text",
            "SUMMARY": "summary",
        }
        
        current_section = None
        current_text = []
        
        for line in text.split("\n"):
            line_upper = line.strip().upper()
            
            # Check for section headers
            for header, key in sections.items():
                if line_upper.startswith(header):
                    if current_section and current_text:
                        parsed[current_section] = "\n".join(current_text).strip()
                    current_section = key
                    current_text = []
                    # Include text after the header on the same line
                    remainder = line.split(":", 1)[-1].strip()
                    if remainder:
                        current_text.append(remainder)
                    break
            else:
                if current_section:
                    current_text.append(line)
        
        # Save last section
        if current_section and current_text:
            parsed[current_section] = "\n".join(current_text).strip()
        
        # Extract bullet points as findings
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith(("-", "•", "*", "·")):
                parsed["findings"].append(line[1:].strip())
            elif line.startswith(tuple(f"{i}." for i in range(1, 20))):
                parsed["findings"].append(line.split(".", 1)[-1].strip())
        
        return parsed
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['BumbleClient']

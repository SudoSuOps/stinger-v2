#!/usr/bin/env python3
"""
Bumble Inference Server
bumble.swarmbee.eth

OpenAI-compatible API serving medical inference via Ollama.
"""

import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

app = FastAPI(
    title="Bumble70B",
    description="Medical AI Inference Server — bumble.swarmbee.eth",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL = "meditron:70b"  # Medical 70B model

# Model aliases
MODEL_ALIASES = {
    "meditron-70b-awq": "meditron:70b",
    "meditron-70b": "meditron:70b",
    "med42-70b": "meditron:70b",
    "qwen2.5": "qwen2.5:latest",
}

class Message(BaseModel):
    role: str
    content: str | List[Dict[str, Any]]

class CompletionRequest(BaseModel):
    model: str = MODEL
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: int = 4096
    stream: bool = False

class CompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@app.get("/health")
async def health():
    """Health check"""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            ollama_status = "online" if r.status_code == 200 else "offline"
    except:
        ollama_status = "offline"

    return {
        "status": "online",
        "model": MODEL,
        "ollama": ollama_status,
        "ens": "bumble.swarmbee.eth"
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: CompletionRequest):
    """OpenAI-compatible chat completions endpoint"""

    # Convert messages to Ollama format
    prompt = ""
    for msg in request.messages:
        role = msg.role
        content = msg.content

        # Handle multimodal content
        if isinstance(content, list):
            text_parts = [p["text"] for p in content if p.get("type") == "text"]
            content = "\n".join(text_parts)

        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"

    prompt += "Assistant: "

    # Resolve model alias
    model = request.model or MODEL
    model = MODEL_ALIASES.get(model, model)

    # Call Ollama
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    }
                }
            )
            response.raise_for_status()
            data = response.json()

            return CompletionResponse(
                id=f"bumble-{int(time.time())}",
                created=int(time.time()),
                model=request.model or MODEL,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("response", "")
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            )
        except httpx.HTTPError as e:
            raise HTTPException(500, f"Ollama error: {e}")

# Legacy endpoint for compatibility
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Legacy completions endpoint"""
    return await chat_completions(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

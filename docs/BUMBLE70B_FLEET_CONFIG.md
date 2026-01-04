# Bumble70B Fleet Configuration

## Optimized for 2× RTX 5090 Rigs

**Last Updated**: January 4, 2026
**Tested Throughput**: 1,405 spine studies/hour

---

## Hardware Requirements

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | 2× RTX 5090 (32GB each) | Tensor parallel across both |
| RAM | 256GB DDR5 | Linux buffer cache helps |
| CPU | Intel Xeon w9-3475X or AMD TR | 32+ cores recommended |
| Storage | NVMe for datasets | 1TB+ free recommended |

---

## Ollama Configuration

### Service File: `/etc/systemd/system/ollama.service`

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="CUDA_VISIBLE_DEVICES=0,1"

[Install]
WantedBy=default.target
```

### Model: meditron:70b

- **Quantization**: Q4_0 (38GB on disk)
- **VRAM Usage**: ~41GB split across 2 GPUs
- **Context**: 2048 tokens (reduced from 4096 for speed)

---

## Optimized API Call Parameters

```python
{
    "model": "meditron:70b",
    "prompt": "<your_prompt>",
    "stream": False,
    "options": {
        "num_predict": 256,    # Cap output tokens
        "num_ctx": 2048,       # Reduced context
        "temperature": 0.1,    # Low for consistency
    }
}
```

### Token Cap Rationale

| Token Limit | Avg Inference | Use Case |
|-------------|---------------|----------|
| 128 | 1.9s | Ultra-brief impressions |
| 256 | 3.5s | **Recommended** - concise clinical reports |
| 512 | 7.0s | Detailed reports when needed |
| Unlimited | 14-16s | Full verbose output |

---

## Benchmark Results

### Single Rig (2× RTX 5090)

| Config | Studies/Hour | LLM Time | Notes |
|--------|-------------|----------|-------|
| Sequential, no cap | 824 | 3.4s avg | Baseline |
| Sequential, 256 cap | 1,000+ | 2.5s avg | Token cap helps |
| 2 Workers, 256 cap | **1,405** | 3.9s avg | **Production config** |
| 4 Workers, 256 cap | 1,300 | 6.0s avg | Diminishing returns |

### Pipeline Breakdown

```
DICOM Load:        950ms (NVMe storage)
Spine Detection:    25ms (YOLO on GPU)
LLM Inference:   3,900ms (meditron:70b)
PDF Generation:      3ms (reportlab)
─────────────────────────
TOTAL:           4,900ms per study
```

---

## Storage Optimization

### HuggingFace Cache Symlink

Move HF cache off root drive:

```bash
# Copy cache to NVMe
rsync -av ~/.cache/huggingface /mnt/nvme/cache/

# Replace with symlink
rm -rf ~/.cache/huggingface
ln -s /mnt/nvme/cache/huggingface ~/.cache/huggingface
```

### Ollama Models Location

Default: `/usr/share/ollama/.ollama/models/`

To move:
```bash
# Stop ollama
sudo systemctl stop ollama

# Move models
sudo mv /usr/share/ollama/.ollama /mnt/nvme/ollama_models

# Symlink back
sudo ln -s /mnt/nvme/ollama_models /usr/share/ollama/.ollama

# Restart
sudo systemctl start ollama
```

---

## Fleet Scaling

### 296 GPU Fleet Projection

| Rigs | GPUs | Config | Projected Throughput |
|------|------|--------|---------------------|
| 1 | 2× 5090 | 2 workers | 1,405/hour |
| 10 | 20× 5090 | 2 workers each | 14,050/hour |
| 50 | 100× 5090 | 2 workers each | 70,250/hour |
| 148 | 296× 5090 | 2 workers each | 207,740/hour |

**At full fleet: 207K+ spine studies per hour**

---

## Quick Deploy Commands

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull meditron
ollama pull meditron:70b

# Create optimized model (optional)
cat << 'EOF' > /tmp/bumble70b.Modelfile
FROM meditron:70b
PARAMETER num_predict 256
PARAMETER num_ctx 2048
PARAMETER temperature 0.1
EOF
ollama create bumble70b-fast -f /tmp/bumble70b.Modelfile

# Enable service
sudo systemctl enable ollama
sudo systemctl start ollama

# Verify
curl http://localhost:11434/api/tags
```

---

## Monitoring

```bash
# GPU utilization
nvidia-smi -l 1

# Ollama logs
journalctl -u ollama -f

# Throughput test
python benchmark_spine_parallel.py -n 20 -w 2
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM on single GPU | Model needs both GPUs, check CUDA_VISIBLE_DEVICES |
| Slow inference | Check num_predict cap, reduce num_ctx |
| High latency variance | Normal - depends on output length |
| Root drive full | Move ~/.cache/huggingface to NVMe |

---

*stinger.swarmbee.eth | trustcat.ai*

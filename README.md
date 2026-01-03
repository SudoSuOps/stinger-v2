# 🎯 STINGER V2 — Intelligent Medical AI Gateway

```
stinger.swarmbee.eth
```

**The sovereign gateway to TrustCat's medical AI infrastructure.**

End-to-end pipeline: `Client → Stinger → QueenBee → Bumble70B → PDF → Client`

💎 **Diamond Hands Edition** — No shortcuts. No jeets.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                         ┌─────────────┐                            │
│                         │   CLIENT    │                            │
│                         │  (Upload)   │                            │
│                         └──────┬──────┘                            │
│                                │                                   │
│                                ▼                                   │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │                     STINGER V2                              │  │
│   │              stinger.swarmbee.eth                          │  │
│   │                                                            │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │  │
│   │  │  DICOM   │ │   ECG    │ │   CGM    │ │  ROUTER  │      │  │
│   │  │ PARSER   │ │ PARSER   │ │ PARSER   │ │  LOGIC   │      │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │  │
│   └───────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│                               ▼                                   │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │                     QUEENBEE                                │  │
│   │              queenbee.swarmbee.eth                         │  │
│   │                                                            │  │
│   │                    15 GOLD PROMPTS                         │  │
│   │     Spine │ Cardiac │ Chest │ Neuro │ ECG │ CGM            │  │
│   └───────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│                               ▼                                   │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │                     BUMBLE70B                               │  │
│   │              bumble.swarmbee.eth                           │  │
│   │                                                            │  │
│   │           MEDITRON-70B + Domain LoRAs                      │  │
│   │           Attending-Level Clinical Reasoning               │  │
│   └───────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│                               ▼                                   │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │              PDF + MERKLE + IPFS + LEDGER                   │  │
│   └───────────────────────────┬────────────────────────────────┘  │
│                               │                                   │
│                               ▼                                   │
│                         ┌─────────────┐                           │
│                         │   CLIENT    │                           │
│                         │  (Report)   │                           │
│                         └─────────────┘                           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Input Processing
- **DICOM Parser** — Full DICOM intelligence with metadata extraction
- **ECG Parser** — HL7 aECG, SCP-ECG, WFDB support
- **CGM Parser** — Dexcom, Libre, CSV formats

### Intelligent Routing
- **Study Classifier** — ML-powered study type detection
- **Model Orchestrator** — Multi-model coordination
- **Load Balancer** — GPU fleet distribution

### Gold Prompts
- **Spine** — ACR-compliant radiological interpretation
- **Cardiac** — ASE/SCMR guidelines
- **Chest** — Fleischner Society guidelines
- **ECG** — AHA/ACC 12-lead analysis
- **CGM** — ADA Standards of Care

### Cryptographic Proof
- **Merkle Trees** — Job attestation
- **EIP-191 Signing** — Ethereum-compatible signatures
- **IPFS Pinning** — Decentralized storage

### Reporting
- **PDF Generation** — Clinical-grade reports with TrustCat branding
- **SwarmPool Ledger** — Job recording and epoch management

---

## 🚀 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/swarmhealth/stinger-v2.git
cd stinger-v2

# Install
pip install -e ".[full]"

# Or with Docker
docker-compose up -d
```

### Run Server

```bash
# Development
uvicorn stinger.main:app --host 0.0.0.0 --port 8100 --reload

# Production
uvicorn stinger.main:app --host 0.0.0.0 --port 8100 --workers 4
```

### Run Tests

```bash
# All tests
python -m pytest tests/

# E2E test
python tests/test_e2e.py
```

---

## 📡 API Endpoints

### Health Check
```bash
GET /
```

### Analyze Study
```bash
POST /analyze
Content-Type: multipart/form-data

file: <medical_file>
patient_id: "PATIENT001"
priority: "routine"
include_pdf: true
include_proof: true
```

### Get Job Status
```bash
GET /job/{job_id}
```

### Download Report
```bash
GET /reports/{filename}
```

### Ledger Endpoints
```bash
GET /ledger/epoch          # Current epoch
GET /ledger/job/{job_id}   # Job record
GET /ledger/merkle/{epoch} # Merkle proof
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Service URLs
QUEENBEE_URL=http://192.168.0.52:8200
BUMBLE_URL=http://192.168.0.250:8000
IPFS_GATEWAY=http://localhost:5001

# Paths
OUTPUT_DIR=/opt/stinger/outputs
LEDGER_DB_PATH=/opt/stinger/ledger.db

# Crypto
SIGNER_PRIVATE_KEY=0x...

# Fleet
MAX_CONCURRENT_JOBS=10
JOB_TIMEOUT_SECONDS=300
```

---

## 🏠 Network Architecture

| Service | ENS Domain | Port | Description |
|---------|-----------|------|-------------|
| Stinger | stinger.swarmbee.eth | 8100 | Gateway |
| QueenBee | queenbee.swarmbee.eth | 8200 | Prompt orchestrator |
| Bumble | bumble.swarmbee.eth | 8000 | 70B inference |
| SwarmPool | swarmpool.swarmbee.eth | - | Job ledger |
| Merlin | merlin.swarmbee.eth | - | Air-gapped signer |

---

## 📊 Supported Study Types

| Type | Modalities | Gold Prompt |
|------|------------|-------------|
| Spine | XR, CT, MRI | ACR guidelines |
| Cardiac | Echo, MRI, CT | ASE/SCMR |
| Chest | XR, CT | Fleischner |
| Neuro | CT, MRI | ASNR |
| ECG | 12-lead | AHA/ACC |
| CGM | Time series | ADA |

---

## 🔐 Security

- **Air-gapped signing** via Merlin
- **EIP-191** compliant signatures
- **Merkle proofs** for all jobs
- **IPFS pinning** for immutability
- **Ethereum L1** settlement ready

---

## 📈 Performance

- **Parsing**: <100ms per study
- **Classification**: <50ms
- **Inference**: ~26s (Bumble70B)
- **PDF Generation**: <500ms
- **Total Pipeline**: ~30s typical

---

## 🐝 Part of the SwarmOS Ecosystem

```
SwarmOS (swarmos.eth)
├── Stinger (stinger.swarmbee.eth) — Gateway ← YOU ARE HERE
├── QueenBee (queenbee.swarmbee.eth) — Prompts
├── Bumble (bumble.swarmbee.eth) — 70B Inference
├── SwarmPool (swarmpool.swarmbee.eth) — Ledger
└── Merlin (merlin.swarmbee.eth) — Signer
```

---

## 📜 License

Apache 2.0

---

## 🏢 Built by TrustCat

**Sovereign Medical AI Infrastructure**

- 🌐 [trustcat.ai](https://trustcat.ai)
- 🐙 [github.com/swarmhealth](https://github.com/swarmhealth)
- 🔗 [swarmos.eth.limo](https://swarmos.eth.limo)

---

💎 **Diamond Hands. Full Stack. No Jeets.** 💎

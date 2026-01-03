"""
STINGER V2 — The Intelligent Medical AI Gateway
stinger.swarmbee.eth

End-to-end sovereign medical AI pipeline:
Client → Stinger → QueenBee → Bumble70B → PDF → Client

Part of the TrustCat sovereign medical AI infrastructure.
"""

__version__ = "2.0.0"
__author__ = "TrustCat"
__ens__ = "stinger.swarmbee.eth"

from stinger.config import Settings, settings
from stinger.parsers import DicomParser, EcgParser, CgmParser, detect_input_type
from stinger.router import StudyClassifier, ModelOrchestrator, LoadBalancer, Classification
from stinger.context import ContextAssembler, UnifiedContext
from stinger.queenbee import QueenBeeClient, GOLD_PROMPTS, get_gold_prompt
from stinger.bumble import BumbleClient
from stinger.reports import PDFReportGenerator
from stinger.crypto import MerkleTree, EIP191Signer, IPFSClient
from stinger.ledger import SwarmPoolLedger, Job, JobStatus, Epoch, EpochStatus

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__ens__',
    # Config
    'Settings',
    'settings',
    # Parsers
    'DicomParser',
    'EcgParser', 
    'CgmParser',
    'detect_input_type',
    # Router
    'StudyClassifier',
    'ModelOrchestrator',
    'LoadBalancer',
    'Classification',
    # Context
    'ContextAssembler',
    'UnifiedContext',
    # QueenBee
    'QueenBeeClient',
    'GOLD_PROMPTS',
    'get_gold_prompt',
    # Bumble
    'BumbleClient',
    # Reports
    'PDFReportGenerator',
    # Crypto
    'MerkleTree',
    'EIP191Signer',
    'IPFSClient',
    # Ledger
    'SwarmPoolLedger',
    'Job',
    'JobStatus',
    'Epoch',
    'EpochStatus',
]

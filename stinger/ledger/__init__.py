"""
Stinger V2 - SwarmPool Ledger
Job recording, epoch management, audit trail

swarmpool.swarmbee.eth
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import sqlite3


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"
    CLASSIFYING = "classifying"
    ASSEMBLING = "assembling"
    PROMPTING = "prompting"
    INFERRING = "inferring"
    EXTRACTING = "extracting"
    GENERATING = "generating"
    SIGNING = "signing"
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"


class EpochStatus(str, Enum):
    OPEN = "open"
    SEALING = "sealing"
    SEALED = "sealed"


@dataclass
class Job:
    """Job record"""
    job_id: str
    status: JobStatus
    progress: int
    message: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    study_type: Optional[str] = None
    findings: Optional[Dict[str, Any]] = None
    proof: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    result: Any = None
    
    def update(
        self,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Any = None,
    ):
        """Update job status"""
        if status is not None:
            self.status = status
        if progress is not None:
            self.progress = progress
        if message is not None:
            self.message = message
        if result is not None:
            self.result = result
        if status == JobStatus.COMPLETED:
            self.completed_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "study_type": self.study_type,
            "findings": self.findings,
            "proof": self.proof,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class Epoch:
    """Epoch record"""
    epoch_id: str
    status: EpochStatus
    started_at: datetime
    sealed_at: Optional[datetime] = None
    job_count: int = 0
    merkle_root: Optional[str] = None
    ipfs_cid: Optional[str] = None
    signature: Optional[str] = None
    signer: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "sealed_at": self.sealed_at.isoformat() if self.sealed_at else None,
            "job_count": self.job_count,
            "merkle_root": self.merkle_root,
            "ipfs_cid": self.ipfs_cid,
            "signature": self.signature,
            "signer": self.signer,
        }


# =============================================================================
# SWARMPOOL LEDGER
# =============================================================================

class SwarmPoolLedger:
    """
    SwarmPool job ledger with SQLite backend.
    Records all jobs, manages epochs, provides audit trail.
    
    swarmpool.swarmbee.eth
    """
    
    def __init__(self, db_path: Path = Path("ledger.db")):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._current_epoch: Optional[Epoch] = None
        self._epoch_counter = 0
    
    async def initialize(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                study_type TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                processing_time_ms INTEGER,
                findings TEXT,
                proof TEXT,
                epoch_id TEXT,
                FOREIGN KEY (epoch_id) REFERENCES epochs(epoch_id)
            );
            
            CREATE TABLE IF NOT EXISTS epochs (
                epoch_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                sealed_at TEXT,
                job_count INTEGER DEFAULT 0,
                merkle_root TEXT,
                ipfs_cid TEXT,
                signature TEXT,
                signer TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_jobs_epoch ON jobs(epoch_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_epochs_status ON epochs(status);
        """)
        self.conn.commit()
        
        # Initialize or get current epoch
        await self._ensure_current_epoch()
        
        print(f"   ✅ Ledger initialized: {self.db_path}")
        print(f"   ✅ Current epoch: {self._current_epoch.epoch_id}")
    
    async def health(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "status": "online" if self.conn else "offline",
            "current_epoch": self._current_epoch.epoch_id if self._current_epoch else None,
            "db_path": str(self.db_path),
        }
    
    async def _ensure_current_epoch(self):
        """Ensure there's an open epoch"""
        cursor = self.conn.execute(
            "SELECT * FROM epochs WHERE status = ? ORDER BY started_at DESC LIMIT 1",
            (EpochStatus.OPEN.value,)
        )
        row = cursor.fetchone()
        
        if row:
            self._current_epoch = Epoch(
                epoch_id=row["epoch_id"],
                status=EpochStatus(row["status"]),
                started_at=datetime.fromisoformat(row["started_at"]),
                job_count=row["job_count"],
            )
        else:
            # Create new epoch
            await self._create_new_epoch()
    
    async def _create_new_epoch(self) -> Epoch:
        """Create a new epoch"""
        # Get epoch number
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM epochs")
        count = cursor.fetchone()["count"]
        
        # Generate epoch ID (e.g., epoch-0001-alpha)
        epoch_num = count + 1
        nato = self._get_nato_letter(epoch_num)
        epoch_id = f"epoch-{epoch_num:04d}-{nato}"
        
        now = datetime.now(timezone.utc)
        
        self.conn.execute(
            """INSERT INTO epochs (epoch_id, status, started_at, job_count)
               VALUES (?, ?, ?, ?)""",
            (epoch_id, EpochStatus.OPEN.value, now.isoformat(), 0)
        )
        self.conn.commit()
        
        self._current_epoch = Epoch(
            epoch_id=epoch_id,
            status=EpochStatus.OPEN,
            started_at=now,
            job_count=0,
        )
        
        return self._current_epoch
    
    def _get_nato_letter(self, num: int) -> str:
        """Get NATO phonetic alphabet word for epoch number"""
        nato = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet", "kilo", "lima",
            "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango", "uniform", "victor", "whiskey", "xray",
            "yankee", "zulu"
        ]
        return nato[(num - 1) % len(nato)]
    
    async def record_job(
        self,
        job_id: str,
        study_type: str,
        status: str,
        findings: Optional[Dict[str, Any]] = None,
        proof: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[int] = None,
    ):
        """Record a completed job"""
        now = datetime.now(timezone.utc)
        
        self.conn.execute(
            """INSERT OR REPLACE INTO jobs 
               (job_id, status, study_type, created_at, completed_at, 
                processing_time_ms, findings, proof, epoch_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                status,
                study_type,
                now.isoformat(),
                now.isoformat() if status == "completed" else None,
                processing_time_ms,
                json.dumps(findings) if findings else None,
                json.dumps(proof) if proof else None,
                self._current_epoch.epoch_id if self._current_epoch else None,
            )
        )
        
        # Update epoch job count
        if self._current_epoch:
            self.conn.execute(
                "UPDATE epochs SET job_count = job_count + 1 WHERE epoch_id = ?",
                (self._current_epoch.epoch_id,)
            )
            self._current_epoch.job_count += 1
        
        self.conn.commit()
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job record"""
        cursor = self.conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                "job_id": row["job_id"],
                "status": row["status"],
                "study_type": row["study_type"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "processing_time_ms": row["processing_time_ms"],
                "findings": json.loads(row["findings"]) if row["findings"] else None,
                "proof": json.loads(row["proof"]) if row["proof"] else None,
                "epoch_id": row["epoch_id"],
            }
        return None
    
    async def get_current_epoch(self) -> str:
        """Get current epoch ID"""
        return self._current_epoch.epoch_id if self._current_epoch else "unknown"
    
    async def get_epoch_info(self) -> Dict[str, Any]:
        """Get current epoch information"""
        if not self._current_epoch:
            return {"error": "No current epoch"}
        
        # Get job stats
        cursor = self.conn.execute(
            """SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(processing_time_ms) as avg_time_ms
               FROM jobs WHERE epoch_id = ?""",
            (self._current_epoch.epoch_id,)
        )
        stats = cursor.fetchone()
        
        return {
            **self._current_epoch.to_dict(),
            "stats": {
                "total_jobs": stats["total"] or 0,
                "completed": stats["completed"] or 0,
                "failed": stats["failed"] or 0,
                "avg_processing_time_ms": stats["avg_time_ms"],
            }
        }
    
    async def get_epoch_jobs(self, epoch_id: str) -> List[Dict[str, Any]]:
        """Get all jobs in an epoch"""
        cursor = self.conn.execute(
            "SELECT * FROM jobs WHERE epoch_id = ? ORDER BY created_at",
            (epoch_id,)
        )
        
        jobs = []
        for row in cursor.fetchall():
            jobs.append({
                "job_id": row["job_id"],
                "status": row["status"],
                "study_type": row["study_type"],
                "created_at": row["created_at"],
                "processing_time_ms": row["processing_time_ms"],
            })
        
        return jobs
    
    async def get_epoch_merkle(self, epoch_id: str) -> Dict[str, Any]:
        """Get Merkle tree info for an epoch"""
        cursor = self.conn.execute(
            "SELECT * FROM epochs WHERE epoch_id = ?",
            (epoch_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return {"error": "Epoch not found"}
        
        return {
            "epoch_id": row["epoch_id"],
            "merkle_root": row["merkle_root"],
            "ipfs_cid": row["ipfs_cid"],
            "signature": row["signature"],
            "signer": row["signer"],
            "job_count": row["job_count"],
        }
    
    async def seal_epoch(
        self,
        merkle_root: str,
        ipfs_cid: str,
        signature: str,
        signer: str,
    ):
        """Seal the current epoch and start a new one"""
        if not self._current_epoch:
            return
        
        now = datetime.now(timezone.utc)
        
        # Update epoch
        self.conn.execute(
            """UPDATE epochs 
               SET status = ?, sealed_at = ?, merkle_root = ?, 
                   ipfs_cid = ?, signature = ?, signer = ?
               WHERE epoch_id = ?""",
            (
                EpochStatus.SEALED.value,
                now.isoformat(),
                merkle_root,
                ipfs_cid,
                signature,
                signer,
                self._current_epoch.epoch_id,
            )
        )
        self.conn.commit()
        
        # Create new epoch
        await self._create_new_epoch()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get overall ledger statistics"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(processing_time_ms) as avg_time_ms,
                COUNT(DISTINCT epoch_id) as total_epochs
            FROM jobs
        """)
        stats = cursor.fetchone()
        
        cursor = self.conn.execute(
            "SELECT COUNT(*) as sealed FROM epochs WHERE status = ?",
            (EpochStatus.SEALED.value,)
        )
        sealed = cursor.fetchone()
        
        return {
            "total_jobs": stats["total_jobs"] or 0,
            "completed_jobs": stats["completed"] or 0,
            "failed_jobs": stats["failed"] or 0,
            "avg_processing_time_ms": stats["avg_time_ms"],
            "total_epochs": stats["total_epochs"] or 0,
            "sealed_epochs": sealed["sealed"] or 0,
            "current_epoch": self._current_epoch.epoch_id if self._current_epoch else None,
        }
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'JobStatus',
    'EpochStatus',
    'Job',
    'Epoch',
    'SwarmPoolLedger',
]

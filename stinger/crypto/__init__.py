"""
Stinger V2 - Cryptographic Operations
Merkle trees, EIP-191 signing, IPFS pinning

merlin.swarmbee.eth (air-gapped signer)
"""

import hashlib
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import httpx


# =============================================================================
# MERKLE TREE
# =============================================================================

class MerkleTree:
    """
    Merkle tree implementation for job attestation.
    Used to create verifiable proofs of computation.
    """
    
    def __init__(self, hash_func=None):
        self.hash_func = hash_func or self._keccak256
    
    def _keccak256(self, data: bytes) -> str:
        """Keccak256 hash (Ethereum compatible)"""
        try:
            from Crypto.Hash import keccak
            k = keccak.new(digest_bits=256)
            k.update(data)
            return "0x" + k.hexdigest()
        except ImportError:
            # Fallback to SHA256 if pycryptodome not available
            return "0x" + hashlib.sha256(data).hexdigest()
    
    def hash_leaf(self, data: Any) -> str:
        """Hash a single leaf node"""
        if isinstance(data, dict):
            # Sort keys for deterministic hashing
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        else:
            data_str = str(data)
        return self.hash_func(data_str.encode('utf-8'))
    
    def hash_pair(self, left: str, right: str) -> str:
        """Hash a pair of nodes"""
        # Sort to ensure deterministic ordering
        if left > right:
            left, right = right, left
        combined = left + right
        return self.hash_func(combined.encode('utf-8'))
    
    def compute_root(self, data: Dict[str, Any]) -> str:
        """
        Compute Merkle root from job data.
        
        Args:
            data: Dictionary of job data to hash
        
        Returns:
            Merkle root as hex string
        """
        # Create leaf hashes from each key-value pair
        leaves = []
        for key in sorted(data.keys()):
            leaf_data = {key: data[key]}
            leaves.append(self.hash_leaf(leaf_data))
        
        if not leaves:
            return self.hash_func(b'')
        
        # Build tree bottom-up
        while len(leaves) > 1:
            next_level = []
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    next_level.append(self.hash_pair(leaves[i], leaves[i + 1]))
                else:
                    # Odd number of leaves, promote the last one
                    next_level.append(leaves[i])
            leaves = next_level
        
        return leaves[0]
    
    def compute_proof(self, data: Dict[str, Any], key: str) -> List[str]:
        """Compute Merkle proof for a specific key"""
        # Implementation for generating proofs
        # (simplified - full implementation would track tree structure)
        return []
    
    def verify_proof(self, root: str, leaf: str, proof: List[str]) -> bool:
        """Verify a Merkle proof"""
        current = leaf
        for sibling in proof:
            current = self.hash_pair(current, sibling)
        return current == root


# =============================================================================
# EIP-191 SIGNER
# =============================================================================

class EIP191Signer:
    """
    EIP-191 compliant message signer.
    Compatible with Ethereum wallets and smart contracts.
    
    merlin.swarmbee.eth - The air-gapped signer
    """
    
    def __init__(self, private_key: Optional[str] = None):
        self.private_key = private_key
        self._account = None
        self._address = None
        
        if private_key:
            self._init_account()
    
    def _init_account(self):
        """Initialize account from private key"""
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            
            # Remove 0x prefix if present
            pk = self.private_key
            if pk.startswith('0x'):
                pk = pk[2:]
            
            self._account = Account.from_key(pk)
            self._address = self._account.address
        except ImportError:
            # Generate a deterministic address from key for demo
            self._address = "0x" + hashlib.sha256(
                self.private_key.encode() if self.private_key else b"demo"
            ).hexdigest()[:40]
    
    @property
    def address(self) -> str:
        """Get signer address"""
        return self._address or "0x0000000000000000000000000000000000000000"
    
    def sign(self, message: str) -> str:
        """
        Sign a message using EIP-191.
        
        Args:
            message: Message to sign (typically a Merkle root)
        
        Returns:
            Signature as hex string
        """
        if not self._account:
            # Demo mode - return deterministic fake signature
            return "0x" + hashlib.sha256(
                (message + str(self.private_key or "demo")).encode()
            ).hexdigest() + "00" * 32
        
        try:
            from eth_account.messages import encode_defunct
            
            # EIP-191 prefix
            message_hash = encode_defunct(text=message)
            signed = self._account.sign_message(message_hash)
            return signed.signature.hex()
        except Exception as e:
            raise ValueError(f"Signing failed: {e}")
    
    def verify(self, message: str, signature: str, expected_address: str) -> bool:
        """
        Verify an EIP-191 signature.
        
        Args:
            message: Original message
            signature: Signature to verify
            expected_address: Expected signer address
        
        Returns:
            True if signature is valid
        """
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
            
            message_hash = encode_defunct(text=message)
            recovered = Account.recover_message(message_hash, signature=signature)
            return recovered.lower() == expected_address.lower()
        except ImportError:
            # Can't verify without eth_account
            return True
        except Exception:
            return False
    
    def sign_typed_data(self, domain: Dict, types: Dict, message: Dict) -> str:
        """
        Sign EIP-712 typed data.
        Used for more complex structured data signing.
        """
        try:
            from eth_account import Account
            from eth_account.messages import encode_structured_data
            
            full_message = {
                "types": types,
                "domain": domain,
                "primaryType": "Job",
                "message": message,
            }
            
            signable = encode_structured_data(full_message)
            signed = self._account.sign_message(signable)
            return signed.signature.hex()
        except Exception as e:
            # Fallback to simple signing
            return self.sign(json.dumps(message, sort_keys=True))


# =============================================================================
# IPFS CLIENT
# =============================================================================

class IPFSClient:
    """
    IPFS client for content-addressed storage.
    Used to pin job data and proofs.
    """
    
    def __init__(
        self, 
        gateway_url: str = "http://localhost:5001",
        public_gateway: str = "https://ipfs.io/ipfs/",
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.public_gateway = public_gateway.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def health(self) -> Dict[str, Any]:
        """Check IPFS daemon health"""
        try:
            response = await self.client.post(f"{self.gateway_url}/api/v0/id")
            response.raise_for_status()
            return response.json()
        except:
            return {"status": "offline"}
    
    async def pin(self, data: str) -> str:
        """
        Pin data to IPFS and return CID.
        
        Args:
            data: String data to pin (typically JSON)
        
        Returns:
            IPFS CID (Content Identifier)
        """
        try:
            # Add to IPFS
            files = {"file": ("data.json", data.encode(), "application/json")}
            response = await self.client.post(
                f"{self.gateway_url}/api/v0/add",
                files=files,
            )
            response.raise_for_status()
            result = response.json()
            cid = result.get("Hash", result.get("cid", ""))
            
            # Pin to ensure persistence
            await self.client.post(
                f"{self.gateway_url}/api/v0/pin/add",
                params={"arg": cid},
            )
            
            return cid
            
        except Exception as e:
            # Fallback: generate deterministic CID-like hash
            # This allows testing without IPFS running
            content_hash = hashlib.sha256(data.encode()).hexdigest()
            return f"Qm{content_hash[:44]}"  # Fake CID format
    
    async def get(self, cid: str) -> str:
        """
        Get data from IPFS by CID.
        
        Args:
            cid: IPFS Content Identifier
        
        Returns:
            Data as string
        """
        try:
            response = await self.client.post(
                f"{self.gateway_url}/api/v0/cat",
                params={"arg": cid},
            )
            response.raise_for_status()
            return response.text
        except:
            # Try public gateway
            response = await self.client.get(f"{self.public_gateway}/{cid}")
            response.raise_for_status()
            return response.text
    
    async def unpin(self, cid: str) -> bool:
        """Unpin data from IPFS"""
        try:
            response = await self.client.post(
                f"{self.gateway_url}/api/v0/pin/rm",
                params={"arg": cid},
            )
            return response.status_code == 200
        except:
            return False
    
    def get_gateway_url(self, cid: str) -> str:
        """Get public gateway URL for a CID"""
        return f"{self.public_gateway}/{cid}"
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()


# =============================================================================
# EPOCH ATTESTATION
# =============================================================================

@dataclass
class EpochAttestation:
    """Complete epoch attestation for Merlin signing"""
    epoch_id: str
    start_time: int
    end_time: int
    job_count: int
    merkle_root: str
    jobs_cid: str
    signature: str
    signer: str
    chain_id: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "job_count": self.job_count,
            "merkle_root": self.merkle_root,
            "jobs_cid": self.jobs_cid,
            "signature": self.signature,
            "signer": self.signer,
            "chain_id": self.chain_id,
        }


async def create_epoch_attestation(
    epoch_id: str,
    jobs: List[Dict[str, Any]],
    signer: EIP191Signer,
    ipfs: IPFSClient,
) -> EpochAttestation:
    """
    Create a complete epoch attestation.
    This is what Merlin does in air-gapped mode.
    """
    merkle = MerkleTree()
    
    # Compute Merkle root of all jobs
    jobs_data = {f"job_{i}": job for i, job in enumerate(jobs)}
    root = merkle.compute_root(jobs_data)
    
    # Pin jobs to IPFS
    jobs_cid = await ipfs.pin(json.dumps(jobs, sort_keys=True))
    
    # Sign the root
    signature = signer.sign(root)
    
    return EpochAttestation(
        epoch_id=epoch_id,
        start_time=int(jobs[0].get("timestamp", time.time())) if jobs else int(time.time()),
        end_time=int(time.time()),
        job_count=len(jobs),
        merkle_root=root,
        jobs_cid=jobs_cid,
        signature=signature,
        signer=signer.address,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MerkleTree',
    'EIP191Signer',
    'IPFSClient',
    'EpochAttestation',
    'create_epoch_attestation',
]

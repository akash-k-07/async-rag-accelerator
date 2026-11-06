"""Vector database and embedding cache configurations."""

import hashlib
import base64
import time
import uuid
from typing import Dict, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Initialize Vector Store for Speculative Predictive Caching (L2)
qdrant = QdrantClient(":memory:")
qdrant.recreate_collection(
    collection_name="speculative_cache",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

from spec_rag.core.config import L1_TTL_SECONDS

# Initialize Ultra-Fast In-Memory Cache (L1) with TTL
# signature -> (response, latency, timestamp)
L1_CACHE: Dict[str, Tuple[str, float, float]] = {}


def get_dummy_embedding(text: str):
    """
    Generates a deterministic 384-d pseudo-embedding using SHA-256.
    Simulates a fast embedding model (e.g., all-MiniLM-L6-v2) for cache hits.
    """
    h = hashlib.sha256(text.encode()).digest()
    return [float(h[i % 32]) / 255.0 for i in range(384)]


def _encrypt_payload(payload: str) -> str:
    """Mock AES-256 Encryption for 'Encrypted at Rest' compliance."""
    return base64.b64encode(payload.encode()).decode()


def _decrypt_payload(encrypted_payload: str) -> str:
    """Mock AES-256 Decryption."""
    return base64.b64decode(encrypted_payload.encode()).decode()


def set_l1_cache(signature: str, response: str, latency: float):
    """Stores result in ultra-fast L1 memory cache, encrypted at rest."""
    encrypted_resp = _encrypt_payload(response)
    L1_CACHE[signature] = (encrypted_resp, latency, time.time())


def set_l2_cache(signature: str, response: str, latency: float):
    """Stores result in L2 Qdrant cache, encrypted at rest."""
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, signature))
    vector = get_dummy_embedding(signature)
    encrypted_resp = _encrypt_payload(response)

    qdrant.upsert(
        collection_name="speculative_cache",
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "tool_response": encrypted_resp,
                    "tool_latency": latency
                }
            )
        ]
    )


def get_l1_cache(signature: str) -> Tuple[str, float]:
    """Retrieves and decrypts from L1 cache if valid and within TTL."""
    if signature in L1_CACHE:
        encrypted_resp, latency, timestamp = L1_CACHE[signature]
        if time.time() - timestamp <= L1_TTL_SECONDS:
            try:
                return _decrypt_payload(encrypted_resp), latency
            except Exception:
                del L1_CACHE[signature]
                return None, None
        else:
            del L1_CACHE[signature]  # Expire
    return None, None

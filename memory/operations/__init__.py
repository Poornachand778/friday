"""
Memory Operations
=================

Higher-level operations for memory processing.

Operations:
    - TripletExtractor: Extract knowledge triplets from text
    - DecayDaemon: Background decay processing (Ebbinghaus-inspired)
    - Consolidation: STM → LTM consolidation
"""

from memory.operations.triplet_extractor import (
    TripletExtractor,
    ExtractedTriplet,
    extract_triplets,
)

from memory.operations.decay import (
    DecayDaemon,
    DecayConfig,
    DecayCycleResult,
    run_decay,
)

__all__ = [
    # Triplet extraction
    "TripletExtractor",
    "ExtractedTriplet",
    "extract_triplets",
    # Decay algorithm
    "DecayDaemon",
    "DecayConfig",
    "DecayCycleResult",
    "run_decay",
]

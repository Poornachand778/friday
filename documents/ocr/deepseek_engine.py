"""
DeepSeek-OCR 2 Engine Wrapper

Handles model loading, GPU management, and inference for document OCR.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Union

from documents.config import OCRConfig, get_document_config
from documents.models import OCRResult

LOGGER = logging.getLogger(__name__)


class DeepSeekOCR:
    """
    DeepSeek-OCR 2 wrapper for document understanding.

    Handles:
    - Model loading with configurable quantization
    - GPU memory management
    - Batch processing for efficiency
    - Graceful fallback when model unavailable
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or get_document_config().ocr
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._device = self.config.device

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

    async def load_model(self) -> bool:
        """
        Load DeepSeek-OCR 2 model with specified quantization.

        Returns True if successful, False otherwise.
        """
        if self._loaded:
            return True

        try:
            LOGGER.info(
                "Loading DeepSeek-OCR 2 from %s (quantization: %s)",
                self.config.model_path,
                self.config.quantization,
            )

            # Import here to avoid startup overhead
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Determine torch dtype based on quantization
            torch_dtype = torch.bfloat16
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch_dtype,
            }

            # Add flash attention if available
            if self.config.use_flash_attention:
                try:
                    load_kwargs["_attn_implementation"] = "flash_attention_2"
                except Exception:
                    LOGGER.warning("Flash attention not available, using default")

            # Load model with quantization
            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["quantization_config"] = quantization_config
            elif self.config.quantization == "8bit":
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["quantization_config"] = quantization_config

            # Load model and tokenizer
            self._model = AutoModel.from_pretrained(
                self.config.model_path, **load_kwargs
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, trust_remote_code=True
            )

            # Move to device if not using quantization
            if self.config.quantization == "none" and self._device == "cuda":
                self._model = self._model.cuda()

            self._model.eval()
            self._loaded = True

            LOGGER.info("DeepSeek-OCR 2 loaded successfully")
            return True

        except ImportError as e:
            LOGGER.error(
                "Missing dependencies for DeepSeek-OCR 2: %s. "
                "Install with: pip install torch transformers flash-attn",
                e,
            )
            return False
        except Exception as e:
            LOGGER.error("Failed to load DeepSeek-OCR 2: %s", e)
            return False

    async def unload_model(self) -> None:
        """Unload model to free GPU memory"""
        if not self._loaded:
            return

        import gc

        import torch

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOGGER.info("DeepSeek-OCR 2 unloaded, GPU memory freed")

    async def process_image(
        self,
        image_path: Union[str, Path],
        language_hint: Optional[str] = None,
        grounding: bool = True,
    ) -> OCRResult:
        """
        Process a single image through DeepSeek-OCR 2.

        Args:
            image_path: Path to the image file
            language_hint: Optional language hint (en, te, mixed)
            grounding: If True, use grounding mode for layout preservation

        Returns:
            OCRResult with extracted text and confidence
        """
        import time

        start_time = time.time()

        if not self._loaded:
            loaded = await self.load_model()
            if not loaded:
                return await self._fallback_process(image_path)

        try:
            # Build prompt
            if grounding:
                prompt = "<image>\n<|grounding|>Convert the document to markdown"
            else:
                prompt = "<image>\nFree OCR"

            if language_hint == "te":
                prompt += " (Telugu text present)"

            # Load and process image
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # Resize if too large
            max_dim = self.config.max_image_dimension
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Run inference
            result = self._model.infer(
                self._tokenizer,
                prompt=prompt,
                image=image,
            )

            processing_time = int((time.time() - start_time) * 1000)

            # Parse result for confidence and metadata
            text = result if isinstance(result, str) else str(result)

            return OCRResult(
                text=text,
                confidence=0.9,  # DeepSeek doesn't return confidence, use high default
                has_images="![" in text or "<img" in text,
                has_tables="|" in text and "---" in text,
                detected_headers=self._extract_headers(text),
                model_used="deepseek-ocr-2",
                processing_time_ms=processing_time,
            )

        except Exception as e:
            LOGGER.error("OCR processing failed for %s: %s", image_path, e)
            return await self._fallback_process(image_path)

    async def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        language_hint: Optional[str] = None,
    ) -> List[OCRResult]:
        """
        Process multiple images in sequence.

        Note: True batch processing depends on DeepSeek model capabilities.
        Currently processes sequentially with model reuse.
        """
        if not self._loaded:
            loaded = await self.load_model()
            if not loaded:
                return [await self._fallback_process(p) for p in image_paths]

        results = []
        for i, path in enumerate(image_paths):
            LOGGER.debug("Processing image %d/%d: %s", i + 1, len(image_paths), path)
            result = await self.process_image(path, language_hint)
            results.append(result)

            # Small delay to prevent GPU throttling
            if i < len(image_paths) - 1:
                await asyncio.sleep(0.1)

        return results

    async def _fallback_process(self, image_path: Union[str, Path]) -> OCRResult:
        """
        Fallback processing when DeepSeek model unavailable.

        Uses basic text extraction or API fallback.
        """
        if self.config.fallback_enabled:
            LOGGER.info("Using fallback OCR for %s", image_path)
            return await self._api_fallback(image_path)

        # Return empty result if no fallback
        return OCRResult(
            text="[OCR UNAVAILABLE - DeepSeek model not loaded]",
            confidence=0.0,
            model_used="none",
        )

    async def _api_fallback(self, image_path: Union[str, Path]) -> OCRResult:
        """
        API-based fallback OCR.

        Supports Google Cloud Vision, Azure, etc.
        """
        # Placeholder for API fallback implementation
        # Would use google-cloud-vision or azure-ai-vision
        LOGGER.warning(
            "API fallback not implemented. Set up %s API for fallback.",
            self.config.fallback_provider,
        )
        return OCRResult(
            text=f"[OCR FALLBACK NOT CONFIGURED - Provider: {self.config.fallback_provider}]",
            confidence=0.0,
            model_used=f"fallback-{self.config.fallback_provider}",
        )

    def _extract_headers(self, text: str) -> List[str]:
        """Extract markdown headers from OCR output"""
        import re

        headers = []
        for line in text.split("\n"):
            # Markdown headers
            if line.startswith("#"):
                header = line.lstrip("#").strip()
                if header:
                    headers.append(header)
            # Bold text as potential headers
            elif line.startswith("**") and line.endswith("**"):
                header = line.strip("*").strip()
                if header and len(header) < 100:
                    headers.append(header)

        return headers[:10]  # Limit to 10 headers


class MockDeepSeekOCR(DeepSeekOCR):
    """
    Mock OCR for testing without GPU.

    Returns placeholder text for development.
    """

    async def load_model(self) -> bool:
        self._loaded = True
        LOGGER.info("MockDeepSeekOCR initialized (no GPU required)")
        return True

    async def process_image(
        self,
        image_path: Union[str, Path],
        language_hint: Optional[str] = None,
        grounding: bool = True,
    ) -> OCRResult:
        """Return mock OCR result"""
        path = Path(image_path)
        return OCRResult(
            text=f"# Mock OCR Output\n\nProcessed: {path.name}\n\nThis is placeholder text for testing without GPU.\n\nThe actual DeepSeek-OCR 2 model would extract real content here.",
            confidence=1.0,
            has_images=False,
            has_tables=False,
            detected_headers=["Mock OCR Output"],
            model_used="mock",
            processing_time_ms=10,
        )

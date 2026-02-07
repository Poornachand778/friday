"""
Tests for DeepSeekOCR and MockDeepSeekOCR
==========================================

Comprehensive tests for the DeepSeek-OCR 2 engine wrapper including
model loading with quantization, GPU memory management, image processing,
batch processing, fallback behavior, header extraction, and the mock class.

Run with: pytest tests/test_deepseek_ocr.py -v
"""

import asyncio
import gc
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.config import OCRConfig
from documents.models import OCRResult


# =========================================================================
# Helpers / Fixtures
# =========================================================================


def make_ocr_config(**overrides) -> OCRConfig:
    """Create an OCRConfig with sensible test defaults."""
    defaults = dict(
        model_path="deepseek-ai/DeepSeek-OCR-2",
        use_flash_attention=False,
        quantization="none",
        max_batch_size=4,
        device="cpu",
        timeout_seconds=60.0,
        image_dpi=300,
        max_image_dimension=2048,
        fallback_enabled=True,
        fallback_provider="google",
        fallback_api_key="",
        retry_on_low_confidence=True,
        low_confidence_threshold=0.5,
    )
    defaults.update(overrides)
    return OCRConfig(**defaults)


def _build_mock_torch(cuda_available=False):
    """Build a mock torch module."""
    mock_torch = MagicMock(spec=[])
    mock_torch.__name__ = "torch"
    mock_torch.bfloat16 = "bfloat16"
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = cuda_available
    mock_cuda.empty_cache = MagicMock()
    mock_torch.cuda = mock_cuda
    return mock_torch


def _build_mock_transformers():
    """Build a mock transformers module with AutoModel, AutoTokenizer, BitsAndBytesConfig."""
    mock_transformers = MagicMock(spec=[])
    mock_transformers.__name__ = "transformers"

    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.cuda = MagicMock(return_value=mock_model)
    mock_model.infer = MagicMock(return_value="# Heading\n\nSome text content.")

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained = MagicMock(return_value=mock_model)
    mock_transformers.AutoModel = mock_auto_model

    mock_tokenizer = MagicMock()
    mock_auto_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)
    mock_transformers.AutoTokenizer = mock_auto_tokenizer

    mock_bnb_config = MagicMock()
    mock_transformers.BitsAndBytesConfig = mock_bnb_config

    return mock_transformers, mock_model, mock_tokenizer


def _build_mock_pil_image_module():
    """Build a mock PIL.Image module for use with patch.dict on sys.modules.

    Returns (mock_image_module, mock_image_instance).
    The mock_image_module stands in for ``PIL.Image`` so that
    ``from PIL import Image`` inside ``process_image`` resolves correctly
    when we do ``patch.dict("sys.modules", {"PIL.Image": mock_image_module})``.
    """
    mock_image_module = MagicMock()

    mock_image = MagicMock()
    mock_image.size = (1000, 800)
    mock_image.width = 1000
    mock_image.height = 800
    mock_image.convert = MagicMock(return_value=mock_image)
    mock_image.resize = MagicMock(return_value=mock_image)

    mock_image_module.open = MagicMock(return_value=mock_image)
    mock_image_module.Resampling = MagicMock()
    mock_image_module.Resampling.LANCZOS = "LANCZOS"

    return mock_image_module, mock_image


@pytest.fixture
def ocr_config():
    """Return a basic OCRConfig for tests."""
    return make_ocr_config()


@pytest.fixture
def ocr_config_4bit():
    return make_ocr_config(quantization="4bit")


@pytest.fixture
def ocr_config_8bit():
    return make_ocr_config(quantization="8bit")


@pytest.fixture
def ocr_config_flash():
    return make_ocr_config(use_flash_attention=True)


@pytest.fixture
def ocr_config_cuda():
    return make_ocr_config(device="cuda", quantization="none")


@pytest.fixture
def ocr_config_no_fallback():
    return make_ocr_config(fallback_enabled=False)


def _make_engine(config=None):
    """Create a DeepSeekOCR instance, mocking get_document_config."""
    if config is None:
        config = make_ocr_config()
    with patch("documents.ocr.deepseek_engine.get_document_config") as mock_gdc:
        mock_doc_config = MagicMock()
        mock_doc_config.ocr = config
        mock_gdc.return_value = mock_doc_config
        from documents.ocr.deepseek_engine import DeepSeekOCR

        engine = DeepSeekOCR(config)
    return engine


def _make_engine_default_config():
    """Create a DeepSeekOCR with no config arg (exercises default path)."""
    with patch("documents.ocr.deepseek_engine.get_document_config") as mock_gdc:
        mock_doc_config = MagicMock()
        mock_doc_config.ocr = make_ocr_config()
        mock_gdc.return_value = mock_doc_config
        from documents.ocr.deepseek_engine import DeepSeekOCR

        engine = DeepSeekOCR()
    return engine


def _pil_patch(mock_image_module):
    """Return a context-manager that patches PIL.Image in sys.modules."""
    return patch.dict("sys.modules", {"PIL.Image": mock_image_module})


# =========================================================================
# 1. DeepSeekOCR __init__
# =========================================================================


class TestDeepSeekOCRInit:
    """Tests for DeepSeekOCR.__init__"""

    def test_init_with_custom_config(self, ocr_config):
        """DeepSeekOCR accepts a custom OCRConfig."""
        engine = _make_engine(ocr_config)
        assert engine.config is ocr_config
        assert engine._model is None
        assert engine._tokenizer is None
        assert engine._loaded is False
        assert engine._device == ocr_config.device

    def test_init_default_config_via_singleton(self):
        """DeepSeekOCR falls back to get_document_config() when no config given."""
        engine = _make_engine_default_config()
        assert engine.config is not None
        assert isinstance(engine.config, OCRConfig)
        assert engine._loaded is False

    def test_init_sets_device_from_config(self):
        """Device is read from config."""
        config = make_ocr_config(device="mps")
        engine = _make_engine(config)
        assert engine._device == "mps"

    def test_init_model_none(self, ocr_config):
        """Model starts as None."""
        engine = _make_engine(ocr_config)
        assert engine._model is None

    def test_init_tokenizer_none(self, ocr_config):
        """Tokenizer starts as None."""
        engine = _make_engine(ocr_config)
        assert engine._tokenizer is None


# =========================================================================
# 2. is_loaded property
# =========================================================================


class TestIsLoaded:
    """Tests for the is_loaded property."""

    def test_is_loaded_initially_false(self, ocr_config):
        engine = _make_engine(ocr_config)
        assert engine.is_loaded is False

    def test_is_loaded_true_after_flag_set(self, ocr_config):
        engine = _make_engine(ocr_config)
        engine._loaded = True
        assert engine.is_loaded is True

    def test_is_loaded_false_after_reset(self, ocr_config):
        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._loaded = False
        assert engine.is_loaded is False


# =========================================================================
# 3. load_model
# =========================================================================


class TestLoadModel:
    """Tests for load_model."""

    @pytest.mark.asyncio
    async def test_load_model_success(self, ocr_config):
        """load_model returns True and sets _loaded on success."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        assert engine._loaded is True
        assert engine._model is not None
        assert engine._tokenizer is not None

    @pytest.mark.asyncio
    async def test_load_model_already_loaded_returns_true(self, ocr_config):
        """If already loaded, load_model returns True immediately."""
        engine = _make_engine(ocr_config)
        engine._loaded = True

        result = await engine.load_model()
        assert result is True

    @pytest.mark.asyncio
    async def test_load_model_import_error_returns_false(self, ocr_config):
        """ImportError during load returns False."""
        engine = _make_engine(ocr_config)

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def failing_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = await engine.load_model()

        assert result is False
        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_load_model_general_exception_returns_false(self, ocr_config):
        """General exception during load returns False."""
        mock_torch = _build_mock_torch()
        mock_transformers, _, _ = _build_mock_transformers()
        mock_transformers.AutoModel.from_pretrained.side_effect = RuntimeError(
            "GPU OOM"
        )

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is False
        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_load_model_4bit_quantization(self, ocr_config_4bit):
        """4bit quantization creates BitsAndBytesConfig correctly."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config_4bit)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        mock_transformers.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    @pytest.mark.asyncio
    async def test_load_model_8bit_quantization(self, ocr_config_8bit):
        """8bit quantization creates BitsAndBytesConfig correctly."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config_8bit)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        mock_transformers.BitsAndBytesConfig.assert_called_once_with(
            load_in_8bit=True,
        )

    @pytest.mark.asyncio
    async def test_load_model_none_quantization_no_bnb(self, ocr_config):
        """With quantization='none', BitsAndBytesConfig is not used."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config)
        assert engine.config.quantization == "none"

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        mock_transformers.BitsAndBytesConfig.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_none_quantization_cuda_moves_to_gpu(
        self, ocr_config_cuda
    ):
        """With quantization='none' and device='cuda', model.cuda() is called."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config_cuda)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        mock_model.cuda.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_none_quantization_cpu_no_cuda_call(self, ocr_config):
        """With quantization='none' and device='cpu', model.cuda() is NOT called."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        mock_model.cuda.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_flash_attention_enabled(self, ocr_config_flash):
        """Flash attention adds _attn_implementation to load_kwargs."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config_flash)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        call_kwargs = mock_transformers.AutoModel.from_pretrained.call_args
        assert call_kwargs[1].get("_attn_implementation") == "flash_attention_2"

    @pytest.mark.asyncio
    async def test_load_model_no_flash_attention(self, ocr_config):
        """Without flash attention, _attn_implementation is not in kwargs."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            result = await engine.load_model()

        assert result is True
        call_kwargs = mock_transformers.AutoModel.from_pretrained.call_args
        assert "_attn_implementation" not in call_kwargs[1]

    @pytest.mark.asyncio
    async def test_load_model_calls_eval(self, ocr_config):
        """Model.eval() is called after loading."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            await engine.load_model()

        mock_model.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_trust_remote_code(self, ocr_config):
        """trust_remote_code=True is passed to from_pretrained."""
        mock_torch = _build_mock_torch()
        mock_transformers, _, _ = _build_mock_transformers()

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            await engine.load_model()

        model_call_kwargs = mock_transformers.AutoModel.from_pretrained.call_args[1]
        assert model_call_kwargs["trust_remote_code"] is True

        tokenizer_call_kwargs = (
            mock_transformers.AutoTokenizer.from_pretrained.call_args[1]
        )
        assert tokenizer_call_kwargs["trust_remote_code"] is True

    @pytest.mark.asyncio
    async def test_load_model_uses_bfloat16_dtype(self, ocr_config):
        """torch_dtype is set to bfloat16."""
        mock_torch = _build_mock_torch()
        mock_transformers, _, _ = _build_mock_transformers()

        engine = _make_engine(ocr_config)

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            await engine.load_model()

        call_kwargs = mock_transformers.AutoModel.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == "bfloat16"


# =========================================================================
# 4. unload_model
# =========================================================================


class TestUnloadModel:
    """Tests for unload_model."""

    @pytest.mark.asyncio
    async def test_unload_sets_loaded_false(self, ocr_config):
        """unload_model sets _loaded to False."""
        mock_torch = _build_mock_torch(cuda_available=False)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect") as mock_gc:
                await engine.unload_model()

        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_unload_clears_model(self, ocr_config):
        """unload_model sets _model to None."""
        mock_torch = _build_mock_torch(cuda_available=False)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect"):
                await engine.unload_model()

        assert engine._model is None

    @pytest.mark.asyncio
    async def test_unload_clears_tokenizer(self, ocr_config):
        """unload_model sets _tokenizer to None."""
        mock_torch = _build_mock_torch(cuda_available=False)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect"):
                await engine.unload_model()

        assert engine._tokenizer is None

    @pytest.mark.asyncio
    async def test_unload_calls_gc_collect(self, ocr_config):
        """unload_model calls gc.collect()."""
        mock_torch = _build_mock_torch(cuda_available=False)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect") as mock_gc:
                await engine.unload_model()

        mock_gc.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_calls_cuda_empty_cache_when_available(self, ocr_config):
        """unload_model calls torch.cuda.empty_cache() when CUDA is available."""
        mock_torch = _build_mock_torch(cuda_available=True)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect"):
                await engine.unload_model()

        mock_torch.cuda.empty_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_unload_skips_cuda_empty_cache_when_unavailable(self, ocr_config):
        """unload_model does not call empty_cache when CUDA is unavailable."""
        mock_torch = _build_mock_torch(cuda_available=False)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect"):
                await engine.unload_model()

        mock_torch.cuda.empty_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_unload_noop_when_not_loaded(self, ocr_config):
        """unload_model does nothing when model is not loaded."""
        engine = _make_engine(ocr_config)
        engine._loaded = False

        # Should not raise or import anything
        await engine.unload_model()
        assert engine._loaded is False

    @pytest.mark.asyncio
    async def test_unload_then_is_loaded_false(self, ocr_config):
        """After unload, is_loaded returns False."""
        mock_torch = _build_mock_torch(cuda_available=False)

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("gc.collect"):
                await engine.unload_model()

        assert engine.is_loaded is False


# =========================================================================
# 5. process_image
# =========================================================================


class TestProcessImage:
    """Tests for process_image."""

    @pytest.mark.asyncio
    async def test_process_image_auto_loads_model(self, ocr_config):
        """process_image auto-loads the model when not loaded."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, mock_tokenizer = _build_mock_transformers()
        mock_image_module, mock_image = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        assert engine._loaded is False

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "PIL.Image": mock_image_module,
            },
        ):
            result = await engine.process_image("/test/image.png")

        assert engine._loaded is True
        assert isinstance(result, OCRResult)

    @pytest.mark.asyncio
    async def test_process_image_grounding_prompt(self, ocr_config):
        """With grounding=True, prompt contains grounding tag."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="# Heading\n\nText.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/image.png", grounding=True)

        call_kwargs = mock_model.infer.call_args[1]
        assert "<|grounding|>" in call_kwargs["prompt"]
        assert "Convert the document to markdown" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_process_image_free_ocr_prompt(self, ocr_config):
        """With grounding=False, prompt uses Free OCR."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/image.png", grounding=False)

        call_kwargs = mock_model.infer.call_args[1]
        assert "Free OCR" in call_kwargs["prompt"]
        assert "<|grounding|>" not in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_process_image_telugu_language_hint(self, ocr_config):
        """Telugu language hint appends note to prompt."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/image.png", language_hint="te")

        call_kwargs = mock_model.infer.call_args[1]
        assert "(Telugu text present)" in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_process_image_english_no_telugu_hint(self, ocr_config):
        """English language hint does not add Telugu note."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/image.png", language_hint="en")

        call_kwargs = mock_model.infer.call_args[1]
        assert "(Telugu text present)" not in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_process_image_resizes_large_images(self):
        """Images larger than max_image_dimension are resized."""
        config = make_ocr_config(max_image_dimension=1024)
        mock_image_module, mock_image = _build_mock_pil_image_module()
        # Large image: 3000x2000
        mock_image.size = (3000, 2000)
        mock_image.width = 3000
        mock_image.height = 2000

        engine = _make_engine(config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/large.png")

        mock_image.resize.assert_called_once()
        resize_args = mock_image.resize.call_args[0]
        new_size = resize_args[0]
        # ratio = 1024 / 3000 ~ 0.341
        assert new_size[0] < 3000
        assert new_size[1] < 2000

    @pytest.mark.asyncio
    async def test_process_image_no_resize_small_images(self):
        """Images within max_image_dimension are not resized."""
        config = make_ocr_config(max_image_dimension=2048)
        mock_image_module, mock_image = _build_mock_pil_image_module()
        mock_image.size = (800, 600)
        mock_image.width = 800
        mock_image.height = 600

        engine = _make_engine(config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/small.png")

        mock_image.resize.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_image_returns_ocr_result(self, ocr_config):
        """process_image returns a valid OCRResult."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="# Title\n\nParagraph text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert isinstance(result, OCRResult)
        assert result.text == "# Title\n\nParagraph text."
        assert result.confidence == 0.9
        assert result.model_used == "deepseek-ocr-2"

    @pytest.mark.asyncio
    async def test_process_image_detects_images_in_output(self, ocr_config):
        """has_images is True when output contains markdown image syntax."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(
            return_value="Text with ![alt](image.png) embedded."
        )
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert result.has_images is True

    @pytest.mark.asyncio
    async def test_process_image_detects_img_tag(self, ocr_config):
        """has_images is True when output contains <img tag."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(
            return_value='Some text <img src="photo.jpg"> more text'
        )
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert result.has_images is True

    @pytest.mark.asyncio
    async def test_process_image_no_images_detected(self, ocr_config):
        """has_images is False when output contains no image markers."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Just plain text without images.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert result.has_images is False

    @pytest.mark.asyncio
    async def test_process_image_detects_tables(self, ocr_config):
        """has_tables is True when output contains | and ---."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(
            return_value="| Col1 | Col2 |\n| --- | --- |\n| A | B |"
        )
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert result.has_tables is True

    @pytest.mark.asyncio
    async def test_process_image_no_tables_detected(self, ocr_config):
        """has_tables is False when no table markers present."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="No tables here, just text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert result.has_tables is False

    @pytest.mark.asyncio
    async def test_process_image_extracts_headers(self, ocr_config):
        """detected_headers is populated from markdown headers."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(
            return_value="# Chapter 1\n\nText\n\n## Section A\n\nMore text"
        )
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert "Chapter 1" in result.detected_headers
        assert "Section A" in result.detected_headers

    @pytest.mark.asyncio
    async def test_process_image_fallback_on_exception(self, ocr_config):
        """process_image falls back when inference raises an exception."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._model.infer.side_effect = RuntimeError("Inference failed")
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        # Should get a fallback result (API fallback since fallback_enabled=True)
        assert isinstance(result, OCRResult)
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_process_image_fallback_on_load_failure(self, ocr_config):
        """process_image falls back when auto-load fails."""
        engine = _make_engine(ocr_config)
        engine._loaded = False

        with patch.object(
            engine, "load_model", new_callable=AsyncMock, return_value=False
        ):
            result = await engine.process_image("/test/image.png")

        assert isinstance(result, OCRResult)
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_process_image_non_string_result_converted(self, ocr_config):
        """Non-string inference results are converted to str."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value={"output": "some data"})
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert isinstance(result.text, str)

    @pytest.mark.asyncio
    async def test_process_image_processing_time_set(self, ocr_config):
        """processing_time_ms is set to a non-negative value."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            result = await engine.process_image("/test/image.png")

        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_process_image_converts_to_rgb(self, ocr_config):
        """Image is converted to RGB."""
        mock_image_module, mock_image = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/image.png")

        # open() returns mock_image, then .convert("RGB") is called on it
        # but since open returns mock_image and convert returns mock_image,
        # we just check convert was called
        mock_image_module.open.assert_called_once()
        raw_img = mock_image_module.open.return_value
        raw_img.convert.assert_called_with("RGB")

    @pytest.mark.asyncio
    async def test_process_image_resize_uses_lanczos(self):
        """Resize uses LANCZOS resampling."""
        config = make_ocr_config(max_image_dimension=512)
        mock_image_module, mock_image = _build_mock_pil_image_module()
        mock_image.size = (2000, 1500)
        mock_image.width = 2000
        mock_image.height = 1500

        engine = _make_engine(config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            await engine.process_image("/test/image.png")

        resize_args = mock_image.resize.call_args
        # Second positional arg should be the resampling method
        assert resize_args[0][1] == mock_image_module.Resampling.LANCZOS


# =========================================================================
# 6. process_batch
# =========================================================================


class TestProcessBatch:
    """Tests for process_batch."""

    @pytest.mark.asyncio
    async def test_process_batch_processes_all_images(self, ocr_config):
        """process_batch returns results for all images."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        paths = ["/test/img1.png", "/test/img2.png", "/test/img3.png"]

        with _pil_patch(mock_image_module):
            results = await engine.process_batch(paths)

        assert len(results) == 3
        assert all(isinstance(r, OCRResult) for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_returns_list(self, ocr_config):
        """process_batch returns a list."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            results = await engine.process_batch(["/test/img1.png"])

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_process_batch_auto_loads_model(self, ocr_config):
        """process_batch auto-loads the model when not loaded."""
        mock_torch = _build_mock_torch()
        mock_transformers, mock_model, _ = _build_mock_transformers()
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        assert engine._loaded is False

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
                "PIL.Image": mock_image_module,
            },
        ):
            results = await engine.process_batch(["/test/img1.png"])

        assert engine._loaded is True
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_process_batch_fallback_on_load_failure(self, ocr_config):
        """process_batch returns fallback results when load fails."""
        engine = _make_engine(ocr_config)
        engine._loaded = False

        with patch.object(
            engine, "load_model", new_callable=AsyncMock, return_value=False
        ):
            results = await engine.process_batch(["/test/img1.png", "/test/img2.png"])

        assert len(results) == 2
        assert all(r.confidence == 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_empty_list(self, ocr_config):
        """process_batch with empty list returns empty results."""
        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        results = await engine.process_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_passes_language_hint(self, ocr_config):
        """process_batch passes language_hint to process_image."""
        engine = _make_engine(ocr_config)
        engine._loaded = True
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()

        with patch.object(
            engine,
            "process_image",
            new_callable=AsyncMock,
            return_value=OCRResult(text="test", confidence=0.9),
        ) as mock_pi:
            await engine.process_batch(["/test/img1.png"], language_hint="te")

        mock_pi.assert_called_once_with("/test/img1.png", "te")

    @pytest.mark.asyncio
    async def test_process_batch_single_image(self, ocr_config):
        """process_batch with a single image works correctly."""
        mock_image_module, _ = _build_mock_pil_image_module()

        engine = _make_engine(ocr_config)
        engine._loaded = True
        mock_model = MagicMock()
        mock_model.infer = MagicMock(return_value="Text.")
        engine._model = mock_model
        engine._tokenizer = MagicMock()

        with _pil_patch(mock_image_module):
            results = await engine.process_batch(["/test/only.png"])

        assert len(results) == 1
        assert isinstance(results[0], OCRResult)


# =========================================================================
# 7. _fallback_process
# =========================================================================


class TestFallbackProcess:
    """Tests for _fallback_process."""

    @pytest.mark.asyncio
    async def test_fallback_enabled_calls_api_fallback(self, ocr_config):
        """When fallback_enabled, _fallback_process calls _api_fallback."""
        engine = _make_engine(ocr_config)
        assert engine.config.fallback_enabled is True

        with patch.object(
            engine,
            "_api_fallback",
            new_callable=AsyncMock,
            return_value=OCRResult(text="fallback", confidence=0.0),
        ) as mock_api:
            result = await engine._fallback_process("/test/image.png")

        mock_api.assert_called_once_with("/test/image.png")
        assert result.text == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_disabled_returns_empty_result(self, ocr_config_no_fallback):
        """When fallback disabled, returns unavailable OCRResult."""
        engine = _make_engine(ocr_config_no_fallback)
        assert engine.config.fallback_enabled is False

        result = await engine._fallback_process("/test/image.png")

        assert isinstance(result, OCRResult)
        assert "OCR UNAVAILABLE" in result.text
        assert result.confidence == 0.0
        assert result.model_used == "none"

    @pytest.mark.asyncio
    async def test_fallback_disabled_does_not_call_api(self, ocr_config_no_fallback):
        """When fallback disabled, _api_fallback is never called."""
        engine = _make_engine(ocr_config_no_fallback)

        with patch.object(engine, "_api_fallback", new_callable=AsyncMock) as mock_api:
            await engine._fallback_process("/test/image.png")

        mock_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_enabled_result_from_api(self):
        """Fallback enabled returns the API fallback result."""
        config = make_ocr_config(fallback_enabled=True, fallback_provider="azure")
        engine = _make_engine(config)

        result = await engine._fallback_process("/test/image.png")

        assert isinstance(result, OCRResult)
        assert "azure" in result.model_used


# =========================================================================
# 8. _api_fallback
# =========================================================================


class TestApiFallback:
    """Tests for _api_fallback."""

    @pytest.mark.asyncio
    async def test_api_fallback_returns_not_configured(self, ocr_config):
        """_api_fallback returns a not-configured result."""
        engine = _make_engine(ocr_config)
        result = await engine._api_fallback("/test/image.png")

        assert isinstance(result, OCRResult)
        assert "FALLBACK NOT CONFIGURED" in result.text
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_api_fallback_includes_provider_name(self, ocr_config):
        """_api_fallback includes the provider name in text and model_used."""
        engine = _make_engine(ocr_config)
        result = await engine._api_fallback("/test/image.png")

        assert "google" in result.text
        assert result.model_used == "fallback-google"

    @pytest.mark.asyncio
    async def test_api_fallback_custom_provider(self):
        """_api_fallback works with different provider names."""
        config = make_ocr_config(fallback_provider="azure")
        engine = _make_engine(config)
        result = await engine._api_fallback("/test/image.png")

        assert "azure" in result.text
        assert result.model_used == "fallback-azure"

    @pytest.mark.asyncio
    async def test_api_fallback_aws_provider(self):
        """_api_fallback works with AWS provider."""
        config = make_ocr_config(fallback_provider="aws")
        engine = _make_engine(config)
        result = await engine._api_fallback("/test/image.png")

        assert "aws" in result.text
        assert result.model_used == "fallback-aws"

    @pytest.mark.asyncio
    async def test_api_fallback_zero_confidence(self, ocr_config):
        """_api_fallback always returns zero confidence."""
        engine = _make_engine(ocr_config)
        result = await engine._api_fallback("/test/image.png")
        assert result.confidence == 0.0


# =========================================================================
# 9. _extract_headers
# =========================================================================


class TestExtractHeaders:
    """Tests for _extract_headers."""

    def test_extract_markdown_h1_headers(self, ocr_config):
        """Extracts # headers."""
        engine = _make_engine(ocr_config)
        text = "# Introduction\nSome text\n# Chapter 1\nMore text"
        headers = engine._extract_headers(text)
        assert "Introduction" in headers
        assert "Chapter 1" in headers

    def test_extract_markdown_h2_headers(self, ocr_config):
        """Extracts ## headers."""
        engine = _make_engine(ocr_config)
        text = "## Section A\nText\n## Section B\nMore"
        headers = engine._extract_headers(text)
        assert "Section A" in headers
        assert "Section B" in headers

    def test_extract_markdown_h3_headers(self, ocr_config):
        """Extracts ### headers."""
        engine = _make_engine(ocr_config)
        text = "### Subsection\nDetails here"
        headers = engine._extract_headers(text)
        assert "Subsection" in headers

    def test_extract_bold_headers(self, ocr_config):
        """Extracts **bold** lines as headers."""
        engine = _make_engine(ocr_config)
        text = "**Important Title**\nSome content\n**Another Title**\nMore content"
        headers = engine._extract_headers(text)
        assert "Important Title" in headers
        assert "Another Title" in headers

    def test_extract_headers_limit_to_10(self, ocr_config):
        """Limits extracted headers to 10."""
        engine = _make_engine(ocr_config)
        lines = [f"# Header {i}" for i in range(20)]
        text = "\n".join(lines)
        headers = engine._extract_headers(text)
        assert len(headers) == 10

    def test_extract_headers_mixed_types(self, ocr_config):
        """Extracts mix of markdown and bold headers."""
        engine = _make_engine(ocr_config)
        text = "# Title\n\nText\n\n**Subtitle**\n\nMore text\n## Section"
        headers = engine._extract_headers(text)
        assert "Title" in headers
        assert "Subtitle" in headers
        assert "Section" in headers
        assert len(headers) == 3

    def test_extract_headers_empty_text(self, ocr_config):
        """Empty text returns empty list."""
        engine = _make_engine(ocr_config)
        headers = engine._extract_headers("")
        assert headers == []

    def test_extract_headers_no_headers_in_text(self, ocr_config):
        """Text without headers returns empty list."""
        engine = _make_engine(ocr_config)
        text = "Just plain text\nwith no headers\nat all."
        headers = engine._extract_headers(text)
        assert headers == []

    def test_extract_headers_empty_hash_line_ignored(self, ocr_config):
        """Lines that are just '#' with no content are ignored."""
        engine = _make_engine(ocr_config)
        text = "#\n# Real Header\n##\n"
        headers = engine._extract_headers(text)
        assert headers == ["Real Header"]

    def test_extract_headers_bold_too_long_ignored(self, ocr_config):
        """Bold text over 100 chars is not treated as a header."""
        engine = _make_engine(ocr_config)
        long_title = "A" * 101
        text = f"**{long_title}**\n**Short Title**"
        headers = engine._extract_headers(text)
        assert "Short Title" in headers
        assert long_title not in headers

    def test_extract_headers_bold_exactly_at_limit(self, ocr_config):
        """Bold text at exactly 99 chars is included (under 100 limit)."""
        engine = _make_engine(ocr_config)
        title_99 = "B" * 99
        text = f"**{title_99}**"
        headers = engine._extract_headers(text)
        assert title_99 in headers

    def test_extract_headers_strips_hash_symbols(self, ocr_config):
        """Hash symbols are stripped from the header text."""
        engine = _make_engine(ocr_config)
        text = "### Deeply Nested Header"
        headers = engine._extract_headers(text)
        assert headers == ["Deeply Nested Header"]

    def test_extract_headers_bold_inline_not_detected(self, ocr_config):
        """Bold text that doesn't start AND end the line is not detected."""
        engine = _make_engine(ocr_config)
        text = "Some **bold** text in a line\nAnother line"
        headers = engine._extract_headers(text)
        assert headers == []

    def test_extract_headers_empty_bold_ignored(self, ocr_config):
        """Empty bold markers (****) are ignored."""
        engine = _make_engine(ocr_config)
        text = "****\n**Real Header**"
        headers = engine._extract_headers(text)
        assert headers == ["Real Header"]


# =========================================================================
# 10. MockDeepSeekOCR
# =========================================================================


class TestMockDeepSeekOCR:
    """Tests for MockDeepSeekOCR."""

    def _make_mock_engine(self, config=None):
        if config is None:
            config = make_ocr_config()
        with patch("documents.ocr.deepseek_engine.get_document_config") as mock_gdc:
            mock_doc_config = MagicMock()
            mock_doc_config.ocr = config
            mock_gdc.return_value = mock_doc_config
            from documents.ocr.deepseek_engine import MockDeepSeekOCR

            return MockDeepSeekOCR(config)

    @pytest.mark.asyncio
    async def test_mock_load_model_returns_true(self):
        """MockDeepSeekOCR.load_model() always returns True."""
        engine = self._make_mock_engine()
        result = await engine.load_model()
        assert result is True

    @pytest.mark.asyncio
    async def test_mock_load_model_sets_loaded(self):
        """MockDeepSeekOCR.load_model() sets _loaded to True."""
        engine = self._make_mock_engine()
        assert engine._loaded is False
        await engine.load_model()
        assert engine._loaded is True

    @pytest.mark.asyncio
    async def test_mock_is_loaded_after_load(self):
        """is_loaded is True after load_model."""
        engine = self._make_mock_engine()
        await engine.load_model()
        assert engine.is_loaded is True

    @pytest.mark.asyncio
    async def test_mock_process_image_returns_ocr_result(self):
        """MockDeepSeekOCR.process_image() returns an OCRResult."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert isinstance(result, OCRResult)

    @pytest.mark.asyncio
    async def test_mock_process_image_contains_filename(self):
        """Mock result text contains the filename."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/my_document.png")
        assert "my_document.png" in result.text

    @pytest.mark.asyncio
    async def test_mock_process_image_confidence_is_one(self):
        """Mock result has confidence of 1.0."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_mock_process_image_model_used(self):
        """Mock result model_used is 'mock'."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert result.model_used == "mock"

    @pytest.mark.asyncio
    async def test_mock_process_image_has_images_false(self):
        """Mock result has_images is False."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert result.has_images is False

    @pytest.mark.asyncio
    async def test_mock_process_image_has_tables_false(self):
        """Mock result has_tables is False."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert result.has_tables is False

    @pytest.mark.asyncio
    async def test_mock_process_image_has_headers(self):
        """Mock result has detected headers."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert result.detected_headers == ["Mock OCR Output"]

    @pytest.mark.asyncio
    async def test_mock_process_image_processing_time(self):
        """Mock result has a low processing time."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert result.processing_time_ms == 10

    @pytest.mark.asyncio
    async def test_mock_process_image_with_language_hint(self):
        """Mock process_image accepts language_hint without error."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png", language_hint="te")
        assert isinstance(result, OCRResult)

    @pytest.mark.asyncio
    async def test_mock_process_image_with_grounding_false(self):
        """Mock process_image accepts grounding=False without error."""
        mock_engine = self._make_mock_engine()
        result = await mock_engine.process_image("/test/image.png", grounding=False)
        assert isinstance(result, OCRResult)

    @pytest.mark.asyncio
    async def test_mock_process_image_with_path_object(self):
        """Mock process_image works with Path objects."""
        engine = self._make_mock_engine()
        result = await engine.process_image(Path("/test/image.png"))
        assert "image.png" in result.text

    @pytest.mark.asyncio
    async def test_mock_process_image_placeholder_text(self):
        """Mock result text contains placeholder indicator."""
        engine = self._make_mock_engine()
        result = await engine.process_image("/test/image.png")
        assert "placeholder" in result.text.lower()

    @pytest.mark.asyncio
    async def test_mock_inherits_from_deepseek_ocr(self):
        """MockDeepSeekOCR is a subclass of DeepSeekOCR."""
        from documents.ocr.deepseek_engine import DeepSeekOCR, MockDeepSeekOCR

        assert issubclass(MockDeepSeekOCR, DeepSeekOCR)

    @pytest.mark.asyncio
    async def test_mock_load_model_multiple_calls(self):
        """Calling load_model multiple times is safe."""
        engine = self._make_mock_engine()
        r1 = await engine.load_model()
        r2 = await engine.load_model()
        assert r1 is True
        assert r2 is True
        assert engine._loaded is True

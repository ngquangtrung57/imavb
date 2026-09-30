"""Model-specific adapters for hidden state extraction (IMAVB paper release).

Each adapter defines:
1. How to load the model and processor
2. Where decoder layers are (for forward hooks)
3. How to run the forward/generate pass
4. Any dtype/device fixes needed

Paper reference: §4.3 + Appendix J (app:eval-impl)
"Hidden State Extraction" paragraph — forward hooks registered on decoder
layers via register_forward_hook(). Layer access pattern varies by architecture
(e.g., model.thinker.model.layers for the Qwen family). Each extraction
produces a tensor of shape (num_layers, hidden_dim) per sample.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseModelAdapter(ABC):
    """Base adapter for hidden state extraction."""

    def __init__(self, model_name: str, model_args: Dict[str, Any]):
        self.model_name = model_name
        self.model_args = model_args
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load(self) -> None:
        """Load model, tokenizer, and processor."""

    @abstractmethod
    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        """Extract last-token hidden states from all decoder layers.

        Returns:
            np.ndarray of shape (num_layers, hidden_dim)
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _format_question(self, sample: Dict) -> str:
        """Format MCQ question + options as a string."""
        options = (
            f"A. {sample.get('option_a', '')}\n"
            f"B. {sample.get('option_b', '')}\n"
            f"C. {sample.get('option_c', '')}\n"
            f"D. {sample.get('option_d', '')}\n"
            f"E. {sample.get('option_e', 'The visual detail in the question is incorrect')}\n"
            f"F. {sample.get('option_f', 'The audio detail in the question is incorrect')}"
        )
        return (
            f"Question:\n{sample['question']}\n\n"
            f"Options:\n{options}\n\n"
            "Answer with the option's letter from the given choices directly."
        )


def _extract_last_token_states(hidden_states: tuple) -> np.ndarray:
    """Extract last-token hidden state at each layer.

    Args:
        hidden_states: tuple of tensors, each [batch, seq_len, hidden_dim]

    Returns:
        np.ndarray of shape (num_layers, hidden_dim)
    """
    last_token_states = []
    for layer_hs in hidden_states:
        last_token_states.append(layer_hs[0, -1, :].float().cpu().numpy())
    return np.stack(last_token_states)


def _get_decoder_layers(model: Any) -> Optional[Any]:
    """Try common attribute paths to find the decoder layer list."""
    for attr_path in [
        "model.layers",
        "llm.model.layers",
        "model.model.layers",
        "language_model.model.layers",
    ]:
        obj = model
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return obj
    return None


# ---------------------------------------------------------------------------
# OLA
# ---------------------------------------------------------------------------


class OlaAdapter(BaseModelAdapter):
    """Adapter for OLA-7b.

    Uses generate() with audio (whisper mel spectrograms) and video frames.
    Hidden states captured via forward hooks on model.model.layers.
    """

    def load(self) -> None:
        ola_dir = os.environ.get("OLA_SOURCE_DIR", "")
        if not ola_dir:
            raise RuntimeError(
                "Set OLA_SOURCE_DIR environment variable to the path containing the Ola package "
                "(e.g., export OLA_SOURCE_DIR=/path/to/Ola)"
            )
        if ola_dir not in sys.path:
            sys.path.insert(0, ola_dir)

        os.environ.setdefault("LOWRES_RESIZE", "384x32")
        os.environ.setdefault("HIGHRES_BASE", "0x32")
        os.environ.setdefault("VIDEO_RESIZE", "0x64")
        os.environ.setdefault("VIDEO_MAXRES", "480")
        os.environ.setdefault("VIDEO_MINRES", "288")
        os.environ.setdefault("MAXRES", "1536")
        os.environ.setdefault("MINRES", "0")
        os.environ.setdefault("FORCE_NO_DOWNSAMPLE", "1")
        os.environ.setdefault("LOAD_VISION_EARLY", "1")
        os.environ.setdefault("PAD2STRIDE", "1")
        os.environ.setdefault("USE_SPEECH", "1")

        from ola.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_SPEECH_TOKEN,
            IMAGE_TOKEN_INDEX,
            SPEECH_TOKEN_INDEX,
        )
        from ola.datasets.preprocess import tokenizer_speech_image_token
        from ola.model.builder import load_pretrained_model

        pretrained = self.model_args.get("pretrained", "THUdyh/Ola-7b")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            pretrained, None, device="cuda"
        )
        self.model.eval().bfloat16()
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_SPEECH_TOKEN = DEFAULT_SPEECH_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SPEECH_TOKEN_INDEX = SPEECH_TOKEN_INDEX
        self.tokenizer_speech_image_token = tokenizer_speech_image_token

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        import PIL
        import whisper
        import librosa
        from decord import VideoReader, cpu
        from ola.conversation import conv_templates
        from ola.mm_utils import process_anyres_video

        context = self._format_question(sample)

        # Load video frames (50 uniformly sampled per paper Appendix J)
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        frame_idx = np.linspace(0, total - 1, min(50, total), dtype=int).tolist()
        frames = [PIL.Image.fromarray(vr.get_batch([i]).asnumpy()[0]) for i in frame_idx]

        processed = []
        for frame in frames:
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            processed.append(
                process_anyres_video(frame, self.image_processor).unsqueeze(0)
            )
        video_tensor = torch.cat(processed, dim=0).bfloat16().cuda()
        video_data = ((video_tensor, video_tensor), (384, 384), "video")

        # Extract audio; fall back to silence if no audio track
        CHUNK_LIM = 480000
        audio_np = None
        try:
            from moviepy import VideoFileClip
            clip = VideoFileClip(video_path)
            if clip.audio is not None:
                tmp_wav = tempfile.mktemp(suffix=".wav")
                clip.audio.write_audiofile(tmp_wav, logger=None)
                clip.close()
                audio_np, _ = librosa.load(tmp_wav, sr=16000)
                audio_np = audio_np.astype(np.float32)
                if len(audio_np.shape) > 1:
                    audio_np = audio_np[:, 0]
                os.remove(tmp_wav)
            else:
                clip.close()
        except Exception as e:
            logger.warning(f"OLA audio extraction failed ({e}), using silence")

        if audio_np is None:
            audio_np = np.zeros(CHUNK_LIM, dtype=np.float32)

        mel_chunks: list = []
        wav_chunks: list = []
        if len(audio_np) <= CHUNK_LIM:
            audio_np = whisper.pad_or_trim(audio_np)
            mel_chunks.append(audio_np)
            wav_chunks.append(torch.from_numpy(audio_np).unsqueeze(0))
        else:
            for i in range(0, len(audio_np), CHUNK_LIM):
                chunk = audio_np[i : i + CHUNK_LIM]
                if len(chunk) < CHUNK_LIM:
                    chunk = whisper.pad_or_trim(chunk)
                mel_chunks.append(chunk)
                wav_chunks.append(torch.from_numpy(chunk).unsqueeze(0))

        mels = []
        for chunk in mel_chunks:
            mel = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
            mels.append(mel)
        mels_t = torch.cat(mels, dim=0)
        wavs_t = torch.cat(wav_chunks, dim=0)
        if mels_t.shape[0] > 20:
            mels_t = mels_t[:20]
            wavs_t = wavs_t[:20]
        speechs = mels_t.bfloat16().cuda()
        speech_lengths = torch.LongTensor([mels_t.shape[1]] * mels_t.shape[0]).cuda()
        speech_chunks = torch.LongTensor([mels_t.shape[0]]).cuda()
        speech_wavs = wavs_t.bfloat16().cuda()

        qs = f"{self.DEFAULT_SPEECH_TOKEN}{self.DEFAULT_IMAGE_TOKEN}\n{context}"
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_speech_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_id).long().cuda()

        hidden_collector: list = []
        prefill_done = [False]
        hooks = []
        try:
            decoder_layers = self.model.model.layers
            n_layers = len(decoder_layers)
        except AttributeError as e:
            raise RuntimeError(f"Cannot find OLA decoder layers: {e}") from e

        def _hook(module, inp, output):
            if prefill_done[0]:
                return
            if isinstance(output, tuple) and len(output) > 0:
                hs = output[0]
            elif isinstance(output, torch.Tensor):
                hs = output
            else:
                return
            if isinstance(hs, torch.Tensor) and hs.dim() == 3:
                hidden_collector.append(hs[:, -1, :].float().cpu())
                if len(hidden_collector) >= n_layers:
                    prefill_done[0] = True

        for layer in decoder_layers:
            hooks.append(layer.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                self.model.generate(
                    inputs=input_ids,
                    images=video_data[0][0],
                    images_highres=video_data[0][1],
                    modalities=video_data[2],
                    speech=speechs,
                    speech_lengths=speech_lengths,
                    speech_chunks=speech_chunks,
                    speech_wav=speech_wavs,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                )
        finally:
            for h in hooks:
                h.remove()

        if hidden_collector:
            states = hidden_collector[:n_layers]
            return torch.stack(states, dim=0).squeeze(1).numpy()
        raise RuntimeError("OLA hidden state extraction failed — no states captured via hooks")


# ---------------------------------------------------------------------------
# OmniVinci
# ---------------------------------------------------------------------------


class OmniVinciAdapter(BaseModelAdapter):
    """Adapter for OmniVinci (VILA-based).

    Hidden states captured via forward hooks on decoder layers.
    """

    def load(self) -> None:
        from transformers import AutoModel, AutoProcessor

        pretrained = self.model_args.get("pretrained", "nvidia/OmniVinci")
        self.model = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        context = self._format_question(sample)
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "num_video_frames": 50},
                    {"type": "text", "text": context},
                ],
            },
        ]
        if hasattr(self.model, "config"):
            self.model.config.num_video_frames = 50
        vila_text = self.processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor([vila_text])

        gen_kwargs: dict = {}
        if inputs.input_ids is not None:
            gen_kwargs["input_ids"] = inputs.input_ids.to("cuda")
        if getattr(inputs, "attention_mask", None) is not None:
            gen_kwargs["attention_mask"] = inputs.attention_mask.to("cuda")
        if getattr(inputs, "media", None) is not None:
            gen_kwargs["media"] = inputs.media
        if getattr(inputs, "media_config", None) is not None:
            gen_kwargs["media_config"] = inputs.media_config

        with torch.no_grad():
            outputs = self.model(
                **gen_kwargs,
                output_hidden_states=True,
                return_dict=True,
            )
        return _extract_last_token_states(outputs.hidden_states)


# ---------------------------------------------------------------------------
# Qwen2.5-Omni
# ---------------------------------------------------------------------------


class Qwen25OmniAdapter(BaseModelAdapter):
    """Adapter for Qwen2.5-Omni.

    Hidden states captured via forward hooks on model.thinker.model.layers.
    Uses prefill-only collection (first forward pass through all decoder layers).
    Generates only 1 token to trigger the prefill pass.
    """

    def load(self) -> None:
        from transformers import (
            Qwen2_5OmniConfig,
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        pretrained = self.model_args.get("pretrained", "Qwen/Qwen2.5-Omni-7B")
        # Disable tensor parallelism plan (not needed for single-node inference)
        Qwen2_5OmniForConditionalGeneration._tp_plan = []
        config = Qwen2_5OmniConfig.from_pretrained(pretrained, enable_audio_output=False)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            pretrained, config=config, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(pretrained)
        self.tokenizer = self.processor.tokenizer

        try:
            from qwen_omni_utils import process_mm_info
            self.process_mm_info = process_mm_info
        except ImportError:
            raise ImportError("Install qwen-omni-utils[decord]")

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        return self._qwen_omni_extract(sample, video_path)

    def _qwen_omni_extract(self, sample: Dict, video_path: str) -> np.ndarray:
        """Extract hidden states via forward hooks on thinker decoder layers.

        Only collects from the PREFILL pass (first n_layers hook invocations)
        to avoid corruption from decode steps.
        """
        from moviepy import VideoFileClip

        context = self._format_question(sample)
        use_audio = False
        try:
            clip = VideoFileClip(video_path)
            use_audio = clip.audio is not None
            clip.close()
        except Exception:
            pass

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": 50},
                    {"type": "text", "text": context},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = self.process_mm_info(message, use_audio_in_video=use_audio)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
        ).to(self.device)

        # Cast float32 to bfloat16 (needed for Qwen3)
        cast_inputs = {
            k: v.to(dtype=torch.bfloat16)
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32
            else v
            for k, v in inputs.items()
        }

        hidden_collector: list = []
        prefill_done = [False]
        hooks = []

        try:
            thinker = getattr(self.model, "thinker", self.model)
            llm = getattr(thinker, "model", thinker)
            decoder_layers = llm.layers
            n_layers = len(decoder_layers)
        except AttributeError as e:
            raise RuntimeError(f"Cannot find Qwen decoder layers: {e}") from e

        def _hook(module, inp, output):
            if prefill_done[0]:
                return
            if isinstance(output, tuple) and len(output) > 0:
                hs = output[0]
            elif isinstance(output, torch.Tensor):
                hs = output
            else:
                return
            if isinstance(hs, torch.Tensor) and hs.dim() == 3:
                hidden_collector.append(hs[:, -1, :].float().cpu())
                if len(hidden_collector) >= n_layers:
                    prefill_done[0] = True

        for layer in decoder_layers:
            hooks.append(layer.register_forward_hook(_hook))
        logger.debug(
            f"Registered {len(hooks)} hooks on {n_layers} decoder layers for {self.model_name}"
        )

        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

        try:
            with torch.no_grad():
                self.model.generate(
                    **cast_inputs,
                    thinker_max_new_tokens=1,
                    max_new_tokens=1,
                    do_sample=False,
                    use_audio_in_video=use_audio,
                )
        finally:
            for h in hooks:
                h.remove()

        if hidden_collector:
            states = hidden_collector[:n_layers]
            return torch.stack(states, dim=0).squeeze(1).numpy()
        raise RuntimeError(
            f"{self.model_name} hidden state extraction failed — no states captured via hooks"
        )


# ---------------------------------------------------------------------------
# Qwen3-Omni
# ---------------------------------------------------------------------------


class Qwen3OmniAdapter(Qwen25OmniAdapter):
    """Adapter for Qwen3-Omni (inherits hook-based extraction from Qwen2.5)."""

    def load(self) -> None:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        pretrained = self.model_args.get(
            "pretrained", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype="auto", device_map="auto"
        ).eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(pretrained)
        self.tokenizer = self.processor.tokenizer
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

        try:
            from qwen_omni_utils import process_mm_info
            self.process_mm_info = process_mm_info
        except ImportError:
            raise ImportError("Install qwen-omni-utils")


# ---------------------------------------------------------------------------
# Baichuan-Omni
# ---------------------------------------------------------------------------


class BaichuanOmniAdapter(BaseModelAdapter):
    """Adapter for Baichuan-Omni-1.5.

    Baichuan-Omni's generate() does not support output_hidden_states, so we
    register forward hooks on model.model.layers. Uses the original video
    (with audio) plus chunked audio (via baichuan_audio_fix) to avoid OOM.
    """

    def load(self) -> None:
        import torchaudio
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Compatibility shim: torchaudio >= 2.10 removed list_audio_backends
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile", "sox"]

        pretrained = self.model_args.get("pretrained", "baichuan-inc/Baichuan-Omni-1d5")
        num_gpus = torch.cuda.device_count()
        load_kwargs: dict = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
        if num_gpus == 1:
            load_kwargs["device_map"] = "cuda:0"
        else:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {i: "10GiB" for i in range(num_gpus)}
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        self._baichuan_cache = tempfile.mkdtemp(prefix="baichuan_hs_")
        self.model.bind_processor(
            self.tokenizer, training=False, relative_path=self._baichuan_cache
        )
        self._processor = getattr(self.model, "processor", None)

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        from lmms_eval.models.model_utils.baichuan_audio_fix import (
            prepare_baichuan_video_audio,
        )

        context = self._format_question(sample)

        _, audio_chunks = prepare_baichuan_video_audio(
            video_path, self._baichuan_cache
        )
        video_json = json.dumps({"path": video_path, "type": "video"})
        full_prompt = f"<video_start>{video_json}<video_end>"
        for audio_path in audio_chunks:
            audio_json = json.dumps({"path": audio_path})
            full_prompt += f"<audio_start>{audio_json}<audio_end>"
        full_prompt += f"\n{context}"

        hidden_collector: list = []
        prefill_done = [False]

        try:
            decoder_layers = self.model.model.layers
            n_layers = len(decoder_layers)
        except AttributeError as e:
            raise RuntimeError(f"Cannot find Baichuan decoder layers: {e}") from e

        def _hook(module, inp, output):
            if prefill_done[0]:
                return
            if isinstance(output, tuple) and len(output) > 0:
                hs = output[0]
                if hs.dim() == 3:
                    hidden_collector.append(hs[:, -1, :].float().cpu())
                    if len(hidden_collector) >= n_layers:
                        prefill_done[0] = True

        hooks = []
        for layer in decoder_layers:
            hooks.append(layer.register_forward_hook(_hook))

        try:
            if self._processor is not None:
                proc_out = self._processor([full_prompt])
                input_ids = proc_out.input_ids.to(self.device)
                extra: dict = {}
                for attr in (
                    "audios", "encoder_length", "bridge_length",
                    "images", "patch_nums", "images_grid",
                    "videos", "videos_patch_nums", "videos_grid",
                ):
                    val = getattr(proc_out, attr, None)
                    if val is not None:
                        extra[attr] = val.to(self.device) if hasattr(val, "to") else val
                with torch.no_grad():
                    self.model.generate(
                        input_ids=input_ids,
                        max_new_tokens=1,
                        do_sample=False,
                        **extra,
                    )
            else:
                tok = self.tokenizer(context, return_tensors="pt")
                tok = {k: v.to(self.device) for k, v in tok.items()}
                with torch.no_grad():
                    self.model.generate(**tok, max_new_tokens=1, do_sample=False)
        finally:
            for h in hooks:
                h.remove()

        if hidden_collector:
            return torch.stack(hidden_collector, dim=0).squeeze(1).numpy()
        raise RuntimeError(
            "Baichuan-Omni hidden state extraction failed — no states captured via hooks"
        )


# ---------------------------------------------------------------------------
# Uni-MoE-2.0-Omni
# ---------------------------------------------------------------------------


class UniMoE2OmniAdapter(BaseModelAdapter):
    """Adapter for Uni-MoE-2.0-Omni.

    Supports output_hidden_states via standard HF forward() interface.
    """

    def load(self) -> None:
        from uni_moe.model import deepspeed_moe_inference_utils  # noqa: F401
        from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
        from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
        from uni_moe.qwen_vl_utils import process_mm_info

        pretrained = self.model_args.get("pretrained", "HIT-TMG/Uni-MoE-2.0-Omni")
        self.processor = Qwen2VLProcessor.from_pretrained(pretrained)
        self.model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
            pretrained,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor.data_args = self.model.config
        self.tokenizer = self.processor.tokenizer
        self.process_mm_info = process_mm_info

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        context = self._format_question(sample)
        query = f"<video>\n{context}"
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "video",
                        "video": video_path,
                        "nframes": 50,
                        "resized_height": 252,
                        "resized_width": 448,
                    },
                ],
            }
        ]]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, no=True
        )
        for pi in range(len(prompt)):
            prompt[pi] = (
                prompt[pi]
                .replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                .replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>")
                .replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            )

        image_inputs, video_inputs, audio_inputs = self.process_mm_info(messages)
        inputs = self.processor(
            text=prompt,
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "second_grid_ts" in inputs:
            inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
            del inputs["second_grid_ts"]

        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

        target_device = next(self.model.parameters()).device
        result: dict = {}
        for k, v in inputs.items():
            result[k] = v.to(device=target_device) if hasattr(v, "to") else v
        for k in ("pixel_values", "pixel_values_videos", "audio_features"):
            if k in result:
                result[k] = result[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model(**result, output_hidden_states=True)

        return _extract_last_token_states(outputs.hidden_states)


# ---------------------------------------------------------------------------
# MiniCPM-o
# ---------------------------------------------------------------------------


class MiniCPMOAdapter(BaseModelAdapter):
    """Adapter for MiniCPM-o-2.6.

    MiniCPM-o's generate() does not support output_hidden_states directly.
    Forward hooks are registered on llm.model.layers during generate().
    """

    def load(self) -> None:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        pretrained = self.model_args.get("pretrained", "openbmb/MiniCPM-o-2_6")
        self.model = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            init_vision=True,
            init_audio=True,
            init_tts=False,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.minicpm_processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True
        )

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        from PIL import Image

        context = self._format_question(sample)
        frames = self._extract_frames(video_path, max_frames=50)
        audios, audio_parts = self._extract_audio(video_path)
        has_audio = len(audios) > 0

        images = []
        cur_parts: list = []
        content = frames + [context] if frames else [context]
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_parts.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_parts.append(c)

        if has_audio:
            for _ in audios:
                cur_parts.insert(len(cur_parts) - 1, "(<audio>./</audio>)")

        msg_text = "".join(cur_parts) if has_audio else "\n".join(cur_parts)
        msgs_for_template = [{"role": "user", "content": msg_text}]

        tts_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' "
            "+ message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n"
            "<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}"
            "{% endif %}"
        )
        prompt_text = self.minicpm_processor.tokenizer.apply_chat_template(
            msgs_for_template,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=tts_template if has_audio else None,
        )
        proc_inputs = self.minicpm_processor(
            [prompt_text],
            [images],
            [audios] if has_audio else [[]],
            [audio_parts] if has_audio else [[]],
            max_slice_nums=None,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=32768,
        ).to(self.model.device)
        proc_inputs.pop("image_sizes", None)

        hidden_collector: list = []
        prefill_done = [False]

        try:
            decoder_layers = self.model.llm.model.layers
            n_layers = len(decoder_layers)
        except AttributeError as e:
            raise RuntimeError(f"Cannot find MiniCPM-o decoder layers: {e}") from e

        def _hook(module, inp, output):
            if prefill_done[0]:
                return
            if isinstance(output, tuple) and len(output) > 0:
                hs = output[0]
                if hs.dim() == 3:
                    hidden_collector.append(hs[:, -1, :].float().cpu())
                    if len(hidden_collector) >= n_layers:
                        prefill_done[0] = True

        hooks = []
        for layer in decoder_layers:
            hooks.append(layer.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                self.model.generate(
                    **proc_inputs,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1,
                    do_sample=False,
                    decode_text=False,
                )
        finally:
            for h in hooks:
                h.remove()

        if hidden_collector:
            return torch.stack(hidden_collector, dim=0).squeeze(1).numpy()
        raise RuntimeError(
            "MiniCPM-o hidden state extraction failed — no states captured via hooks"
        )

    def _extract_frames(self, video_path: str, max_frames: int = 50) -> List:
        """Extract frames from video as PIL images using decord."""
        from PIL import Image

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total = len(vr)
            indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
            indices = np.unique(indices)
            frames = vr.get_batch(indices).asnumpy()
            pil_frames = [Image.fromarray(f.astype("uint8")) for f in frames]
            return [f.resize((448, 252)) for f in pil_frames]
        except Exception as e:
            logger.warning(f"MiniCPM-o frame extraction failed ({e}), no video frames")
            return []

    def _extract_audio(self, video_path: str) -> Tuple[list, list]:
        """Extract audio from video as 16 kHz mono float32 chunks (30 s each).

        Returns (audios, audio_parts). Falls back to ([], []) when the video
        has no audio track or extraction fails.
        """
        import librosa

        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.video.io.VideoFileClip import VideoFileClip

            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return [], []

            wav_path = tempfile.mktemp(suffix=".wav")
            try:
                clip.audio.write_audiofile(wav_path, logger=None)
                clip.close()
                audio, _ = librosa.load(wav_path, sr=16000)
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            chunk_limit = 30 * 16000
            chunks = [
                audio[start : start + chunk_limit]
                for start in range(0, len(audio), chunk_limit)
            ]
            audio_parts = [0] * len(chunks)
            return chunks, audio_parts

        except Exception as e:
            logger.warning(f"MiniCPM-o audio extraction failed ({e}), no audio")
            return [], []


# ---------------------------------------------------------------------------
# Video-SALMONN-2
# ---------------------------------------------------------------------------


class VideoSALMONN2Adapter(BaseModelAdapter):
    """Adapter for Video-SALMONN-2.

    Uses Qwen2.5-VL processor + separate audio feature extraction.
    Hidden states come from generate() with return_dict_in_generate=True.
    """

    def load(self) -> None:
        from peft import PeftModel
        from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor

        pretrained = self.model_args.get(
            "pretrained", "tsinghua-ee/video_SALMONN2plus_7B_audioAlign"
        )
        lora_ckpt = self.model_args.get(
            "lora_ckpt", "tsinghua-ee/video-SALMONN-2_plus_7B"
        )
        self.model = video_SALMONN2_plus.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        if lora_ckpt:
            logger.info(f"Loading LoRA adapter from {lora_ckpt}")
            audio_layers = getattr(getattr(self.model, "audio", None), "layers", None)
            if audio_layers is not None:
                self.model.audio.layers = None
            self.model = PeftModel.from_pretrained(self.model, lora_ckpt)
            if audio_layers is not None:
                self.model.base_model.model.audio.layers = audio_layers
            self.model = self.model.merge_and_unload()
        if hasattr(self.model, "audio"):
            self.model.audio = self.model.audio.cuda()

        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=61250, min_pixels=784
        )
        model_tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        audio_pad_id = model_tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        if audio_pad_id is not None and audio_pad_id != model_tokenizer.unk_token_id:
            self.processor.tokenizer = model_tokenizer
        self.tokenizer = self.processor.tokenizer

        _TOKEN_DEFAULTS = {
            "image_token_id": ("<|image_pad|>", 151655),
            "video_token_id": ("<|video_pad|>", 151656),
            "vision_start_token_id": ("<|vision_start|>", 151652),
            "vision_end_token_id": ("<|vision_end|>", 151653),
        }
        cfg = self.model.config
        for attr, (tok_name, default) in _TOKEN_DEFAULTS.items():
            if not hasattr(cfg, attr):
                val = model_tokenizer.convert_tokens_to_ids(tok_name)
                if val is None or val == model_tokenizer.unk_token_id:
                    val = default
                setattr(cfg, attr, val)
        if not getattr(cfg, "rope_scaling", None):
            cfg.rope_scaling = {
                "type": "default",
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
            }
        for module in self.model.modules():
            if hasattr(module, "rope_scaling") and module.rope_scaling is None:
                module.rope_scaling = cfg.rope_scaling

        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError:
            raise ImportError("Install qwen-vl-utils")

    def _extract_audio_features(
        self, video_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """Extract audio features from video for Video-SALMONN-2."""
        AUDIO_SR = 16000
        CHUNK_SECS = 30
        TOKENS_PER_CHUNK = 60
        AUDIO_PAD = "<|audio_pad|>"
        try:
            import librosa
            audio, _ = librosa.load(video_path, sr=AUDIO_SR, mono=True)
        except Exception:
            return None, None, ""

        try:
            audio_processor = self.model.audio.audio_processor
        except AttributeError:
            return None, None, ""

        chunks = [
            audio[i : i + AUDIO_SR * CHUNK_SECS]
            for i in range(0, len(audio), AUDIO_SR * CHUNK_SECS)
        ]
        features_list = []
        lengths = []
        for chunk in chunks:
            feat = audio_processor(
                chunk, sampling_rate=AUDIO_SR, return_tensors="pt"
            ).input_features
            features_list.append(feat)
            lengths.append(feat.shape[-1] // 2)

        if not features_list:
            return None, None, ""

        audio_features = torch.cat(features_list, dim=0).to(self.device).bfloat16()
        audio_lengths = torch.tensor(lengths, dtype=torch.long).to(self.device)
        n_tokens = sum(TOKENS_PER_CHUNK for _ in chunks)
        pad_str = f"<|vision_start|>{AUDIO_PAD * n_tokens}<|vision_end|>"
        return audio_features, audio_lengths, pad_str

    def extract_hidden_states(self, sample: Dict, video_path: str) -> np.ndarray:
        context = self._format_question(sample)
        audio_features, audio_lengths, audio_token_str = self._extract_audio_features(
            video_path
        )

        user_content = [
            {
                "type": "video",
                "video": video_path,
                "nframes": 50,
                "resized_height": 252,
                "resized_width": 448,
            }
        ]
        if audio_token_str:
            user_content.append({"type": "text", "text": audio_token_str})
        user_content.append({"type": "text", "text": context})

        messages = [{"role": "user", "content": user_content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        if inputs.get("video_grid_thw") is None:
            inputs["video_grid_thw"] = torch.zeros(
                (0, 3), dtype=torch.long, device=inputs["input_ids"].device
            )

        gen_kwargs = dict(inputs)
        if audio_features is not None:
            gen_kwargs["audio_feature"] = audio_features
            gen_kwargs["audio_lengths"] = audio_lengths

        with torch.no_grad():
            out = self.model.generate(
                **gen_kwargs,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        if hasattr(out, "decoder_hidden_states") and out.decoder_hidden_states:
            return _extract_last_token_states(out.decoder_hidden_states[0])
        if hasattr(out, "hidden_states") and out.hidden_states:
            return _extract_last_token_states(out.hidden_states[0])
        raise RuntimeError(
            "Video-SALMONN-2 did not return decoder_hidden_states or hidden_states"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, type] = {
    "ola": OlaAdapter,
    "omnivinci": OmniVinciAdapter,
    "qwen2_5_omni": Qwen25OmniAdapter,
    "qwen3_omni": Qwen3OmniAdapter,
    "baichuan_omni": BaichuanOmniAdapter,
    "uni_moe_2_omni": UniMoE2OmniAdapter,
    "minicpm_o": MiniCPMOAdapter,
    "video_salmonn_2": VideoSALMONN2Adapter,
}

MODEL_DEFAULTS: Dict[str, str] = {
    "ola": "THUdyh/Ola-7b",
    "omnivinci": "nvidia/OmniVinci",
    "qwen2_5_omni": "Qwen/Qwen2.5-Omni-7B",
    "qwen3_omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "baichuan_omni": "baichuan-inc/Baichuan-Omni-1d5",
    "uni_moe_2_omni": "HIT-TMG/Uni-MoE-2.0-Omni",
    "minicpm_o": "openbmb/MiniCPM-o-2_6",
    "video_salmonn_2": "tsinghua-ee/video_SALMONN2plus_7B_audioAlign",
}


def get_adapter(model_name: str, model_args: Optional[Dict] = None) -> BaseModelAdapter:
    """Get adapter for a model by name."""
    if model_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[model_name](model_name, model_args or {})

import os
import re
import time
import random
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from src.utils import setup_logger, trim_silence_with_vad, load_t3_state
from src.config import TrainConfig
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS


logger = setup_logger("Chatterbox-Inference")

cfg = TrainConfig()

IS_TURBO = cfg.is_turbo
IS_LORA = cfg.is_lora
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir
MODE_NAME = "Nano" if cfg.is_nano else "Turbo" if IS_TURBO else "Standard"

LORA_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "new_lang_adapter")
FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, cfg.full_ft_filename)  # Full fine-tune only

# Generation params (Turbo/Nano ignore cfg_weight and exaggeration)
if IS_TURBO:
    PARAMS = {
        "temperature": 0.4,
        "repetition_penalty": 2.0,
    }
else:
    PARAMS = {
        "temperature": 0.7,
        "exaggeration": 0.8,
        "cfg_weight": 0.2,
        "repetition_penalty": 1.2,
    }

TEXT_FILE = "test.md"
AUDIO_PROMPT = cfg.inference_prompt_path
OUTPUT_FILE = f"./output-{datetime.now():%H-%M}.wav"  # output-<hour>-<minute>.wav


# -----------------------------------------------------------------------------
# Engine loaders (vocab is NOT extended, so a single base-engine load is enough)
# -----------------------------------------------------------------------------

def load_base_engine():
    """Loads the base engine on CPU and verifies it matches the config."""
    EngineClass = ChatterboxTurboTTS if IS_TURBO else ChatterboxTTS
    engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")

    if IS_TURBO:
        if engine.model_label != MODE_NAME:
            raise RuntimeError(f"Config expects {MODE_NAME}, but '{BASE_MODEL_DIR}' contains {engine.model_label}.")
        tok_len = len(engine.tokenizer)
        vocab = engine.t3.hp.text_tokens_dict_size
        if tok_len != vocab:
            logger.warning(f"Tokenizer len {tok_len} != T3 text vocab size {vocab}.")

    return engine


def _to_device(engine, device):
    engine.t3.to(device).eval()
    engine.s3gen.to(device).eval()
    engine.ve.to(device).eval()
    engine.device = device
    return engine


def load_finetuned_engine_lora(device):
    """Loads the base engine and attaches the LoRA adapter."""
    from peft import PeftModel

    logger.info(f"Loading {MODE_NAME} [LoRA] from: {BASE_MODEL_DIR}")

    if not os.path.exists(LORA_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Adapter path not found: {LORA_CHECKPOINT_PATH}")

    tts_engine = load_base_engine()

    # Turbo/Nano: remove wte to match the training architecture
    if IS_TURBO and hasattr(tts_engine.t3.tfmr, "wte"):
        logger.info(f"{MODE_NAME} Mode: Removing 'wte' layer.")
        del tts_engine.t3.tfmr.wte

    logger.info(f"Loading LoRA adapters & embeddings from: {LORA_CHECKPOINT_PATH}")
    tts_engine.t3 = PeftModel.from_pretrained(
        tts_engine.t3,
        LORA_CHECKPOINT_PATH,
        is_trainable=False,
    )

    return _to_device(tts_engine, device)


def load_finetuned_engine_full(device):
    """Loads the base engine and restores fully fine-tuned T3 weights from safetensors."""

    logger.info(f"Loading {MODE_NAME} [Full Fine-Tune] from: {BASE_MODEL_DIR}")

    if not os.path.exists(FINETUNED_WEIGHTS):
        raise FileNotFoundError(f"Fine-tuned file not found: {FINETUNED_WEIGHTS}")

    tts_engine = load_base_engine()

    if IS_TURBO and hasattr(tts_engine.t3.tfmr, "wte"):
        logger.info(f"{MODE_NAME} Mode: Removing 'wte' layer.")
        del tts_engine.t3.tfmr.wte

    logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
    state_dict = load_file(FINETUNED_WEIGHTS, device="cpu")
    load_t3_state(tts_engine.t3, state_dict)
    logger.info("Fine-tuned weights loaded successfully.")

    return _to_device(tts_engine, device)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Split after . ? ! (optionally followed by a closing quote) and whitespace.
SENTENCE_SPLIT = re.compile(r'(?:(?<=[.?!])|(?<=[.?!][”"’]))\s+')


def read_sentences(path):
    """
    Reads the markdown file, skips '#' headers and blank lines, joins the lines
    of each section (hard-wrapped lines belong to the same sentence), then splits
    into sentences. Every sentence is stripped of surrounding whitespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sections, current = [], []
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if current:
                sections.append(" ".join(current))
                current = []
            continue
        if line:
            current.append(line)
    if current:
        sections.append(" ".join(current))

    # sentences = []
    # for section in sections:
    #     sentences.extend(s.strip() for s in SENTENCE_SPLIT.split(section) if s.strip())
    return sections


def generate_sentence_audio(engine, text, prompt_path, **kwargs):
    """Generates audio for a single sentence, trims silence, and returns (sr, wav, gen_seconds)."""
    start = time.perf_counter()
    try:
        wav_tensor = engine.generate(text=text, audio_prompt_path=prompt_path, **kwargs)
        if isinstance(wav_tensor, tuple):
            wav_tensor = wav_tensor[0]
        wav_np = wav_tensor.squeeze().cpu().numpy()  # .cpu() syncs the GPU
        elapsed = time.perf_counter() - start
        trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
        return engine.sr, trimmed_wav, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Error generating sentence '{text[:30]}...': {e}")
        return 24000, np.zeros(0), elapsed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference running on: {device}")
    logger.info(f"Mode: {MODE_NAME} | Strategy: {'LoRA' if IS_LORA else 'Full Fine-Tune'}")

    if IS_LORA:
        engine = load_finetuned_engine_lora(device)
    else:
        engine = load_finetuned_engine_full(device)

    sentences = read_sentences(TEXT_FILE)
    logger.info(f"Found {len(sentences)} sentences to synthesize from {TEXT_FILE}.")

    all_chunks = []
    sample_rate = 24000
    total_time = 0.0

    set_seed(42)

    for i, sent in enumerate(sentences):
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        sr, audio_chunk, elapsed = generate_sentence_audio(engine, sent, AUDIO_PROMPT, **PARAMS)
        total_time += elapsed
        print(f"[{i+1}/{len(sentences)}] generated in {elapsed:.2f}s")

        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))

    print(f"Total generation time: {total_time:.2f}s")

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(OUTPUT_FILE, final_audio, sample_rate)
        logger.info(f"Result saved to: {OUTPUT_FILE}")
    else:
        logger.error("No audio was generated.")


if __name__ == "__main__":
    main()
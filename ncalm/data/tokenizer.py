"""Tokenizer interface and dataset-aware tokenizer implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, cast

from ncalm.data.formatting import CHAT_SPECIAL_TOKENS, example_to_text

logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data_cache"


class TokenizerInterface(ABC):
    """Minimum interface required by training and chat scripts."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        ...

    @property
    def pad_token_id(self) -> int | None:
        return None

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        ...


class HFTokenizerWrapper(TokenizerInterface):
    """Thin wrapper around a Hugging Face `tokenizers.Tokenizer`."""

    EOS_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._eos_id = self._resolve_special_id(self.EOS_TOKEN)
        self._pad_id = self._resolve_special_id(self.PAD_TOKEN)

    def _resolve_special_id(self, token: str) -> int | None:
        token_id = self._tokenizer.token_to_id(token)
        return token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        if self._eos_id is None:
            raise RuntimeError("Tokenizer is missing <|endoftext|> token")
        return self._eos_id

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)


class PretrainedHFTokenizer(TokenizerInterface):
    """Wrapper around a pretrained fast tokenizer from transformers."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            cache_dir=cache_dir,
        )

        if self._tokenizer.eos_token_id is None and self._tokenizer.eos_token is None:
            self._tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                self._tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    @property
    def vocab_size(self) -> int:
        # Use len() not .vocab_size — the latter excludes added/special tokens,
        # which can produce token IDs beyond vocab_size (e.g., GPT-NeoX has
        # .vocab_size=50254 but token IDs up to 50276). The embedding table
        # must accommodate ALL possible token IDs.
        return int(len(self._tokenizer))

    @property
    def eos_token_id(self) -> int:
        if self._tokenizer.eos_token_id is None:
            raise RuntimeError("Pretrained tokenizer is missing eos_token_id")
        return int(self._tokenizer.eos_token_id)

    @property
    def pad_token_id(self) -> int | None:
        token_id = self._tokenizer.pad_token_id
        return int(token_id) if token_id is not None else None

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)


class BPETokenizer(HFTokenizerWrapper):
    """Legacy BPE tokenizer kept for backward compatibility."""

    def __init__(self, vocab_size: int = 10000, cache_dir: str | None = None):
        tokenizer = _load_or_train_bpe_tokenizer(
            dataset_name="roneneldan/TinyStories",
            dataset_config=None,
            dataset_splits=["train"],
            vocab_size=vocab_size,
            cache_dir=cache_dir,
            cache_stem=f"tinystories_bpe_{vocab_size // 1000}k",
        )
        super().__init__(tokenizer)


class ModernUnigramTokenizer(HFTokenizerWrapper):
    """Modern unigram tokenizer trained on arbitrary HF datasets."""

    def __init__(
        self,
        vocab_size: int = 32000,
        dataset_name: str = "HuggingFaceTB/smoltalk2",
        dataset_config: str = "SFT",
        dataset_splits: list[str] | None = None,
        cache_dir: str | None = None,
        cache_stem: str | None = None,
        train_max_examples: int | None = None,
    ):
        if dataset_splits is None:
            dataset_splits = [
                "smoltalk_smollm3_everyday_conversations_no_think",
                "smoltalk_smollm3_systemchats_30k_no_think",
                "smoltalk_smollm3_smol_magpie_ultra_no_think",
                "smoltalk_smollm3_smol_rewrite_no_think",
                "smoltalk_smollm3_smol_summarize_no_think",
                "OpenHermes_2.5_no_think",
                "OpenThoughts3_1.2M_no_think_no_think",
                "tulu_3_sft_personas_instruction_following_no_think",
            ]

        if cache_stem is None:
            safe_cfg = (dataset_config or "default").lower()
            cache_stem = f"smoltalk2_{safe_cfg}_unigram_{vocab_size // 1000}k"

        tokenizer = _load_or_train_unigram_tokenizer(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_splits=dataset_splits,
            vocab_size=vocab_size,
            cache_dir=cache_dir,
            cache_stem=cache_stem,
            max_examples=train_max_examples,
        )
        super().__init__(tokenizer)


def _resolve_cache_dir(cache_dir: str | None) -> Path:
    resolved = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _dataset_text_iterator(
    dataset_name: str,
    dataset_config: str | None,
    dataset_splits: list[str],
    *,
    streaming: bool = True,
    max_examples: int | None = None,
) -> Iterable[str]:
    from datasets import load_dataset

    seen = 0

    for split_name in dataset_splits:
        shard = load_dataset(
            dataset_name,
            dataset_config,
            split=split_name,
            streaming=streaming,
        )
        for example in shard:
            if not isinstance(example, dict):
                continue
            text = example_to_text(cast(dict[str, Any], example))
            if text:
                yield text
                seen += 1
                if max_examples is not None and seen >= max_examples:
                    return


def _load_or_train_bpe_tokenizer(
    dataset_name: str,
    dataset_config: str | None,
    dataset_splits: list[str],
    vocab_size: int,
    cache_dir: str | None,
    cache_stem: str,
    max_examples: int | None = None,
):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    cache_root = _resolve_cache_dir(cache_dir)
    cache_file = cache_root / f"{cache_stem}.json"

    if cache_file.exists():
        logger.info(f"Loading cached tokenizer from {cache_file.name}...")
        return Tokenizer.from_file(str(cache_file))

    logger.info(
        f"Training BPE tokenizer on {dataset_name}"
        f" config={dataset_config!r} splits={dataset_splits} vocab={vocab_size}"
    )
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=CHAT_SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _dataset_text_iterator(
            dataset_name,
            dataset_config,
            dataset_splits,
            max_examples=max_examples,
        ),
        trainer=trainer,
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.save(str(cache_file))
    logger.info(f"Saved tokenizer to {cache_file}")
    return tokenizer


def _load_or_train_unigram_tokenizer(
    dataset_name: str,
    dataset_config: str | None,
    dataset_splits: list[str],
    vocab_size: int,
    cache_dir: str | None,
    cache_stem: str,
    max_examples: int | None = None,
):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    cache_root = _resolve_cache_dir(cache_dir)
    cache_file = cache_root / f"{cache_stem}.json"

    if cache_file.exists():
        logger.info(f"Loading cached tokenizer from {cache_file.name}...")
        return Tokenizer.from_file(str(cache_file))

    logger.info(
        f"Training unigram tokenizer on {dataset_name}"
        f" config={dataset_config!r} splits={dataset_splits} vocab={vocab_size}"
    )

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="_")

    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="<|unk|>",
        special_tokens=CHAT_SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        _dataset_text_iterator(
            dataset_name,
            dataset_config,
            dataset_splits,
            max_examples=max_examples,
        ),
        trainer=trainer,
    )
    tokenizer.decoder = decoders.Metaspace(replacement="_", prepend_scheme="never")
    tokenizer.save(str(cache_file))
    logger.info(f"Saved tokenizer to {cache_file}")
    return tokenizer


def create_tokenizer(
    name: str,
    *,
    vocab_size: int,
    cache_dir: str | None = None,
    cache_stem: str | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_splits: list[str] | None = None,
    train_max_examples: int | None = None,
    pretrained_name: str | None = None,
):
    """Factory for tokenizers used by training/chat scripts."""
    normalized = name.lower()
    if normalized == "bpe":
        return BPETokenizer(vocab_size=vocab_size, cache_dir=cache_dir)
    if normalized in {"pretrained_hf", "hf_pretrained", "mistral"}:
        if pretrained_name is None:
            raise ValueError("pretrained_name is required for pretrained_hf tokenizer")
        return PretrainedHFTokenizer(pretrained_name, cache_dir=cache_dir)
    if normalized in {"modern_unigram", "unigram", "smoltalk2_unigram"}:
        return ModernUnigramTokenizer(
            vocab_size=vocab_size,
            dataset_name=dataset_name or "HuggingFaceTB/smoltalk2",
            dataset_config=dataset_config or "SFT",
            dataset_splits=dataset_splits,
            cache_dir=cache_dir,
            cache_stem=cache_stem,
            train_max_examples=train_max_examples,
        )
    raise ValueError(f"Unknown tokenizer '{name}'")

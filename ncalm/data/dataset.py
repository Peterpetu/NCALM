"""Dataset loading, caching, and random-window data iterators for causal LM."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ncalm.data.formatting import example_to_text
from ncalm.data.tokenizer import TokenizerInterface

logger = logging.getLogger(__name__)


def _ensure_zstd_support():
    """Register zstd compression with fsspec if available.

    The Pile dataset uses .jsonl.zst files. Some fsspec versions don't
    natively support zstd even when the zstandard package is installed.
    """
    try:
        import zstandard  # noqa: F401
        import fsspec.compression

        if "zstd" not in fsspec.compression.compr:

            def _zstd_open(path, mode="rb", **kwargs):
                import io
                import zstandard as zstd

                dctx = zstd.ZstdDecompressor()
                fh = open(path, "rb")
                reader = dctx.stream_reader(fh)
                return io.BufferedReader(reader)

            fsspec.compression.register_compression("zstd", _zstd_open, "zstd")
            if "zst" not in fsspec.compression.compr:
                fsspec.compression.register_compression("zst", _zstd_open, "zst")
            logger.debug("Registered zstd/zst compression with fsspec")
    except ImportError:
        pass  # zstandard not installed — will fail later with a clear error


_ensure_zstd_support()


def _load_dataset_with_retry(dataset_name: str, dataset_config: str | None, split_name: str):
    from datasets import load_dataset

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            return load_dataset(
                dataset_name,
                dataset_config,
                split=split_name,
                streaming=True,
                token=True,
            )
        except Exception as err:
            last_err = err
            err_text = str(err)
            if "429" not in err_text and "Read timed out" not in err_text and "ReadTimeout" not in err_text:
                raise
            sleep_s = min(60, 5 * (2**attempt))
            logger.warning(
                f"Transient HF error opening dataset split '{split_name}', retrying in {sleep_s}s"
            )
            time.sleep(sleep_s)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to load dataset split '{split_name}'")


def _format_token_progress(current: int, target: int) -> str:
    if target <= 0:
        return f"{current:,} tok"
    pct = min(100.0, 100.0 * current / target)
    bar_width = 24
    filled = int(bar_width * pct / 100.0)
    bar = "#" * filled + "-" * (bar_width - filled)
    return f"[{bar}] {pct:6.2f}% | {current:,}/{target:,} tok"


class MemmapTokenStore:
    """Disk-backed token store for very large flattened corpora."""

    def __init__(self, data_path: Path, length: int):
        self.data_path = data_path
        self.length = int(length)
        self._arr = np.memmap(self.data_path, dtype=np.int32, mode="r", shape=(self.length,))

    def __len__(self) -> int:
        return self.length

    def element_size(self) -> int:
        return 4

    def batch_windows(self, starts: np.ndarray, seq_len: int) -> np.ndarray:
        out = np.empty((len(starts), seq_len), dtype=np.int64)
        for i, start in enumerate(starts.tolist()):
            out[i] = self._arr[start : start + seq_len]
        return out


def _memmap_meta_path(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".meta.json")


class TokenCorpusData:
    """Generic token corpus cached as a flat torch.int32 tensor."""

    DEFAULT_DATASET = "roneneldan/TinyStories"
    MAX_TOKENS = 10_000_000_000

    def __init__(
        self,
        seq_len: int = 256,
        cache_dir: str | None = None,
        tokenizer: TokenizerInterface | None = None,
        cache_name: str = "token_corpus.pt",
        dataset_name: str | None = None,
        dataset_config: str | None = None,
        dataset_splits: list[str] | None = None,
        dataset_sources: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        append_eos: bool = True,
    ):
        self.seq_len = seq_len
        self.tokenizer: TokenizerInterface | None = tokenizer
        self._cache_name = cache_name
        self.dataset_name = dataset_name or self.DEFAULT_DATASET
        self.dataset_config = dataset_config
        self.dataset_splits = dataset_splits or ["train"]
        self.dataset_sources = dataset_sources
        self.max_tokens = max_tokens or self.MAX_TOKENS
        self.append_eos = append_eos

        cache_root = Path(cache_dir) if cache_dir is not None else Path(__file__).parent.parent.parent / "data_cache"
        self.cache_dir: Path = cache_root
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokens = self._load_data()
        self.n_tokens = len(self.tokens)
        logger.info(
            f"Dataset: {self.dataset_name} config={self.dataset_config!r} "
            f"tokens={self.n_tokens:,} "
            f"({self.tokens.element_size() * self.n_tokens / 1024**3:.2f}GB)"
        )

    def _load_data(self) -> torch.Tensor | MemmapTokenStore:
        cache_file = self.cache_dir / self._cache_name
        use_memmap = cache_file.suffix == ".bin"

        if use_memmap and cache_file.exists() and _memmap_meta_path(cache_file).exists():
            logger.info(f"Loading cached memmap tokens from {cache_file.name}...")
            with open(_memmap_meta_path(cache_file), "r", encoding="utf-8") as f:
                meta = json.load(f)
            length = min(int(meta["length"]), int(self.max_tokens))
            return MemmapTokenStore(cache_file, length)

        if not use_memmap and cache_file.exists():
            logger.info(f"Loading cached tokens from {cache_file.name}...")
            tokens = torch.load(cache_file, map_location="cpu", weights_only=True)
            if len(tokens) > self.max_tokens:
                tokens = tokens[: self.max_tokens].clone()
            if tokens.dtype == torch.int64:
                tokens = tokens.to(torch.int32)
            return tokens.contiguous()

        logger.info(
            f"Loading dataset {self.dataset_name} config={self.dataset_config!r} "
            f"splits={self.dataset_splits}"
        )
        source_specs = self._source_specs()

        total_examples = 0
        total_tokens = 0
        if self.tokenizer is None:
            raise ValueError("TokenCorpusData requires a tokenizer")

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if use_memmap:
            return self._build_memmap_cache(source_specs, cache_file, eos_token_id)

        all_tokens: list[int] = []

        for toks in self._iter_token_sequences(source_specs, eos_token_id):
            all_tokens.extend(toks)
            total_examples += 1
            total_tokens += len(toks)

            if total_examples % 10000 == 0:
                logger.info(f"  {total_examples:,} examples ({total_tokens:,} tokens)")
            if total_tokens >= self.max_tokens:
                logger.info(
                    f"Reached max_tokens={self.max_tokens:,}; stopping tokenization"
                )
                break

        tokens_tensor = torch.tensor(all_tokens[: self.max_tokens], dtype=torch.int32)
        logger.info(f"Caching to {cache_file}...")
        torch.save(tokens_tensor, cache_file)
        return tokens_tensor.contiguous()

    def _source_specs(self) -> list[dict[str, Any]]:
        if self.dataset_sources:
            return self.dataset_sources
        return [{
            "config": self.dataset_config,
            "splits": self.dataset_splits,
        }]

    def _iter_token_sequences(self, source_specs: list[dict[str, Any]], eos_token_id: int | None):
        if self.tokenizer is None:
            raise ValueError("TokenCorpusData requires a tokenizer")

        total_tokens = 0
        for spec in source_specs:
            config_name = spec.get("config", self.dataset_config)
            splits = spec.get("splits", self.dataset_splits)
            max_examples = spec.get("max_examples")
            max_source_tokens = spec.get("max_tokens")

            source_examples = 0
            source_tokens = 0
            for split_name in splits:
                shard = _load_dataset_with_retry(self.dataset_name, config_name, split_name)
                for example in shard:
                    if not isinstance(example, dict):
                        continue
                    example_dict = cast(dict[str, Any], example)
                    text = example_to_text(example_dict)
                    if not text:
                        continue
                    toks = self.tokenizer.encode(text)
                    if self.append_eos and eos_token_id is not None:
                        toks = toks + [eos_token_id]

                    global_remaining = int(self.max_tokens) - total_tokens
                    if global_remaining <= 0:
                        return
                    if max_source_tokens is not None:
                        source_remaining = int(max_source_tokens) - source_tokens
                        if source_remaining <= 0:
                            break
                        remaining = min(global_remaining, source_remaining)
                    else:
                        remaining = global_remaining
                    if len(toks) > remaining:
                        toks = toks[:remaining]

                    yield toks
                    source_examples += 1
                    source_tokens += len(toks)
                    total_tokens += len(toks)

                    if max_examples is not None and source_examples >= int(max_examples):
                        break
                    if total_tokens >= self.max_tokens:
                        return
                if max_examples is not None and source_examples >= int(max_examples):
                    break
                if max_source_tokens is not None and source_tokens >= int(max_source_tokens):
                    break

    def _build_memmap_cache(self, source_specs, cache_file: Path, eos_token_id: int | None) -> MemmapTokenStore:
        if self.tokenizer is None:
            raise ValueError("TokenCorpusData requires a tokenizer")

        tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        meta_file = _memmap_meta_path(cache_file)
        tmp_meta = meta_file.with_suffix(meta_file.suffix + ".tmp")

        total_examples = 0
        total_tokens = 0
        flush_tokens = 0
        buffer: list[int] = []
        flush_threshold = 1_000_000
        start_time = time.time()
        last_status_time = start_time

        with open(tmp_file, "wb") as f:
            for toks in self._iter_token_sequences(source_specs, eos_token_id):
                buffer.extend(toks)
                flush_tokens += len(toks)
                total_examples += 1
                total_tokens += len(toks)

                now = time.time()
                if total_examples % 10000 == 0 or (now - last_status_time) >= 5.0:
                    elapsed = max(now - start_time, 1e-6)
                    tok_per_sec = total_tokens / elapsed
                    tmp_size_gb = tmp_file.stat().st_size / 1024**3 if tmp_file.exists() else 0.0
                    status = (
                        f"  {_format_token_progress(total_tokens, int(self.max_tokens))} "
                        f"| {total_examples:,} ex | {tok_per_sec:,.0f} tok/s | tmp={tmp_size_gb:.2f} GB"
                    )
                    print(status, flush=True)
                    last_status_time = now

                if flush_tokens >= flush_threshold:
                    np.asarray(buffer, dtype=np.int32).tofile(f)
                    buffer.clear()
                    flush_tokens = 0

                if total_tokens >= self.max_tokens:
                    logger.info(
                        f"Reached max_tokens={self.max_tokens:,}; stopping tokenization"
                    )
                    break

            if buffer:
                np.asarray(buffer, dtype=np.int32).tofile(f)

        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump({"length": total_tokens, "dtype": "int32"}, f, indent=2)

        tmp_file.replace(cache_file)
        tmp_meta.replace(meta_file)
        logger.info(f"Cached memmap tokens to {cache_file} ({total_tokens:,} tokens)")
        return MemmapTokenStore(cache_file, total_tokens)


class TinyStoriesData(TokenCorpusData):
    """Backward-compatible TinyStories wrapper."""

    HF_DATASET = "roneneldan/TinyStories"
    MAX_TOKENS = 500_000_000

    def __init__(
        self,
        seq_len: int = 256,
        cache_dir: str | None = None,
        tokenizer=None,
        cache_name: str = "tinystories_tokens.pt",
    ):
        super().__init__(
            seq_len=seq_len,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            cache_name=cache_name,
            dataset_name=self.HF_DATASET,
            dataset_config=None,
            dataset_splits=["train"],
            max_tokens=self.MAX_TOKENS,
            append_eos=True,
        )


class ThePileData(TokenCorpusData):
    """The Pile token corpus for pretraining comparison against Pythia.

    Uses monology/pile-uncopyrighted by default (freely accessible).
    Override dataset_name in profile via dataset_hf_name to use other
    Pile variants (e.g. EleutherAI/the_pile_deduplicated).
    """

    DEFAULT_HF_DATASET = "monology/pile-uncopyrighted"
    MAX_TOKENS = 10_000_000_000

    def __init__(
        self,
        seq_len: int = 1024,
        cache_dir: str | None = None,
        tokenizer=None,
        cache_name: str = "pile_tokens_2b.bin",
        dataset_name: str | None = None,
        max_tokens: int = 2_000_000_000,
    ):
        super().__init__(
            seq_len=seq_len,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            cache_name=cache_name,
            dataset_name=dataset_name or self.DEFAULT_HF_DATASET,
            dataset_config=None,
            dataset_splits=["train"],
            max_tokens=max_tokens,
            append_eos=True,
        )


class SmolTalk2Data(TokenCorpusData):
    """SmolTalk2 token corpus for large-scale post-training style data."""

    HF_DATASET = "HuggingFaceTB/smoltalk2"

    DEFAULT_SPLITS = [
        "smoltalk_smollm3_everyday_conversations_no_think",
        "smoltalk_smollm3_systemchats_30k_no_think",
        "smoltalk_smollm3_smol_magpie_ultra_no_think",
        "smoltalk_smollm3_explore_instruct_rewriting_no_think",
        "smoltalk_smollm3_smol_rewrite_no_think",
        "smoltalk_smollm3_smol_summarize_no_think",
        "OpenHermes_2.5_no_think",
        "OpenThoughts3_1.2M_no_think_no_think",
        "tulu_3_sft_personas_instruction_following_no_think",
    ]

    def __init__(
        self,
        seq_len: int = 256,
        cache_dir: str | None = None,
        tokenizer=None,
        cache_name: str = "smoltalk2_sft_tokens.bin",
        dataset_config: str = "SFT",
        dataset_splits: list[str] | None = None,
        dataset_sources: list[dict[str, Any]] | None = None,
        max_tokens: int = 5_000_000_000,
    ):
        super().__init__(
            seq_len=seq_len,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            cache_name=cache_name,
            dataset_name=self.HF_DATASET,
            dataset_config=dataset_config,
            dataset_splits=dataset_splits or self.DEFAULT_SPLITS,
            dataset_sources=dataset_sources,
            max_tokens=max_tokens,
            append_eos=True,
        )


def create_data_loader(
    data: TokenCorpusData,
    batch_size: int,
    device: torch.device,
    split: str = "train",
):
    """Create an infinite random-access data iterator with train/val split."""
    tokens = data.tokens
    seq_len = data.seq_len
    total = len(tokens)

    if split == "train":
        start, end = 0, int(total * 0.995)
    else:
        start, end = int(total * 0.995), total

    n = end - start
    offsets = torch.arange(seq_len, dtype=torch.long)

    if n <= seq_len + 1:
        raise ValueError(
            f"Split '{split}' is too small for seq_len={seq_len}. "
            f"Available tokens: {n}"
        )

    def iterator():
        while True:
            starts = torch.randint(0, n - seq_len - 1, (batch_size,), dtype=torch.long)
            if isinstance(tokens, MemmapTokenStore):
                global_starts = starts.numpy() + start
                batch_np = tokens.batch_windows(global_starts, seq_len)
                batch = torch.from_numpy(batch_np).to(dtype=torch.long, device=device)
            else:
                split_tokens = tokens[start:end].contiguous()
                indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
                batch = split_tokens[indices].to(dtype=torch.long, device=device)
            yield batch

    return iterator()

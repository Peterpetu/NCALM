#!/usr/bin/env python3
"""Prepare tokenizer and dataset caches for long-run training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ncalm.profiles import PROFILES
from ncalm.data.tokenizer import create_tokenizer
from ncalm.data.dataset import SmolTalk2Data, TinyStoriesData
from ncalm.env_utils import load_local_env


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Prepare tokenizer/data assets for long-run training")
    parser.add_argument("--profile", required=True, choices=list(PROFILES.keys()))
    parser.add_argument("--data-cache-dir", default=None)
    parser.add_argument("--tokenizer-cache-dir", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    args = parser.parse_args()

    config = dict(PROFILES[args.profile])
    if args.data_cache_dir:
        config["dataset_cache_dir"] = args.data_cache_dir
    if args.tokenizer_cache_dir:
        config["tokenizer_cache_dir"] = args.tokenizer_cache_dir
    if args.hf_cache_dir:
        config["hf_cache_dir"] = args.hf_cache_dir

    if config.get("hf_cache_dir"):
        hf_cache = str(config["hf_cache_dir"])
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    tokenizer = create_tokenizer(
        config.get("tokenizer_name", "bpe"),
        vocab_size=config.get("tokenizer_vocab_size", config.get("vocab_size", 10000)),
        cache_dir=config.get("tokenizer_cache_dir"),
        cache_stem=config.get("tokenizer_cache_stem"),
        dataset_name=config.get("tokenizer_dataset_name", config.get("dataset_hf_name")),
        dataset_config=config.get("tokenizer_dataset_config", config.get("dataset_config")),
        dataset_splits=config.get("tokenizer_dataset_splits"),
        train_max_examples=config.get("tokenizer_train_max_examples"),
        pretrained_name=config.get("tokenizer_pretrained_name"),
    )

    if config.get("dataset", "tinystories") == "smoltalk2":
        data = SmolTalk2Data(
            seq_len=config["seq_len"],
            tokenizer=tokenizer,
            cache_name=config.get("dataset_cache_name", "smoltalk2_sft_tokens_5b.pt"),
            dataset_config=config.get("dataset_config", "SFT"),
            dataset_splits=config.get("dataset_splits"),
            dataset_sources=config.get("dataset_sources"),
            max_tokens=config.get("max_dataset_tokens", 5_000_000_000),
            cache_dir=config.get("dataset_cache_dir"),
        )
    else:
        data = TinyStoriesData(
            seq_len=config["seq_len"],
            tokenizer=tokenizer,
            cache_name=config.get("dataset_cache_name", "tinystories_tokens_ncalm.pt"),
        )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")
    print(f"Token corpus size:     {data.n_tokens:,} tokens")


if __name__ == "__main__":
    main()

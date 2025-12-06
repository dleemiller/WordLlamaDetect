#!/usr/bin/env python3
"""Count token occurrences in the training split and save as an n_vocab x 1 numpy tensor.

This script mirrors the training pipeline configuration:
- Uses the training config to load the dataset (with language filtering/shuffling).
- Loads the tokenizer from the first configured model.
- Tokenizes with the configured max sequence length and no padding.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from wldetect.config.loader import load_model_config, load_training_config
from wldetect.data.dataset import prepare_dataset
from wldetect.training.trainer import LanguageDetectionDataset


def parse_args() -> tuple[str, str, int, int]:
    parser = ArgumentParser(description="Count token occurrences in the training dataset.")
    parser.add_argument(
        "--training-config",
        default="configs/training/gemma3-27b.yaml",
        help="Path to the training config YAML.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/token_counts.npy",
        help="Where to save the counts tensor (n_vocab x 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for tokenization workers.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()
    return args.training_config, args.output, args.batch_size, args.num_workers


def main() -> None:
    training_config_path, output_path, batch_size, num_workers = parse_args()

    training_config = load_training_config(training_config_path)
    model_config = load_model_config(training_config.model_config_path)

    tokenizer = AutoTokenizer.from_pretrained(model_config.all_models[0].name)
    max_length = model_config.inference.max_sequence_length

    # Get vocab size from embeddings (matches actual model vocab)
    from wldetect.embeddings import EmbeddingsManager

    embeddings_manager = EmbeddingsManager(model_config, cache_dir="artifacts/embeddings")
    embeddings = embeddings_manager.load_cached_embeddings()
    vocab_size = embeddings.shape[0]
    print(f"Using vocab_size={vocab_size} from embeddings (tokenizer reports {len(tokenizer)})")

    language_codes = list(model_config.languages.keys())
    dataset_dict = prepare_dataset(training_config.dataset, language_codes)
    train_split = training_config.dataset.train_split or "train"
    if train_split not in dataset_dict:
        raise ValueError(f"Train split '{train_split}' not found in dataset.")
    raw_train_dataset = dataset_dict[train_split]

    dataset = LanguageDetectionDataset(
        raw_train_dataset,
        tokenizer,
        language_to_idx=model_config.languages,
        max_length=max_length,
    )

    def collate_token_ids(batch: list[dict]) -> list[list[int]]:
        return [sample["token_ids"] for sample in batch]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_token_ids,
        pin_memory=False,
        drop_last=False,
    )

    counts = np.zeros((vocab_size,), dtype=np.int64)

    for token_id_lists in tqdm(dataloader, desc="Counting tokens", total=len(dataloader)):
        flat_ids: list[int] = [token_id for seq in token_id_lists for token_id in seq]
        if not flat_ids:
            continue
        bincount = np.bincount(flat_ids, minlength=vocab_size)
        counts += bincount

    counts = counts.reshape(-1, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, counts)

    total_tokens = int(counts.sum())
    print(
        f"Saved token counts to {output_path} "
        f"(shape={counts.shape}, dtype={counts.dtype}, total_tokens={total_tokens})"
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

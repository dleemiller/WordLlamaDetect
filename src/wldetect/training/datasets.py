"""Shared training datasets and collate utilities."""

import logging

import torch
from torch.utils.data import Dataset


class LanguageDetectionDataset(Dataset):
    """PyTorch dataset for language detection with lazy tokenization."""

    def __init__(
        self,
        dataset,
        tokenizer,
        language_to_idx: dict[str, int],
        max_length: int = 512,
        logger: logging.Logger | None = None,
    ):
        """Initialize dataset."""
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.language_to_idx = language_to_idx
        self.max_length = max_length
        self.logger = logger or logging.getLogger("wldetect")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        example = self.dataset[idx]
        text = example.get("text", "")
        if not isinstance(text, str) or text == "":
            if not hasattr(self, "_warned_missing_text"):
                self.logger.warning(
                    f"Missing or empty text at index {idx}; replacing with empty string"
                )
                self._warned_missing_text = True
            text = ""
        language = example["language"]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        return {
            "token_ids": encoded["input_ids"],
            "labels": self.language_to_idx[language],
        }


def collate_fn(batch):
    """Collate function for variable-length sequences."""
    token_ids_list = [item["token_ids"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    max_len = max(len(ids) for ids in token_ids_list)

    batch_size = len(batch)
    padded_token_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, token_ids in enumerate(token_ids_list):
        seq_len = len(token_ids)
        padded_token_ids[i, :seq_len] = torch.tensor(token_ids, dtype=torch.long)

    return {
        "token_ids": padded_token_ids,
        "labels": labels,
    }

"""PyTorch model for language detection."""

import torch
import torch.nn as nn


class LanguageDetectionModel(nn.Module):
    """Simple projection model for language detection.

    Architecture:
        Input: embeddings (batch, seq_len, hidden_dim), token_ids (batch, seq_len)
        1. Apply learnable per-token weights
        2. Dropout
        3. Linear projection: hidden_dim -> n_languages
        4. Pooling over sequence dimension (max/average/logsumexp/geometric/harmonic)
        Output: logits (batch, n_languages)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_languages: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        dropout: float = 0.1,
        pooling: str = "max",
        token_mask: torch.Tensor | None = None,
    ):
        """Initialize language detection model.

        Args:
            hidden_dim: Input embedding dimension
            n_languages: Number of target languages
            vocab_size: Vocabulary size (for per-token weights)
            embeddings: Static embeddings (vocab_size, hidden_dim) - will be stored on GPU
            dropout: Dropout probability
            pooling: Pooling strategy ('max', 'average', 'logsumexp', 'geometric', 'harmonic')
            token_mask: Optional boolean mask (vocab_size,) where False = zero weight.
                        Applied during initialization to zero-weight under-represented tokens.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_languages = n_languages
        self.vocab_size = vocab_size
        self.pooling = pooling

        # Register embeddings as buffer (non-trainable, auto-moved to GPU with model)
        # This is MUCH faster than CPU memmap lookup!
        self.register_buffer("embeddings", embeddings)

        # Learnable per-token weights (initialized to 1.0)
        # Shape: (vocab_size, 1) so we can broadcast multiply with embeddings
        self.token_weights = nn.Parameter(torch.ones(vocab_size, 1))

        # Apply token mask if provided (boolean, used during pooling)
        self.register_buffer("token_mask", None)
        if token_mask is not None:
            if token_mask.shape[0] != vocab_size:
                raise ValueError(
                    f"Token mask shape {token_mask.shape} doesn't match vocab_size {vocab_size}"
                )
            mask_bool = token_mask.view(-1, 1).bool()
            self.register_buffer("token_mask", mask_bool)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, n_languages)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: Token IDs (batch, seq_len)

        Returns:
            Logits for each language (batch, n_languages)
        """
        # token_ids: (batch, seq_len)

        # Lookup embeddings on GPU (FAST!)
        embeddings = self.embeddings[token_ids]  # (batch, seq_len, hidden_dim)

        # Apply learnable per-token weights
        # token_weights[token_ids]: (batch, seq_len, 1)
        token_weights = self.token_weights
        weighted_embeddings = embeddings * token_weights[token_ids]

        x = self.dropout(weighted_embeddings)

        # Project each token to language logits
        x = self.projection(x)  # (batch, seq_len, n_languages)

        # Mask out disallowed tokens by setting logits to large negative before pooling
        if self.token_mask is not None:
            allowed = self.token_mask.squeeze(1)[token_ids]  # (batch, seq_len)
            x = x.masked_fill(~allowed.unsqueeze(-1), -1e9)

        # Apply pooling over sequence dimension
        if self.pooling == "logsumexp":
            # LogSumExp: smooth differentiable approximation of max
            x = torch.logsumexp(x, dim=1)  # (batch, n_languages)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return x

    def get_projection_matrix(self) -> torch.Tensor:
        """Get the projection matrix weights.

        Returns:
            Projection weight matrix (n_languages, hidden_dim)
        """
        return self.projection.weight.data

    def get_projection_bias(self) -> torch.Tensor:
        """Get the projection bias.

        Returns:
            Projection bias (n_languages,)
        """
        return self.projection.bias.data

    def get_token_weights(self) -> torch.Tensor:
        """Get the learnable token weights.

        Returns:
            Token weights (vocab_size, 1)
        """
        if self.token_mask is None:
            return self.token_weights.data
        return self.token_weights.data * self.token_mask

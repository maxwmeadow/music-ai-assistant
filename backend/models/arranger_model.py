"""
Arranger Transformer Model

This model predicts complementary musical tracks (bass, pads, counter-melody)
based on user-provided melody/chord sequences. Uses encoder-only transformer
architecture with multi-instrument output heads.

Architecture:
    - Input: Note sequences (pitch, start, duration, velocity)
    - Encoder: Multi-head self-attention transformer
    - Output: Multi-instrument predictions (bass, pads, counter-melody)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple


class ArrangerTransformer(nn.Module):
    """
    Transformer-based model for musical arrangement generation.

    Predicts complementary tracks (bass, pads, counter-melody) given input sequences.
    Uses encoder-only architecture with learnable positional encoding.

    Args:
        input_dim (int): Number of input features per note (default: 4)
                        [pitch, start, duration, velocity]
        model_dim (int): Hidden dimension size for transformer (default: 256)
        num_heads (int): Number of attention heads (default: 8)
        num_layers (int): Number of transformer encoder layers (default: 6)
        num_instruments (int): Number of output instruments (default: 3)
                              [bass, pads, counter-melody]
        max_seq_len (int): Maximum sequence length (default: 512)
        dropout (float): Dropout probability (default: 0.1)
        dim_feedforward (int): Hidden dimension in feedforward network (default: 1024)
    """

    def __init__(
        self,
        input_dim: int = 4,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_instruments: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        dim_feedforward: int = 1024
    ):
        super().__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_instruments = num_instruments
        self.max_seq_len = max_seq_len
        self.output_dim = num_instruments * input_dim  # 3 instruments * 4 features = 12

        # Input projection: project input features to model dimension
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Learnable positional encoding
        # Shape: (1, max_seq_len, model_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        # Optional: Instrument type embedding (can be used to condition on input instrument)
        self.instrument_embedding = nn.Embedding(10, model_dim)  # Support up to 10 instrument types

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Input shape: (batch, seq, feature)
            norm_first=True    # Pre-LN for better training stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim)
        )

        # Multi-instrument output head
        # Projects from model_dim to output_dim (3 instruments * 4 features)
        self.output_head = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, self.output_dim)
        )

        # Layer norm before output head
        self.output_norm = nn.LayerNorm(model_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_features: torch.Tensor,
        instrument_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Arranger Transformer.

        Args:
            input_features (torch.Tensor): Input note sequences
                                          Shape: (batch, seq_len, input_dim)
            instrument_ids (torch.Tensor, optional): Instrument type IDs for conditioning
                                                     Shape: (batch,)
            src_key_padding_mask (torch.Tensor, optional): Mask for padding tokens
                                                          Shape: (batch, seq_len)
                                                          True for padding positions

        Returns:
            torch.Tensor: Multi-instrument predictions
                         Shape: (batch, seq_len, output_dim)
                         output_dim = num_instruments * input_dim (12 for 3 instruments)
        """
        batch_size, seq_len, _ = input_features.shape

        # Validate sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        # 1. Project input features to model dimension
        x = self.input_projection(input_features)  # (batch, seq_len, model_dim)

        # 2. Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]  # (batch, seq_len, model_dim)

        # 3. Optional: Add instrument conditioning
        if instrument_ids is not None:
            # Broadcast instrument embedding across sequence
            inst_emb = self.instrument_embedding(instrument_ids)  # (batch, model_dim)
            inst_emb = inst_emb.unsqueeze(1)  # (batch, 1, model_dim)
            x = x + inst_emb  # Broadcasting adds to all time steps

        # 4. Pass through transformer encoder
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, model_dim)

        # 5. Apply layer norm before output projection
        x = self.output_norm(x)  # (batch, seq_len, model_dim)

        # 6. Generate multi-instrument outputs
        output = self.output_head(x)  # (batch, seq_len, output_dim)

        return output

    def predict_instruments(
        self,
        input_features: torch.Tensor,
        instrument_ids: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method that returns outputs split by instrument.

        Args:
            input_features: Input note sequences (batch, seq_len, input_dim)
            instrument_ids: Optional instrument type IDs (batch,)
            src_key_padding_mask: Optional padding mask (batch, seq_len)

        Returns:
            Dict mapping instrument names to their predictions:
                - 'bass': (batch, seq_len, 4)
                - 'pads': (batch, seq_len, 4)
                - 'counter_melody': (batch, seq_len, 4)
        """
        # Get full output
        output = self.forward(input_features, instrument_ids, src_key_padding_mask)

        # Split into instruments (each gets 4 features: pitch, start, duration, velocity)
        bass = output[:, :, 0:4]
        pads = output[:, :, 4:8]
        counter_melody = output[:, :, 8:12]

        return {
            'bass': bass,
            'pads': pads,
            'counter_melody': counter_melody
        }

    def get_attention_weights(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Extract attention weights from a specific transformer layer.
        Useful for visualization and interpretability.

        Args:
            layer_idx: Index of layer to extract from (default: -1 for last layer)

        Returns:
            Attention weights if available, else None
        """
        # Note: PyTorch's TransformerEncoder doesn't expose attention weights by default
        # This would require custom implementation or hooks for full access
        # Placeholder for future enhancement
        return None

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration information."""
        return {
            'input_dim': self.input_dim,
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_instruments': self.num_instruments,
            'output_dim': self.output_dim,
            'max_seq_len': self.max_seq_len,
            'total_parameters': self.count_parameters()
        }


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask for variable-length sequences.

    Args:
        lengths: Actual lengths of each sequence in batch (batch_size,)
        max_len: Maximum sequence length

    Returns:
        Boolean mask where True indicates padding positions
        Shape: (batch_size, max_len)
    """
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
    mask = mask >= lengths.unsqueeze(1)
    return mask


if __name__ == "__main__":
    # Quick sanity check
    print("ArrangerTransformer Model")
    print("-" * 50)

    model = ArrangerTransformer(
        input_dim=4,
        model_dim=256,
        num_heads=8,
        num_layers=6,
        num_instruments=3
    )

    print(f"Total parameters: {model.count_parameters():,}")
    print("\nModel configuration:")
    for key, value in model.get_model_info().items():
        print(f"  {key}: {value}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randn(batch_size, seq_len, 4)

    print(f"\nTest input shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test instrument splitting
    instruments = model.predict_instruments(dummy_input)
    print("\nInstrument outputs:")
    for name, tensor in instruments.items():
        print(f"  {name}: {tensor.shape}")

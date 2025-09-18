import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple
from einops import einsum, rearrange

class Linear(nn.Module):
    """Applies a linear transformation to the input: y = xA^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the linear module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If True, includes a bias term. Defaults to False.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        ...


class Embedding(nn.Module):
    """A lookup table that maps indices to embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Looks up embedding vectors for token IDs.

        Args:
            token_ids (torch.Tensor): Input tensor of shape (...).

        Returns:
            torch.Tensor: Output tensor of shape (..., embedding_dim).
        """
        ...

class RMSNorm(nn.Module):
    """Applies Root Mean Square Layer Normalization (RMSNorm)."""  

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        ...

class SwiGLU(nn.Module):
    """Applies the SwiGLU feedforward transformation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the SwiGLU module.

        Args:
            d_model (int): Hidden dimension of the model.
            d_ff (int): Inner dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the SwiGLU transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        ...

class RoPE(nn.Module):
    """Applies Rotary Position Embeddings (RoPE)."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the RoPE module.

        Args:
            theta (float): Θ value for the rotary embedding.
            d_k (int): Dimension of query and key vectors.
            max_seq_len (int): Maximum sequence length supported.
            device (torch.device, optional): Device to store buffers. Defaults to None.
        """
        ...

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Applies rotary position embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len)
                specifying token positions.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_k).
        """
        ...

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax activation function.

    Applies the softmax function to the input tensor along the specified dimension.

    Args:
    x: Input tensor.
    dim: Dimension along which softmax will be computed. Defaults to -1.

    Returns:
    Tensor with softmax applied along the specified dimension.
    """
    ...

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Scaled dot-product attention function.

    Args:
        query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: Tensor of shape (batch_size, ..., seq_len_k, d_k)  
        value: Tensor of shape (batch_size, ..., seq_len_v, d_v)
        mask: Boolean tensor of shape (seq_len_q, seq_len_k) or broadcastable shape

    Returns:
        Tensor of shape (batch_size, ..., seq_len_q, d_v)
    """
    ...

class CasualMultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """Initializes the attention module.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            device (torch.device, optional): Device to store parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE when enabled. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers.
                Defaults to None.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Applies causal multi-head self-attention.

        Args:
        x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor, optional): Tensor of shape (..., seq_len)
                with token positions; required if `use_rope` is True. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        ...

class TransformerBlock(nn.Module):
    """A single Transformer block with self-attention and feedforward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        """Initializes the Transformer block.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE in self-attention. Defaults to False.
            theta (float, optional): Θ parameter for RoPE. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers. Defaults to None.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        ...

class TransformerLM(nn.Module):
    """A Transformer-based language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_rope: bool = False,
        theta: Optional[float] = None,
    ) -> None:
        """Initializes the Transformer language model.

        Args:
            vocab_size (int): Vocabulary size for token embeddings.
            context_length (int): Maximum sequence length for positional encodings.
            num_layers (int): Number of Transformer blocks.
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden dimension of the feedforward layer.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE. Defaults to None.
        """
        ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Applies the Transformer language model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (..., seq_len).

        Returns:
            torch.Tensor: Logits of shape (..., seq_len, vocab_size).
        """
        ...

class LSTMCell(nn.Module):
    """A single Long Short-Term Memory (LSTM) cell."""

    def __init__(
        self,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the LSTM cell.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_state, cell_state), each of shape (batch_size, d_model).
                If None, both are initialized to zeros. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The next (hidden_state, cell_state),
            each of shape (batch_size, d_model).
        """
        ...

class LSTM(nn.Module):
    """Multi-layer LSTM network with batch-first input."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the multi-layer LSTM.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Applies the multi-layer LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_states, cell_states), each of shape
                (num_layers, batch_size, d_model). Defaults to None.

        Returns:
            tuple:
                - torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
                - tuple[torch.Tensor, torch.Tensor]: Next (hidden_states, cell_states),
                    each of shape (num_layers, batch_size, d_model).
        """
        ...

class LSTMLM(nn.Module):
    """LSTM-based language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        """Initializes the LSTM language model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Applies the LSTM language model.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_states, cell_states), each of shape
                (num_layers, batch_size, d_model). Defaults to None.

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        ...
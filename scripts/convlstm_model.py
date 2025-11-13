"""
Copyright (c) 2025, Mohammad Mosavat
Email Address: smosavat@crimson.ua.edu
All rights reserved.

This code is released under MIT License

Description:
ConvLSTM Model for Spatial-Temporal Runoff Prediction

OPTIMIZED VERSION:
- Vectorized pooling operation (no Python loops)
- Support for mixed precision training
- Optional JIT compilation
- Efficient tensor operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.
    
    Based on: https://arxiv.org/abs/1506.04214
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden state channels
            kernel_size: Size of convolutional kernel
            bias: Whether to use bias
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, x, hidden_state):
        """
        Args:
            x: (batch, input_dim, height, width)
            hidden_state: tuple of (h, c)
                h: (batch, hidden_dim, height, width)
                c: (batch, hidden_dim, height, width)
        
        Returns:
            h_next: (batch, hidden_dim, height, width)
            c_next: (batch, hidden_dim, height, width)
        """
        h_cur, c_cur = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h_cur], dim=1)  # (batch, input_dim + hidden_dim, H, W)
        
        # Convolve and split into gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gate activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell candidate
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """Initialize hidden state."""
        height, width = image_size
        device = self.conv.weight.device
        dtype = self.conv.weight.dtype  # Match dtype for mixed precision
        
        h = torch.zeros(batch_size, self.hidden_dim, height, width, 
                       device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, 
                       device=device, dtype=dtype)
        
        return (h, c)


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM.
    """
    
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, 
                 bias=True, return_all_layers=False):
        """
        Args:
            input_dim: Number of input channels
            hidden_dims: List of hidden dimensions for each layer
            kernel_sizes: List of kernel sizes for each layer
            num_layers: Number of ConvLSTM layers
            bias: Whether to use bias
            return_all_layers: If True, return outputs from all layers
        """
        super(ConvLSTM, self).__init__()
        
        # Verify parameters
        if not len(hidden_dims) == len(kernel_sizes) == num_layers:
            raise ValueError("Inconsistent list lengths")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        
        # Create layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_sizes[i],
                    bias=bias
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (batch, seq_len, input_dim, height, width)
            hidden_state: Optional initial hidden state
        
        Returns:
            layer_output_list: List of outputs from each layer
            last_state_list: List of final (h, c) for each layer
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Iterate through sequence
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    x=cur_layer_input[:, t, :, :, :],
                    hidden_state=(h, c)
                )
                output_inner.append(h)
            
            # Stack temporal outputs
            layer_output = torch.stack(output_inner, dim=1)  # (batch, seq_len, hidden_dim, H, W)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)
            )
        return init_states


class SpatialRunoffModel(nn.Module):
    """
    Complete model: ConvLSTM ? Global Pooling ? MLP ? Runoff prediction.
    
    OPTIMIZED VERSION with:
    - Vectorized pooling (no Python loops!)
    - Mixed precision support
    - Optional mask-weighted pooling
    """
    
    def __init__(
        self,
        input_channels=32,
        hidden_dims=[64, 128, 64],
        kernel_sizes=[5, 3, 3],
        mlp_hidden_dims=[128, 64],
        dropout=0.2,
        use_mask=False
    ):
        """
        Args:
            input_channels: Number of input channels (32 in your case)
            hidden_dims: List of hidden dimensions for ConvLSTM layers
            kernel_sizes: List of kernel sizes for ConvLSTM layers
            mlp_hidden_dims: List of hidden dimensions for MLP
            dropout: Dropout rate
            use_mask: If True, use mask for weighted pooling
        """
        super(SpatialRunoffModel, self).__init__()
        
        self.use_mask = use_mask
        num_layers = len(hidden_dims)
        
        # ConvLSTM encoder
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=hidden_dims,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            bias=True,
            return_all_layers=False
        )
        
        # MLP head
        mlp_layers = []
        in_dim = hidden_dims[-1]  # Output from last ConvLSTM layer
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final output layer
        mlp_layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
    
    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, channels, height, width) - input sequence
            mask: (batch, height, width) - basin mask (optional)
        
        Returns:
            output: (batch, seq_len, 1) - predicted runoff for ALL timesteps
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Pass through ConvLSTM
        layer_output_list, _ = self.convlstm(x)
        
        # Get output from last layer: (batch, seq_len, hidden_dim, H, W)
        convlstm_output = layer_output_list[0]
        
        # OPTIMIZED: Vectorized pooling - no Python loops!
        if self.use_mask and mask is not None:
            # Expand mask for all timesteps and channels
            # mask: (batch, H, W) ? (batch, 1, 1, H, W)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            
            # Apply mask and compute weighted average
            masked_output = convlstm_output * mask_expanded
            
            # Sum over spatial dimensions
            spatial_sum = masked_output.sum(dim=(-2, -1))  # (batch, seq_len, hidden_dim)
            
            # Compute mask normalization factor
            mask_sum = mask_expanded.sum(dim=(-2, -1)).clamp(min=1e-8)  # (batch, 1, 1)
            
            # Weighted average
            pooled = spatial_sum / mask_sum
        else:
            # VECTORIZED: Simple global average pooling
            # This single line replaces the entire Python loop!
            pooled = convlstm_output.mean(dim=(-2, -1))  # (batch, seq_len, hidden_dim)
        
        # Pass ENTIRE SEQUENCE through MLP
        # Reshape for batch processing through MLP
        batch_seq_len = batch_size * seq_len
        pooled_reshaped = pooled.reshape(batch_seq_len, -1)
        
        # Apply MLP to all timesteps at once
        output_reshaped = self.mlp(pooled_reshaped)  # (batch*seq_len, 1)
        
        # Reshape back to (batch, seq_len, 1)
        output = output_reshaped.reshape(batch_size, seq_len, 1)
        
        return output


# Optional: JIT-compiled version for additional speedup
def create_jit_model(**kwargs):
    """
    Create a JIT-compiled version of the model for production.
    
    Usage:
        model = create_jit_model(input_channels=32, ...)
        model = torch.jit.script(model)
    """
    model = SpatialRunoffModel(**kwargs)
    # Note: JIT compilation happens when you call torch.jit.script(model)
    return model


# Test the optimized model
if __name__ == "__main__":
    import time
    
    # Create sample data
    batch_size = 32  # Increased from 4!
    seq_len = 365
    channels = 32
    height = 61
    width = 61
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, seq_len, channels, height, width).to(device)
    mask = torch.ones(batch_size, height, width).to(device)
    
    # Create optimized model
    model = SpatialRunoffModel(
        input_channels=32,
        hidden_dims=[64, 128, 64],
        kernel_sizes=[5, 3, 3],
        mlp_hidden_dims=[128, 64],
        dropout=0.2,
        use_mask=False
    ).to(device)
    
    # Warmup
    for _ in range(3):
        _ = model(x, mask)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.cuda.amp.autocast():  # Mixed precision
        output = model(x, mask)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    
    print(f"Optimized Model Performance:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Forward pass time: {elapsed:.3f} seconds")
    print(f"  Throughput: {batch_size/elapsed:.1f} basins/second")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
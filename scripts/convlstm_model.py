"""
ConvLSTM Model for Spatial-Temporal Runoff Prediction

Architecture A: ConvLSTM Encoder ? Global Pooling ? MLP
- Input: (batch, seq_len, channels, height, width)
- ConvLSTM: Extract spatial-temporal features
- Masked Global Average Pooling: Aggregate spatial info (respecting basin mask)
- MLP: Predict scalar runoff value
"""

import torch
import torch.nn as nn


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
        
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        
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
    Complete model: ConvLSTM ? Masked Pooling ? MLP ? Runoff prediction.
    
    Architecture:
        1. ConvLSTM: Extract spatial-temporal features
        2. Take last timestep output
        3. Masked global average pooling
        4. MLP to predict runoff
    """
    
    def __init__(
        self,
        input_channels=32,
        hidden_dims=[64, 128, 64],
        kernel_sizes=[5, 3, 3],
        mlp_hidden_dims=[128, 64],
        dropout=0.2
    ):
        """
        Args:
            input_channels: Number of input channels (32 in your case)
            hidden_dims: List of hidden dimensions for ConvLSTM layers
            kernel_sizes: List of kernel sizes for ConvLSTM layers
            mlp_hidden_dims: List of hidden dimensions for MLP
            dropout: Dropout rate
        """
        super(SpatialRunoffModel, self).__init__()
        
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
    
    def forward(self, x, mask):
        """
        Args:
            x: (batch, seq_len, channels, height, width) - input sequence
            mask: (batch, height, width) - basin mask (1=valid, 0=invalid)
        
        Returns:
            output: (batch, 1) - predicted runoff for each basin
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Pass through ConvLSTM
        layer_output_list, _ = self.convlstm(x)
        
        # Get output from last layer: (batch, seq_len, hidden_dim, H, W)
        convlstm_output = layer_output_list[0]
        
        # Take last timestep: (batch, hidden_dim, H, W)
        last_output = convlstm_output[:, -1, :, :, :]
        
        # Masked global average pooling
        # Expand mask dimensions: (batch, 1, H, W)
        mask_expanded = mask.unsqueeze(1)
        
        # Mask the features
        masked_features = last_output * mask_expanded  # (batch, hidden_dim, H, W)
        
        # Sum over spatial dimensions
        spatial_sum = masked_features.sum(dim=(-2, -1))  # (batch, hidden_dim)
        
        # Count valid pixels per batch
        valid_pixels = mask_expanded.sum(dim=(-2, -1))  # (batch, 1)
        
        # Average (avoid division by zero)
        pooled = spatial_sum / (valid_pixels + 1e-8)  # (batch, hidden_dim)
        
        # Pass through MLP
        output = self.mlp(pooled)  # (batch, 1)
        
        return output


# Test the model
if __name__ == "__main__":
    # Create sample data
    batch_size = 4
    seq_len = 365
    channels = 32
    height = 61
    width = 61
    
    x = torch.randn(batch_size, seq_len, channels, height, width)
    mask = torch.randint(0, 2, (batch_size, height, width)).float()
    
    # Create model
    model = SpatialRunoffModel(
        input_channels=32,
        hidden_dims=[64, 128, 64],
        kernel_sizes=[5, 3, 3],
        mlp_hidden_dims=[128, 64],
        dropout=0.2
    )
    
    # Forward pass
    output = model(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
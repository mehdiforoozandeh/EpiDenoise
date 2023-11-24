import torch
from torch import nn
import torch.optim as optim

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super(RelativePositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_pos_emb = nn.Embedding(self.max_len*2, self.d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(-seq_len+1, seq_len, device=x.device).unsqueeze(-1)
        pos_emb = self.rel_pos_emb(pos + self.max_len)
        return pos_emb

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        return (attention_weights * x).sum(dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super(PositionalEncoding, self).__init__()

        # Create a long enough `pe` matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register `pe` as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')  # We will handle reduction ourselves

    def forward(self, pred, target, mask):
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()  # Only consider non-masked values

#________________________________________________________________________________________________________________________#

class MaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(MaskedConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x, mask):
        not_mask = mask.clone()
        not_mask = ~mask
        x = x.permute(0,2,1)
        not_mask = not_mask.permute(0,2,1)
        x = x * not_mask
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

class DualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DualConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.data_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, mask):
        mask = mask.clone()

        x = x.permute(0,2,1)
        mask = mask.permute(0,2,1)

        x = self.data_conv(x)
        mask = self.mask_conv(mask.float())

        x = x * mask
        x = x.permute(0,2,1)
        return x

#________________________________________________________________________________________________________________________#

class MaskedConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(MaskedConvEncoder, self).__init__()
        self.masked_conv = MaskedConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.masked_conv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class DualConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(DualConvEncoder, self).__init__()
        
        self.dualconv = DualConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src, src_mask):
        src = self.dualconv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

#________________________________________________________________________________________________________________________#

# Function to mask a certain percentage of the data
def mask_data(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15):
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)
    # If chunk is True, mask the data in chunks
    
    # Get the sequence length from the data tensor
    
    if chunk:
        # Calculate the size of each chunk
        chunk_size = int(mask_percentage * seq_len / n_chunks)
    else: 
        chunk_size = 1
        n_chunks =  int(mask_percentage * seq_len)

    # Initialize an empty list to store the start indices
    start_indices = []
    while len(start_indices) < n_chunks:
        # Generate a random start index
        start = torch.randint(0, seq_len - chunk_size, (1,))
        # Check if the chunk overlaps with any existing chunks
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask
    return masked_data, mask

def mask_missing(data, missing_features_ind, mask_value=-1):
    mask = torch.zeros_like(data, dtype=torch.bool)

    # Loop over the missing feature ids
    for id in missing_features_ind:
        # Set the mask values for the current chunk to True
        mask[:, :, id] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask
    return masked_data, mask

# Function to train the model
def train(model, data, missing_features_ind=[0, 3, 5, 6], epochs=100, mask_percentage=0.15, chunk=False, n_chunks=1):
    # Initializing the loss function
    criterion = nn.MSELoss()
    # Initializing the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # Looping over the number of epochs
    for epoch in range(epochs):
        # Resetting the gradients of the model parameters
        optimizer.zero_grad()

        # If missing_features_ind is not empty, create a mask for the missing data
        if len(missing_features_ind) > 0: 
            padded_data, pad = mask_missing(data, missing_features_ind)

        # If missing_features_ind is empty, create a mask of False values with the same shape as the data
        else:
            # Creating a mask of False values with the same shape as the data
            padded_data = data.clone()
            pad = torch.zeros_like(data, dtype=torch.bool)

        # Masking a subset of the input data
        masked_data, mask = mask_data(padded_data, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)

        # Update mask variable such that if for an entry in pad, pad == True, mask should be False
        mask = mask & ~pad

        # Combining the two masks
        combined_mask = mask | pad
        
        # Getting the output of the model
        output = model(masked_data, combined_mask)

        # Computing the loss only on the masked subset of the input data
        loss = criterion(output[mask], data[mask])

        # Backpropagating the gradients
        loss.backward()
        # Updating the model parameters
        optimizer.step()

# The main function
def main():
    # Defining the hyperparameters
    input_dim = output_dim = 10
    nhead = 5
    hidden_dim = 16
    nlayers = 2
    epochs = 500
    seq_len = 100
    n_samples = 80
    mask_percentage = 0.20
    out_channel = 10
    kernel_size = 5
    chunk = True
    n_chunks = 2

    # Creating an instance of the TransformerEncoder model
    # model = MaskedConvEncoder(input_dim, nhead, hidden_dim, nlayers, output_dim, out_channel, kernel_size=kernel_size)
    model = DualConvEncoder(input_dim, nhead, hidden_dim, nlayers, output_dim, out_channel, kernel_size=kernel_size)

    # Generating some random data
    data = torch.abs(torch.randn(n_samples, seq_len, input_dim))

    # Training the model
    train(model, data, epochs=epochs, mask_percentage=mask_percentage, chunk=chunk, n_chunks=n_chunks)

# Calling the main function
if __name__ == "__main__":
    main()

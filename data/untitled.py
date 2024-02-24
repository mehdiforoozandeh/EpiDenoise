import torch

# Example parameters
N, L, F = 30, 8000, 35  # Size of the input tensor
m = 200  # Number of random points to select
C = 200  # Total length of the sequence

# Simulating an input tensor of size (N, L, F)
input_tensor = torch.randn(N, L, F)

# Select m random points along the L dimension
random_points = torch.randint(low=0, high=L, size=(m,))

# Initialize the output tensor with padding value (e.g., 0) and the padding tracker
x_batch = torch.zeros((m*N, C, F))

for i, point in enumerate(random_points):
    start = point - C // 2
    end = point + C // 2
    adjusted_start = max(start, 0)
    adjusted_end = min(end, L)
    ival = input_tensor[:, adjusted_start:adjusted_end, :]
    if ival.shape[1] < C:
        pad_length = C - ival.shape[1]
        pad = torch.full((ival.shape[0], pad_length, ival.shape[2]), -3)
        if start < 0:
            ival = torch.cat([pad, ival], dim=1)
        elif end > L:
            ival = torch.cat([ival, pad], dim=1)
    x_batch[i*N:(i+1)*N] = ival

x_batch_pad = (x_batch==-3)
x_batch_pad.sum(dim=2)


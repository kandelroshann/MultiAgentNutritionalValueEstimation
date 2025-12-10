import torch

# Create a 1D tensor
x = torch.tensor([1, 2, 3])
print(f"Original tensor: {x}, shape: {x.shape}")

# Unsqueeze at dimension 0 (add a new dimension at the beginning)
y = torch.unsqueeze(x, 0)
print(f"Unsqueeze at dim 0: {y}, shape: {y.shape}")

# Unsqueeze at dimension 1 (add a new dimension at the end)
z = torch.unsqueeze(x, 1)
print(f"Unsqueeze at dim 1: {z}, shape: {z.shape}")

# Unsqueeze a 2D tensor
a = torch.tensor([[1, 2], [3, 4]])
print(f"Original 2D tensor: {a}, shape: {a.shape}")

# Unsqueeze at dimension 0 (add a new dimension at the beginning)
b = torch.unsqueeze(a, 0)
print(f"Unsqueeze 2D at dim 0: {b}, shape: {b.shape}")
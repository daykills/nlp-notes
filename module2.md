 # Multiclass Classification

 - Multiclass Perceptron

predict ŷ = argmax_k w_k · x  
if ŷ != y:
    w_y      += x
    w_ŷ      -= x

```
initialize w[1..K] = 0
for epoch = 1..T:
    shuffle(training_set)
    for (x, y) in training_set:
        y_hat = argmax_k dot(w[k], x)       
        if y_hat != y:
            w[y]     = w[y]     + x
            w[y_hat] = w[y_hat] - x
return w
```
### Neural Net Implementation
- Pytorch: define computations that provides easy access to drivatives or gradients.


### Neural Net Training, Optimization
  - batching: Update the model after computing the loss/gradients on each batch.
            This reduces variance compared to single-sample updates and uses less memory than full-batch.
  ```
  import torch
  from torch.utils.data import DataLoader, TensorDataset
  
  # Example dataset
  X = torch.randn(1000, 20)   # 1000 samples, 20 features
  y = torch.randint(0, 2, (1000,))  # binary labels
  
  # Create DataLoader with batching
  dataset = TensorDataset(X, y)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
  
  # Training loop
  for epoch in range(10):
      for batch_X, batch_y in dataloader:
          # Forward pass
          y_pred = model(batch_X)
          loss = criterion(y_pred, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
- Initialization (not zero, not large)
  1. Zero Initialization: Not recommended for weights (all neurons become symmetric and learn the same thing). Biases can safely be     initialized to 0.
  2. Initialize too large and cells are saturated.
  3. Use initializzer: Xavier Guerrero
- Dropout: zero out parts of the network during training to prevent overfitting, use whole network at test time.
- Optimizer: Adam



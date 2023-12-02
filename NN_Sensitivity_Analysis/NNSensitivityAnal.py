
import torch

# Define a simple model
class NNmodel(torch.nn.Module):
    def __init__(self):

        super(NNmodel, self).__init__()
        self.l1 = torch.nn.Linear(3, 3)
        self.l2 = torch.nn.Linear(3, 2)

        self.relu    = torch.nn.ReLU()

    def forward(self, x):

        x = self.relu(self.l1(x))
        x = (self.l2(x))

        return x

# train data
X = torch.tensor([[0.1, -20.2, 1.8], [-10.1, 3.2, 2], [-0.3, -9.2, 4]], requires_grad=True)
y = torch.tensor([[1.0, 2.0]   , [1.0, 2.0]   , [2.0, 4.0]], requires_grad=True)

# Instantiate the model
model = NNmodel()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

'''
# Train the model
model.train()
for epoch in range(2000):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model, 'test_model.pt')
'''
model = torch.load('test_model.pt')



# ----------------------------
#   Compute sensitivity: Ex
# ----------------------------
print('\n---------')
X        = torch.tensor([[-10.3, -9.2, 2.2]], requires_grad=True)
y_pred   = model(X)

y_pred_m = y_pred.mean()
y_pred_m.backward()

print('outputs:   ', y_pred_m)
print('X.grad:    ', X.grad)
print('---------\n')


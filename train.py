import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assuming you have your dataset loaded as numpy arrays:
# xtrain, ytrain, xval, yval, xtest, ytest
image_size = 32
batch_size = 10

class NeuralNet(nn.Module):
    def construct_CNN(self):
        # Define encoder-decoder CNN
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # Convolution + activation
            nn.ReLU(),
            nn.MaxPool2d(2,2),                 # Downsample
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.ConvTranspose2d(64,32,2,stride=2), # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,2,stride=2),
            nn.ReLU(),
            nn.Conv2d(16,1,1),                  # Output single channel mask
            nn.Flatten()                         # Flatten to 1D
        )

    def __init__(self):
        super(NeuralNet,self).__init__()
        self.construct_CNN()
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x, predict=True):
        logits = self.model(x)
        if predict:
            return torch.sigmoid(logits)  # sigmoid for binary mask
        return logits

    def get_loss(self, x, yt):
        # Flatten target mask
        yt = yt.view(-1,image_size*image_size)
        ylogits = self.forward(x,predict=False)
        # Binary cross-entropy loss with logits
        loss = F.binary_cross_entropy_with_logits(ylogits, yt, reduction='sum')
        return loss

    def train_step(self, xbatch, ybatch):
        # Zero gradients
        self.optimizer.zero_grad()
        # Compute loss
        loss = self.get_loss(xbatch, ybatch)
        # Backpropagation
        loss.backward()
        # Update parameters
        self.optimizer.step()
        return loss.item()

    def predict(self,x):
        with torch.no_grad():
            y = self.forward(x,predict=True)
        return y.cpu().numpy()

# Intersection over Union metric
def iou(ytrue, ypred):
    yp = ypred > 0.5 + 0
    intersect = np.sum(np.minimum(yp, ytrue),1)
    union = np.sum(np.maximum(yp, ytrue),1)
    return np.average(intersect / (union+0.0))

# Convert numpy arrays to tensors
xtrain_tensor = torch.tensor(xtrain,dtype=torch.float32).view(-1,image_size,image_size,3).permute(0,3,1,2)
ytrain_tensor = torch.tensor(ytrain,dtype=torch.float32)
train_loader = DataLoader(TensorDataset(xtrain_tensor,ytrain_tensor), batch_size=batch_size, shuffle=True)

# Set device and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)

best_val_iou = -1
best_model_path = "models/best_model.pth"

# Training loop
for epoch in range(1, 1001):
    total_loss = 0
    for batch in train_loader:
        xbatch, ybatch = batch
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)
        total_loss += model.train_step(xbatch,ybatch)

    # TODO: add validation set IOU calculation
    print(f"Epoch {epoch}, Loss: {total_loss:.3f}")
    # Save the best model (optional)
    # torch.save(model.state_dict(), best_model_path)

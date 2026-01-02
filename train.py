# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set image size globally
image_size = 32

class NeuralNet(nn.Module):
    def construct_CNN(self):
        # Encoder-decoder CNN for segmentation
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Flatten()
        )

    def construct_MLP(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size * 3, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 20),
            nn.ReLU(),
            nn.Linear(20, image_size * image_size)
        )

    def __init__(self, network_type='CNN'):
        super(NeuralNet, self).__init__()
        if network_type == 'MLP':
            self.construct_MLP()
        else:
            self.construct_CNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def forward(self, x, predict=True):
        logits = self.model(x)
        if predict:
            return torch.sigmoid(logits)
        return logits

    def get_loss(self, x, yt):
        yt = yt.view(-1, image_size * image_size)
        ylogits = self.forward(x, predict=False)
        loss = F.binary_cross_entropy_with_logits(ylogits, yt, reduction='sum')
        return loss

    def train_step(self, xbatch, ybatch):
        self.optimizer.zero_grad()
        loss = self.get_loss(xbatch, ybatch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            y = self.forward(x, predict=True)
        return y.cpu().numpy()

# Only include data loading / training code if script is run directly
if __name__ == "__main__":
    print("This file is a module. Import NeuralNet in your Notebook or main script.")

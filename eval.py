import torch
import numpy as np
from train import NeuralNet, image_size

# Set device and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)
# Load the best model weights
model.load_state_dict(torch.load("models/best_model.pth"))

# Convert test set to tensor
xtest_tensor = torch.tensor(xtest,dtype=torch.float32).view(-1,image_size,image_size,3).permute(0,3,1,2).to(device)
# Make predictions
ypred_test = model.predict(xtest_tensor)

# Visualization helper
import matplotlib.pyplot as plt
def draw(image, mask):
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

# Draw one sample prediction
draw(xtest[0], ypred_test[0])

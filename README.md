# Horse Image Segmentation (PyTorch CNN)

This project is an individual ML project for image segmentation of horses using a convolutional neural network (CNN) in PyTorch.

## 📝 Problem
Segment horse images from the background to produce masks for downstream tasks such as object detection or analysis.

## 🛠 Approach
- Encoder–decoder CNN using Conv2d, MaxPool2d, ConvTranspose2d layers
- Trained on 328 horse images resized to 32×32
- Binary mask segmentation (0 = background, 1 = horse)
- Used Intersection-over-Union (IoU) as evaluation metric
- Achieved **70%+ IOU** on test set

## 📊 Results
- Sample predictions are in `notebooks/horse_segmentation_demo.ipynb`
- Model weights saved as `models/best_model.pth`

## 📁 Repository Structure
horse-segmentation-cnn/
├─ notebooks/
│   └─ horse_segmentation_demo.ipynb  # Demo notebook: sample images + predictions + loss/IOU curves
├─ data/
│   ├─ sample_images/                  # 5~10 sample horse images
│   ├─ sample_masks/                   # Corresponding masks
│   └─ README.md                       # Instructions to download full dataset
├─ models/
│   └─ best_model.pth                  # Optional: small trained model or instructions
├─ train.py                             # Training script (CNN)
├─ eval.py                              # Evaluation script + prediction visualization
└─ README.md                            # Project description

## 🔧 Technologies
Python, PyTorch, NumPy, OpenCV, Matplotlib, Jupyter Notebook

## ⚡ Next Steps
- Try U-Net or attention-based architectures
- Use data augmentation to improve performance
- Extend to multi-class segmentation

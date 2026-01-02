# Horse Image Segmentation (PyTorch CNN)

This project is an individual ML project for image segmentation of horses using a convolutional neural network (CNN) in PyTorch.

## 📝 Problem
The goal is to segment horse images from the background, producing accurate masks for downstream tasks like object detection and analysis.

## 🛠 Approach
- Built an encoder–decoder CNN in PyTorch
- Used transposed convolution layers for upsampling
- Preprocessed and split a custom horse image dataset
- Trained the model to maximize Intersection-over-Union (IoU)

## 📊 Results
- Achieved **70%+ IoU** on the test set
- Visualized predictions with sample images in the `notebooks/` folder
- Documented model architecture, training process, and evaluation metrics in Jupyter Notebook

## 📁 Repository Structure
horse-segmentation-cnn/
├─ notebooks/
│   └─ horse_segmentation_demo.ipynb 
├─ data/
│   └─ sample_images/                
│   └─ sample_masks/                
│   └─ README.md                      
├─ models/
│   └─ best_model.pth                 
├─ train.py                           
├─ eval.py                             
└─ README.md                           


## 🔧 Technologies
- Python, PyTorch, NumPy
- Jupyter Notebook, Matplotlib

## ⚡ Next Steps
- Experiment with U-Net and attention-based models
- Explore data augmentation to improve accuracy
- Extend to multi-class segmentation tasks

<!-- Header Badges -->


---

<h1 align="center">Celebrity Face Recognition</h1>
<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Latest-orange?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=flat-square&logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN--Transformer-purple?style=flat-square)
![MTCNN](https://img.shields.io/badge/Face%20Detection-MTCNN-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Academic](https://img.shields.io/badge/Academic-HNU-yellow?style=flat-square)

</div>
A comprehensive deep learning project for celebrity face recognition and classification using multiple state-of-the-art models including **ArcFace**, **FaceNet**, and **Vision Transformers (ViT)**.

**Project Objective:** Deploy a Deep Learning model using CNNs and Transfer Learning to recognize celebrity identities from facial images.

> 🎓 **Part of the AI-Skill Program** at **Helwan National University (HNU)**

## 📋 Project Overview

This project implements end-to-end facial recognition pipelines with a focus on celebrity identification. It combines advanced face detection (MTCNN), embedding extraction, and classification techniques with extensive analysis and visualization capabilities.

**Key Capabilities:**
- 🎯 Multi-model face recognition (ArcFace, FaceNet, ViT)
- 🔍 Face detection with MTCNN
- 📊 Embedding extraction and clustering
- 🖥️ GUI interface with live webcam support
- 📈 Model interpretability (GRAD-CAM, Attention Rollout)
- 🎬 Live detection and real-time classification

---

## 🏗️ Project Structure

```
celebrity-face-recognition/
├── Arcface/                          # ArcFace model implementations
│   ├── embeddings_and_labels_extraction.ipynb
│   ├── embeddings_and_labels_extractions.ipynb
│   ├── Embeddings_Extractor.ipynb
│   ├── Training.ipynb
│   └── Training final.ipynb
├── FaceNet/                          # FaceNet model implementations
│   ├── Classifier_MTCNN.ipynb
│   ├── Classifier_MTCNN_1K.ipynb
│   ├── Classifier_MTCNN_With_Visualization.ipynb
│   ├── Classifier_Using_FaceNet.ipynb
│   ├── Clustering_Embeddings.ipynb
│   ├── Embeddings_Labels_Extractor.ipynb
│   ├── Embeddings_Using_FaceNet.ipynb
│   ├── GRAD-CAM and Occlusion Heatmap.ipynb
│   ├── GUI_+_Webcam_1K.ipynb
│   └── LIVE_DETECTION.ipynb
├── ViT/                              # Vision Transformer model implementations
│   ├── Classifier with MTCNN.ipynb
│   ├── Classifier with MTCNN with Attention Rollout.ipynb
│   ├── Clustering_Embeddings.ipynb
│   ├── Cosine Similarity.ipynb
│   └── Embeddings Extractor.ipynb
├── Images/                           # Sample outputs and visualizations
│   ├── confusion-matrix.jpg
│   ├── grad-cam.jpg
│   ├── gui.jpg
│   └── webcam.jpg
└── README.md
```

---

## 🔧 Model Development

### Used Pretrained Models

- **FaceNet** — Pretrained CNN trained on large-scale face datasets
- **ViT** — Vision Transformer pretrained on ImageNet-21K
- **ArcFace** — Pretrained CNN trained on large-scale face datasets with additive angular margin

### Training Dataset

- **Source:** Top 1000 celebrities with different angles and positions
- **Size:** 18,184 images
- **Resolution:** 256×256 pixels
- **Link:** [Celebrity-1000 on Hugging Face](https://huggingface.co/datasets/tonyassi/celebrity-1000/)

### Data Preprocessing

- Resized each image to fit model requirements (e.g., 160×160×3 for FaceNet)
- Split data into Train, Test, and Evaluation sets
- Applied augmentation: rotation, flipping, geometric transformations

### Embedding Extraction

- Extracted high-dimensional **Face Embeddings** from pretrained models
- Saved embeddings in optimized formats (.csv, .npy) for faster access
- Normalized embeddings for similarity matching

### Classifier Design

- Input layer matching embedding dimensions
- Hidden fully connected layers for feature transformation
- Output layer with softmax activation for multi-class classification

### Classifier Training

- Trained classifiers on embeddings from training set
- Used celebrity names as target labels
- Optimized over multiple epochs with validation monitoring

---

## 🔧 Models Explained

### 1. **ArcFace (Additive Angular Margin)**
- **Purpose:** State-of-the-art face recognition model using angular-based loss
- **Key Features:**
  - Large-scale face embedding extraction
  - High accuracy on celebrity datasets
  - Efficient GPU computation
- **Notebooks:**
  - Embeddings extraction and processing
  - Training pipelines for custom datasets

### 2. **FaceNet**
- **Purpose:** Deep learning model for face embedding and recognition
- **Key Features:**
  - Face detection with MTCNN
  - Triplet loss-based training
  - Clustering capabilities
  - Interpretability analysis (GRAD-CAM, heatmaps)
- **Notebooks:**
  - Classifiers with MTCNN integration
  - Live webcam detection GUI
  - Embedding clustering and analysis

### 3. **Vision Transformers (ViT)**
- **Purpose:** Transformer-based face recognition and analysis
- **Key Features:**
  - Attention-based mechanisms
  - Modern architecture for visual understanding
  - Interpretability via Attention Rollout
  - Cosine similarity-based matching
- **Notebooks:**
  - Classification with attention visualization
  - Embedding extraction with ViT
  - Attention analysis

---

## � Model Evaluation & Performance

We evaluated each model's performance and compared results across all implemented architectures.

### Performance Metrics

| Model | Accuracy | Precision | Recall |
|:-----:|:--------:|:---------:|:------:|
| **FaceNet** | 97% | 95% | 97% |
| **ArcFace** | 96% | 95% | 95% |
| **ViT** | 81% | 79% | 79% |

### Evaluation Results

- **Best Overall:** FaceNet with 97% accuracy
- **Most Stable:** ArcFace with balanced precision-recall
- **Modern Approach:** ViT with attention-based interpretability

### Confusion Matrix Analysis

<div align="center">
  <img src="Images/confusion-matrix.jpg" alt="Confusion Matrix" width="70%">
</div>

---

## 🔍 Explainability & Interpretability

### GRAD-CAM Visualization
We used **Grad-CAM** to interpret and understand model predictions by highlighting critical regions in facial images.

<div align="center">
  <img src="Images/grad-cam.jpg" alt="GRAD-CAM Heatmap" width="70%">
</div>

### Attention Rollout (ViT)
For Vision Transformers, we visualize attention patterns to understand how the model focuses on different facial regions.

---

## 🖥️ User Interface & Applications

### Gradio GUI
We developed a functional GUI using **Gradio** to apply the FaceNet model for celebrity classification.

<div align="center">
  <img src="Images/gui.jpg" alt="Gradio GUI" width="70%">
</div>

### Real-time Webcam Detection
Live classification support for real-time celebrity identification from webcam feeds.

<div align="center">
  <img src="Images/webcam.jpg" alt="Webcam Detection" width="70%">
</div>

---

## �📦 Dataset

The project uses the **Celebrity-1000** dataset from Hugging Face:
- **Dataset:** `tonyassi/celebrity-1000`
- **Size:** 1000+ celebrity identities
- **Format:** Images with labels
- **Loading:** Streaming support for efficient memory usage

```python
from datasets import load_dataset
dataset = load_dataset("tonyassi/celebrity-1000", split="train", streaming=True)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended)
- Jupyter Notebook or Jupyter Lab

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/celebrity-face-recognition.git
   cd celebrity-face-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -q datasets insightface onnxruntime-gpu opencv-python tensorflow keras_facenet
   ```

3. **GPU Support (Optional):**
   - For CUDA acceleration: `onnxruntime-gpu`
   - For TensorFlow GPU: Follow [TensorFlow GPU setup](https://www.tensorflow.org/install/gpu)

### Running Notebooks

Start Jupyter in the project directory:
```bash
jupyter notebook
```

Then open and run any notebook from the respective model folder.

---

## 📖 Usage Guide

### ArcFace Workflow
1. **Extract Embeddings:** Run `embeddings_and_labels_extraction.ipynb`
2. **Train Classifier:** Run `Training final.ipynb`
3. **Evaluate:** Check classifier performance

### FaceNet Workflow
1. **Prepare Data:** Use MTCNN for face detection
2. **Extract Embeddings:** Run `Embeddings_Using_FaceNet.ipynb`
3. **Classify:** Use `Classifier_Using_FaceNet.ipynb`
4. **Visualize:** Run `GRAD-CAM and Occlusion Heatmap.ipynb`

### FaceNet Live Demo
- **GUI + Webcam:** Run `GUI_+_Webcam_1K.ipynb`
- **Real-time Detection:** Run `LIVE_DETECTION.ipynb`

### ViT Workflow
1. **Extract Embeddings:** Run `Embeddings Extractor.ipynb`
2. **Classify:** Run `Classifier with MTCNN.ipynb`
3. **Analyze:** Run `Classifier with MTCNN with Attention Rollout.ipynb`

### Clustering Analysis
- Understand embedding distributions: Run `Clustering_Embeddings.ipynb` in any model folder

---

## 🔍 Key Features

### Face Detection & Preprocessing
- **MTCNN Integration:** Accurate multi-task cascaded CNNs for face detection
- **Data Augmentation:** Rotation, flipping, and geometric transformations
- **Normalization:** Standardized face sizes (160x160 or 256x256)

### Embedding Extraction
- Extract high-dimensional face representations
- Normalize embeddings for similarity matching
- Support for batch processing

### Classification & Matching
- Cosine similarity-based face matching
- Multi-class classification for celebrity identification
- Confusion matrix generation and analysis

### Model Interpretability
- **GRAD-CAM:** Visualize critical regions for classification decisions
- **Attention Rollout:** Understand ViT attention mechanisms
- **Occlusion Heatmaps:** Identify important facial features

### Real-time Applications
- **Webcam Integration:** Live face detection and recognition
- **GUI Interface:** User-friendly classification interface
- **Performance Optimization:** GPU-accelerated inference

---

## 📊 Sample Results

The project includes visualizations demonstrating:
- **Confusion Matrix:** Model accuracy across celebrity classes
- **GRAD-CAM Heatmaps:** Important regions for predictions
- **Attention Rollout:** Transformer attention patterns
- **Webcam Predictions:** Real-time recognition with confidence scores

See the `Images/` folder for example outputs.

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | TensorFlow, PyTorch, Keras |
| **Face Detection** | MTCNN, InsightFace |
| **Embeddings** | ArcFace, FaceNet, ViT |
| **Data** | Hugging Face Datasets |
| **Visualization** | Matplotlib, OpenCV |
| **GPU Acceleration** | CUDA, cuDNN, ONNX Runtime |

---

## 📝 Notebooks Guide

### Quick Navigation

**Training & Embedding Extraction:**
- `Arcface/Training final.ipynb` - Production ArcFace training
- `FaceNet/Embeddings_Using_FaceNet.ipynb` - FaceNet embedding pipeline
- `ViT/Embeddings Extractor.ipynb` - ViT embedding extraction

**Classification:**
- `FaceNet/Classifier_MTCNN_1K.ipynb` - Fast classifier on 1K celebrities
- `ViT/Classifier with MTCNN.ipynb` - ViT-based classification

**Live Applications:**
- `FaceNet/LIVE_DETECTION.ipynb` - Real-time webcam detection
- `FaceNet/GUI_+_Webcam_1K.ipynb` - GUI interface

**Analysis & Interpretation:**
- `FaceNet/GRAD-CAM and Occlusion Heatmap.ipynb` - Feature importance
- `ViT/Classifier with MTCNN with Attention Rollout.ipynb` - Attention analysis
- `*/Clustering_Embeddings.ipynb` - Embedding space analysis

---

## 💡 Tips for Best Results

1. **GPU Usage:** Enable CUDA for faster embedding extraction
2. **Batch Processing:** Process images in batches for efficiency
3. **Model Selection:**
   - **ArcFace:** Best overall accuracy
   - **FaceNet:** Good balance of speed and accuracy
   - **ViT:** Best for interpretability and modern architectures
4. **Data Quality:** Higher resolution images (256x256+) improve accuracy
5. **Fine-tuning:** Customize models on your own dataset as needed

---

## 🔄 Workflow Pipeline

```
Raw Images
    ↓
Face Detection (MTCNN)
    ↓
Face Alignment & Preprocessing
    ↓
Embedding Extraction (ArcFace/FaceNet/ViT)
    ↓
Classification / Clustering
    ↓
Results & Interpretation
```

---

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size in notebooks
- Use `onnxruntime-gpu` for lighter memory footprint
- Process images sequentially instead of in batches

**Model Loading Errors:**
- Ensure internet connection for model downloads
- Clear pip cache: `pip cache purge`
- Reinstall package: `pip install --upgrade insightface`

**Poor Classification Accuracy:**
- Verify face detection is working properly
- Check image quality and face visibility
- Ensure faces are frontal and well-lit
- Use augmented training data

---

## 🎓 Learning Resources

- **ArcFace Paper:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- **FaceNet Paper:** [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- **Vision Transformers:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **MTCNN:** [Joint Face Detection and Alignment using Multi-task Cascaded CNNs](https://arxiv.org/abs/1604.02878)

---

## 📄 License

This project is open source and available under the [MIT License](https://mit-license.org/).

**Terms:** Feel free to use this project for educational, research, and commercial purposes with proper attribution.

---

## 🤝 Contributing

This project is part of the **AI-Skill Program** at **Helwan National University (HNU)**, dedicated to advancing AI research and education.

Contributions are welcome! Areas for improvement:
- Additional model implementations
- Performance optimizations
- Extended dataset support
- Improved documentation
- Bug fixes and feature requests

---

## 👥 Contributors

This project was developed by the AI-Skill team at Helwan National University:

- [**Ali Abdou**](https://www.linkedin.com/in/aliiabdou/)
- [**Amira Azzam**](https://www.linkedin.com/in/amira-azzam2510/)
- [**Yousef Medhat**](https://www.linkedin.com/in/yousef-medhat-7293232a1/)
- [**Yousef Waheed**](https://www.linkedin.com/in/youssef-waheed-8462061a7/)
- [**Mavi Refaat**](https://www.linkedin.com/in/mavi-refaat-96372b2b5)
- [**Nouran Hassan**](https://www.linkedin.com/in/nouran-hassan112/)

**Institution:** Helwan National University (HNU) - AI-Skill Program

---

## 📧 Contact & Support

For questions or issues, please open a GitHub issue or contact the project maintainers.

**Affiliation:** Helwan National University (HNU) - AI-Skill Program  
**Program Focus:** Advanced AI Research, Deep Learning, and Computer Vision

---

## 🎯 Future Enhancements

- [ ] REST API for model inference
- [ ] Web interface for easy deployment
- [ ] Mobile application support
- [ ] Distributed training on multiple GPUs
- [ ] Additional celebrity datasets
- [ ] Real-time video processing pipeline
- [ ] Model quantization for edge devices


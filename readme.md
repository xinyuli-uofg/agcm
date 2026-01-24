# AGCM FastAPI Web Demo

This project provides a **FastAPI-based web demo** for the paper: Interpretable Concept-based Deep Learning Framework for Multimodal Human Behavior Modeling

Users can upload a facial image, and the system returns:

- Task prediction 
- Interpretable concept prediction 
- Weighted concept attention maps
- AU heatmaps

---

## 🔗 Project Website

For detailed methodology, experiments, visualizations, and supplementary material, please visit the **project page**:

https://xinyuli-uofg.github.io/agcm/

---

## 1. Environment Setup

### 1.1 Prerequisites
- Linux 
- Conda
- Python ≥ 3.9
- NVIDIA GPU + CUDA for faster inference

### 1.2 Create Conda Environment

```bash
conda env create -f environment.yml
conda activate agcm
```

## 2. Project Structure
```
experiments/
├── xy_web_inferece_2026.py      # FastAPI entry file
├── agcem_test_affectnet_ROI_web.py
├── models/                         # models 
├── xtools/                      # data loaders & visualisation
├── train/                      # training utilities
└── README.md
```
## 3. Running the FastAPI Server

### 3.1 Navigate to the correct directory

**Important:** First change to the directory where the FastAPI entry file is located.

```bash
cd /{YOUR_LOCATION}/agcm_web
```

### 3.2 Start the FastAPI server
```
python -m uvicorn xy_web_inferece_2026:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

### 3.3 Access the Web Interface
Open a browser and visit:
```http://localhost:8000```

You can now upload an image and run inference.

## 4. Exposing the Server to the Internet (Optional)
If you want to make the web demo accessible externally (e.g., for collaborators):
### 4.1 Install ngrok
Download ngrok from:
```
https://ngrok.com/
```
### 4.2 Start ngrok tunnel
In a new terminal:
```
ngrok http 8000
```
ngrok will provide a public URL such as:
```
https://xxxx-xx-xx-xx.ngrok-free.app
```
Share this link with others to access the demo remotely.
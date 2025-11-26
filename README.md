# 🧬 Advancing Protein–Protein Interaction Analysis with Graph Convolutional Networks

This repository contains a complete end-to-end pipeline for **predicting protein–protein interaction (PPI) strength** using **Graph Convolutional Networks (GCNs)**.  

The workflow includes:

- Extracting protein sequences via **Ensembl REST API**
- Generating **ProtT5 embeddings**
- Building a STRING v12.0–based PPI graph
- Training **GCN models** for 3-class interaction prediction  
- Handling data imbalance using **Weighted Cross Entropy** and **Focal Loss**
- Ensembling models for improved robustness  
- Achieving **state-of-the-art performance (Micro-F1: 97.72%)**  

Note: All experiments were executed on **Google Colab using T4 GPU** for reproducibility.

---

## 📂 Repository Structure

📁 Codes  
   ├─ Final_extracting_protein_embeddins.ipynb  
   └─ PPI_GCN.ipynb

📁 Datasets  
   ├─ link  (Public Google Drive link to dataset / embeddings)  
   └─ LICENSE


---

#  Features

-  Fully automated pipeline: **Protein ID → Sequence → Embedding → Graph → GCN prediction**
-  Uses **ProtT5** transformer model for 1024-dim embeddings
-  Works on **STRING v12.0** (675,122 PPIs)
-  Predicts **Weak / Moderate / Strong** interaction strength
-  Handles heavy class imbalance
-  Uses **GCN + MLP edge classifier**
-  Two-model ensemble for stable predictions
-  Colab-ready notebooks

---

# Installation & Setup

### Clone Repository
```bash
git clone https://github.com/krishnasri07/ppi.git
cd ppi
```
Install Dependencies
Works on Google Colab (Recommended, T4 GPU) or local machine.


```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers accelerate
pip install tqdm scikit-learn pandas numpy
```
---

# Download Dataset
The dataset (STRING files, embeddings, sequence dictionary) is available at:
### [Download Dataset](https://drive.google.com/file/d/1ADa40v5O0n5kBVg7C78U6kBFlp1h4Xuq/view?usp=sharing)


--- 
# Visual Workflow

<img width="1763" height="749" alt="image" src="https://github.com/user-attachments/assets/d91f4975-c54a-493f-b4d6-0085c98f773f" />
<p align="center">Figure1: Graph generation from STRING data</p>

<br>
<br>
<br>
<img width="1524" height="375" alt="image" src="https://github.com/user-attachments/assets/b1d7140b-4134-4bba-901b-bb7497098793" />

<p align="center">Figure2: Proposed GCN model pipeline</p>

--- 

# Running the Pipeline
### 1. Extract Protein Embeddings

Upload notebook in google colab,
Codes/Final_extracting_protein_embeddins.ipynb

This notebook performs:
- Cleans Ensembl IDs
- Fetches sequences via Ensembl REST API
- Generates ProtT5 embeddings (1024-dim)
- Saves embeddings for fast reuse

### 2. Train the GCN Model

Upload notebook in google colab,
Codes/PPI_GCN.ipynb

This notebook includes:
- Graph construction from STRING v12.0
- Undirected graph cleaning (remove duplicates & self-loops)
- Balanced sampling
- GCN training
- Ensemble model inference
- Complete evaluation metrics and plots

### 3. Model Architecture
- Node Encoder (GCN)
- 4× GCNConv layers
- Hidden size: 256
- Each layer: GCN → ReLU → Dropout
- Edge Classifier (MLP)

Input:
[node embedding i || node embedding j || edge attribute]

### 4. Architecture:

- 513 → 256 → ReLU → Dropout → 3 logits
- Loss Functions
- Model 1: Weighted Cross Entropy
- Model 2: Focal Loss
Ensemble Output: Average of both models

---

# Results
#### Per-Class Classification Metrics,

| **Metric**   | **Weak** | **Moderate** | **Strong** |
|--------------|----------|--------------|------------|
| Precision    | 100.00   | 88.95        | 97.44      |
| Recall       | 97.14    | 98.19        | 100.00     |
| F1-score     | 98.55    | 93.34        | 98.71      |


Overall Accuracy: 97.65%

#### Macro / Micro / Weighted Averages

| **Metric**      | **Score** |
|------------------|-----------|
| Macro F1         | 96.87     |
| Micro F1         | 97.72     |
| Weighted F1      | 97.70     |


#### ROC-AUC Scores

| **Class** | **AUC** |
|-----------|---------|
| Weak      | 1.00    |
| Moderate  | 0.97    |
| Strong    | 1.00    |


#### PR-AUC Scores

| **Class** | **AP**   |
|-----------|----------|
| Weak      | 0.993    |
| Moderate  | 0.989    |
| Strong    | 1.000    |


The Strong class achieves perfect ROC-AUC and PR-AUC, demonstrating excellent learning of minority samples thanks to our use of Focal Loss + Ensemble.

#### Comparison with State-of-the-Art

| **Method**                | **Dataset**    | **PPIs**  | **Micro-F1** |
|---------------------------|----------------|-----------|--------------|
| DL-PPI (Wu et al., 2023)  | STRING v10.5   | 593,397   | 94.85        |
| **Our GCN (This Study)**  | STRING v12.0   | 675,122   | **97.72**    |

---

# Future Work
- Extend to more organisms & datasets
- Replace GCN with newer models: GraphSAGE, GAT, GIN, HGT
- Use 3D structure-based embeddings (ESMFold)
- Apply contrastive learning for pretraining

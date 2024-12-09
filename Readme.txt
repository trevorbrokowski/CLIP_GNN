##############################
# CLIP_GNN: Training Models #
##############################

This project trains and evaluates models aligning visual and textual features using 
CLIP and GNNs. The Visual Genome dataset is required for this project.

-------------------------------
1. PREPARE THE DATASET
-------------------------------

**Download Visual Genome Dataset**  
- Go to: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- Download all the files in Version 1.2

**Merge Image Directories**  
- Visual Genome images are stored in two directories: `VG_100K` and `VG_100K_2`.  
- Combine them into a single folder named `all_images` using the following commands:

```bash
mkdir -p VisualGenome/images/all_images
cp VG_100K/* VisualGenome/images/all_images/
cp VG_100K_2/* VisualGenome/images/all_images/

-------------------------------
2. Download the Requirements 
-------------------------------

To install all the required libraries, run the following command:

```bash
pip install -r Requirements.txt

-------------------------------
3. RUNNING THE MODELS AND RESULTS
-------------------------------

Each of the following `.py` files — `Train_Baseline.py`, `Train_CLIP_BB_GNN.py`, and `Train_CLIP_GNN.py` — contains code for:  
1. **Creating their respective datasets and dataloaders**  
2. **Training the models**  
3. **Evaluating the model performances**

### **Baseline Model (CLIP)**
Run the following command to train the baseline model:

```bash
python Train_Baseline.py

This script aligns visual and textual embeddings using the CLIP model.

### **CLIP_GNN**
Run the following command to train the baseline model:

```bash
python Train_CLIP_GNN.py

This script aligns visual and graph embeddings using the CLIP_GNN model 

### **CLIP_BB_GNN**
Run the following command to train the baseline model:

```bash
python Train_CLIP_BB_GNN.py

This script aligns visual and graph embeddings with spatial information using the CLIP_BB_GNN model 



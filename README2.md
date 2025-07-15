
# YOLOX-CAViT: Class-Aware Vision Transformer Enhanced YOLOX

YOLOX-CAViT is a modified version of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) that integrates a **Class-Aware Vision Transformer (CAViT)** between the backbone and neck. This design builds upon [KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT), improving detection accuracy by **emphasizing class-relevant features through class-aware attention**.

> 🧠 CAViT introduces class-aware attention, making it particularly useful for fine-grained or imbalanced object detection scenarios.

---

## 🚀 How to Run YOLOX-CAViT

### 🔧 Step 1: Clone the Base Repository

Use the original KD-YOLOX-ViT repository:

```bash
git clone https://github.com/remaro-network/KD-YOLOX-ViT.git
cd KD-YOLOX-ViT
```

---

### 🔁 Step 2: Replace Model Architecture Files with CAViT Versions

To activate the class-aware transformer logic:

1. Replace the following files from this repository:

   * `yolox/models/network_blocks.py` → contains the **Class-Aware Transformer Layer**
   * `yolox/models/darknet.py` → integrates the transformer into the backbone pipeline
   * `yolox/models/pafpn.py` → connects ViT to the FPN appropriately

These replacements **introduce class-aware attention** and retain compatibility with the rest of the YOLOX pipeline.

---

### ⚙️ Step 3: Create a Custom Exp File

1. Navigate to: `/exps/example/custom/`
2. Copy and rename the default `yolox_l.py` or another variant you prefer.
3. Modify:

   * `num_classes`: Total number of classes in your dataset
   * `data_dir`, `train_ann`, `val_ann`: Your dataset paths
   * `max_epoch`, `input_size`, `test_size`: Training hyperparameters
   * `exp_name`: For uniquely identifying your experiment

> ✅ The CAViT transformer is internally initialized from this file.

---

### 🏷️ Step 4: Update Dataset Class Names

Edit `/yolox/data/datasets/coco_classes.py` to include your own list of class names.

---

### 🔍 Optional: Replace Evaluator for Class-wise AP @ IoU 50

To analyze **per-class AP at IoU=0.50**, you may replace `/yolox/evaluators/coco_evaluator.py` with a custom evaluator supporting class-wise breakdowns.

---

## 🧠 CAViT Architectural Highlights

**Class-Aware Vision Transformer (CAViT)** modifies the transformer block with:

* 🔁 **Class-Specific Attention Modulation**
  Learns `class_weights` and uses a projection layer (`class_proj`) to compute spatial class attention.

* 🧮 **Weighted Class Attention**
  Focuses attention on regions likely belonging to specific classes.

* 🧩 **Class-Aware Bias Modulation**
  Adjusts attention output using class-specific cues to improve discrimination.

* 🧼 **Pre-Normalization**
  Applies layer normalization before MHSA and FFN for stable learning.

* 🧪 **Dropout Regularization**
  Helps reduce overfitting—especially beneficial in class-imbalanced datasets.

---

## 🧪 Training and Evaluation

### ▶️ Training

```bash
export PYTHONPATH="${PYTHONPATH}:/mnt/Users/tarun_kumar_/KD-YOLOX-ViT" && \
python3 tools/train.py -f exps/example/custom/yolox_l.py -d 1 -b 1 --fp16 -o \
-c /mnt/Users/tarun_kumar_/KD-YOLOX-ViT/yolox_l.pth
```

### 📈 Evaluation

```bash
export PYTHONPATH="${PYTHONPATH}:/mnt/Users/tarun_kumar_/KD-YOLOX-ViT" && \
python3 tools/eval.py -n yolox-l -c /mnt/Users/tarun_kumar_/KD-YOLOX-ViT/YOLOX_outputs/yolox_l/epoch_47_ckpt.pth \
-b 1 -d 1 --conf 0.001 -f exps/example/custom/yolox_l.py
```

---

## 📌 Notes

* No knowledge distillation is applied unless added manually.
* Class weights can optionally be initialized using your dataset’s class distribution.
* The CAViT module replaces the original ViT used in YOLOX-ViT.
* All training, evaluation, and export steps remain the same as standard YOLOX.

---

## 📄 Citation

If you use this architecture or code, please consider citing the following:

```
Aubard, M., Antal, L., Madureira, A., & Ábrahám, E. (2024). Knowledge Distillation in YOLOX-ViT for Side-Scan Sonar Object Detection. REMARO Workshop, ETAPS 2024.
```

---

## 🔗 Related Repositories

* KD-YOLOX-ViT: [https://github.com/remaro-network/KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT)
* YOLOX: [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)


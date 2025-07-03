

# YOLOX-ViT: Vision Transformer Enhanced YOLOX

This repository contains the modified YOLOX-ViT architecture based on the original [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), with a Vision Transformer (ViT) layer added between the backbone and neck. It is adapted from [KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT), designed for lightweight object detection with improved global feature extraction.

> ✅ Running YOLOX-ViT is very similar to standard YOLOX, with just a few minor modifications.

---

## 🚀 How to Run YOLOX-ViT on DGX

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/remaro-network/KD-YOLOX-ViT.git
cd KD-YOLOX-ViT
```

---

### 🛠️ Step 2: Apply the Required Modifications

#### 1. Create a Custom Exp File

* Path: `/exps/example/custom`
* If you're using the `yolox-l` model, copy and modify the provided `yolox_l.py` file.
* Update the following in the file:

  * Dataset paths
  * Number of epochs
  * Number of classes
  * Input/output settings

#### 2. Update Dataset Class Names

* Path: `/yolox/data/datasets/coco_classes.py`
* Replace the class list with your own dataset’s class names.

#### 3. (Optional) Replace Evaluator File for Class-wise AP @ IoU 50

* Path: `/yolox/evaluators/coco_evaluator.py`
* Replace it with a custom version from your own YOLOX repo if you want detailed per-class AP at IoU 0.50.

#### 4. Modifying the Architecture (if needed)

* Path: `/yolox/models`
* You can change or extend the model architecture here (e.g., modify ViT integration), similar to how it is done in the original YOLOX.

---

### ✅ All Other Steps Are Same as YOLOX

You can follow the standard YOLOX procedures for:

* Dataset preparation
* Training
* Evaluation
* Exporting models

Refer to the original [YOLOX documentation](https://github.com/Megvii-BaseDetection/YOLOX) for complete instructions.

---

## 🧪 Example Training and Evaluation Commands

### ▶️ Training Script

```bash
export PYTHONPATH="${PYTHONPATH}:/mnt/Users/tarun_kumar_/KD-YOLOX-ViT" && \
python3 tools/train.py -f exps/example/custom/yolox_l.py -d 1 -b 1 --fp16 -o \
-c /mnt/Users/tarun_kumar_/KD-YOLOX-ViT/yolox_l.pth
```

### 📈 Evaluation Script

```bash
export PYTHONPATH="${PYTHONPATH}:/mnt/Users/tarun_kumar_/KD-YOLOX-ViT" && \
python3 tools/eval.py -n yolox-l -c /mnt/Users/tarun_kumar_/KD-YOLOX-ViT/YOLOX_outputs/yolox_l/epoch_47_ckpt.pth \
-b 1 -d 1 --conf 0.001 -f exps/example/custom/yolox_l.py
```

---

## 📌 Notes

* The ViT module is inserted between the SPP bottleneck and the neck in the YOLOX architecture.
* No knowledge distillation is applied unless added manually.
* The repo supports both offline and online training approaches as described in the paper.

---

## 📄 Citation

If you use this codebase, please consider citing the following:

```
Aubard, M., Antal, L., Madureira, A., & Ábrahám, E. (2024). Knowledge Distillation in YOLOX-ViT for Side-Scan Sonar Object Detection. REMARO Workshop, ETAPS 2024.
```

---

## 🔗 Related Repositories

* Original KD-YOLOX-ViT: [https://github.com/remaro-network/KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT)
* YOLOX: [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)


```
```

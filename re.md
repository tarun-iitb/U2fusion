
# Extended TIDE Error Analysis for Object Detection

This script provides an extended pipeline for evaluating object detection models using the [TIDE](https://github.com/dbolya/tide) framework. Beyond the standard mAP breakdown, it extracts and logs **sample-wise errors** per error type (e.g., Localization, Classification, Background), and **visualizes top failure cases** with bounding boxes.

---

## 🔍 Key Features

- Uses **COCO-format** JSON annotations for ground truth and predictions.
- Supports detailed analysis of individual error types:
  - Background (Bkg)
  - Classification (Cls)
  - Localization (Loc)
  - Duplicate Detections (Dupe)
  - Combined Loc+Cls (Both)
  - Missed Detections (Miss)
- Saves per-error-type CSVs containing:
  - Image ID, predicted class, GT class, confidence score, bboxes, etc.
- Visualizes top 5 Background errors with predicted bounding boxes.
- Auto-handles prediction/GT matching with confidence filtering (`thres=0.01`).

---

## 📂 Project Structure

```

project/
│
├── instances\_test2017.json              # COCO-format GT annotations
├── 9cat\_50e\_predictions\_new\.json        # Detection predictions (COCO-format)
├── test2017/                            # Image directory for visualization
├── bkg\_errors.csv, cls\_errors.csv ...  # Output CSVs per error type
├── visualize\_bkg\_errors.png            # Visualized output (inline with matplotlib)
└── extended\_tide\_analysis.py           # The main script

````

---

## ⚙️ Requirements

- Python 3.x
- TIDE
- OpenCV
- matplotlib

Install dependencies:

```bash
pip install tidecv opencv-python matplotlib
````

---

## 🚀 How to Run

1. Place your annotation files and prediction results in the working directory.
2. Set correct paths to:

   * `instances_test2017.json` (ground truth)
   * `9cat_50e_predictions_new.json` (predictions)
   * Image folder (e.g., `test2017/`)
3. Run the script:

```bash
python extended_tide_analysis.py
```

---

## 📊 Outputs

* **Terminal Summary:** Overall metrics and top 10 errors of each type.
* **CSV Files:** Saved for each error type (e.g., `bkg_errors.csv`)
* **Image Visualizations:** Top-5 Bkg errors displayed using matplotlib.

---

## 📝 Notes

* You can easily modify the threshold `thres=0.01` to filter detections by confidence.
* The fallback mapping for classes is provided in case category names are missing in COCO JSON.
* Error types are defined based on TIDE's default taxonomy.

---

## 📎 Credits

* [TIDE](https://github.com/dbolya/tide) by Daniel Bolya
* Extension and visualization logic by \[Your Name or Team]

---

## 📷 Example Output (Visualized Error)

![Example Background Error](visualize_bkg_errors.png)


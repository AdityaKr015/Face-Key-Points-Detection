# **Face-Key-Points-Detection**

**Frameworks & Libraries:-**
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF)

**Platforms:-**
![Roboflow](https://img.shields.io/badge/Roboflow-6100ee?logo=roboflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&logoColor=white)

**License:-** [![AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue)](LICENSE)

A College Deep Learning Project that builds Face Keypoint Detection with YOLOv8-pose.

---

## 🚀 Live Demo

**Run the model directly in your browser — no server, no install:**

**[https://face-key-points-detection.vercel.app/](https://face-key-points-detection.vercel.app/)**

100% client-side inference (WebGPU with automatic WASM fallback) via [LiteRT.js](https://developers.google.com/edge/litert/web) — the image never leaves the browser.

---

## 🌐 Browser Deployment

The trained model runs in the browser through the LiteRT stack:

- **Convert:** `best.pt → best_w8a32.tflite` — LiteRT export with dynamic INT8 quantization (43.9 MB → 11.4 MB), verified <2px keypoint error vs the original on WFLW test images
- **Host:** model served from [HuggingFace Hub — AdiKr25/litert-models](https://huggingface.co/AdiKr25/litert-models)
- **Run:** [`face-keypoints-web/`](face-keypoints-web/) — React + Vite + [`@ultralytics/yolo`](https://www.npmjs.com/package/@ultralytics/yolo), deployed on Vercel

**Run the demo locally:**

```bash
cd face-keypoints-web
npm install
echo "VITE_MODEL_URL=https://huggingface.co/AdiKr25/litert-models/resolve/main/face-keypoints.tflite" > .env.local
npm run dev
```

---

## 📓 Training Notebook

The full model training pipeline can be found here:

[Open Notebook](face-wflw.ipynb)

---

## **Project Overview**

In this project, I built a real-time face keypoint detection system using transfer learning (pretrained YOLOv8-pose model), trained on the WFLW (Wider Facial Landmarks in-the-wild) dataset from Roboflow and fine-tuned on the Kaggle platform.

**The main goal of this project is to:-**

- Detect faces and localize 5 facial keypoints (eyes, nose and mouth corners) for people who need facial landmark analysis.

## **Features**

- Real-time face detection with keypoint localization.
- Bounding Box labeled faces with confidence scores.
- 5 facial keypoints per face with visibility flags.

## **Implementation**

Project implementation started with dataset preparation and ended with real-time detection results. The entire workflow uses Kaggle for training and Roboflow Universe for data preparation.

### 1. **Dataset Integration**

- The dataset was sourced from Roboflow, based on **WFLW** (Wider Facial Landmarks in-the-wild), consisting of **1 class (face)** with **5 keypoints** per face.
- Three Roboflow subsets (`wflw_pose0_29`, `wflw_pose30_49`, `wflw_pose50_61`) were merged into a single dataset with a corrected `data.yaml`.
- The dataset was exported in **YOLO-pose compatible format** (with training, validation, and test splits).
- The dataset was zipped and uploaded to Kaggle to load into the Jupyter Notebook for model training.

### 2. **Environment Setup**

- Installed the **Ultralytics Library** on Kaggle to use YOLO.
- Set up **Dual GPU (2× T4)** for training runtime.
- Model was evaluated on the **val** data split.
- Model ran on the **test** data split to produce bounding boxes with keypoints, labels and confidence scores.
- Zipped the output folder for download to local machine.

### 3. **Model Training**

Fine-tuned the pre-trained **YOLOv8s-pose** model from Ultralytics using transfer learning, optimized for real-time performance, speed, and accuracy on dual NVIDIA T4 GPUs.

**Training parameters:-**

| Parameter | Value | Purpose |
|------|--------|--------|
|Epochs / Batch |100 / 64 |Stable convergence (early-stopped at epoch 31, patience 10), optimized for dual T4 GPU memory |
|Img Size |640 × 640 |Balanced trade-off between detail and inference speed |
|Optimizer |AdamW |Momentum 0.937, weight decay 0.0005 for regularization |
|Learning Rate |0.01 → 0.0001 |Cosine decay (cos_lr=True) with a 3-epoch warmup |
|Loss Weights |box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0 |Tightens bounding boxes and facial keypoint alignment |
|Mixed Precision |Off (amp=False) |Dataset required 32-bit precision to avoid training crashes |

**Data Augmentation Strategy:-**

To ensure robustness against real-world lighting and camera angles, the following augmentations were applied:

  - **Color (HSV):-** Hue (±0.015), Saturation (±0.7), and Brightness (±0.4) variations.
  - **Geometry:-** Translation (±10%), scale/zoom (±50%), and horizontal flip (50% probability).
  - **Context:-** Mosaic (1.0) combining 4 training images to improve small-face detection, plus random erasing (0.4) for occlusion robustness.

### 4. Model Validation

The model was evaluated on a dedicated validation split. Key metrics:

- **Precision:-** Percentage of predicted detections that were correct.
- **Recall:-** Percentage of actual faces successfully detected.
- **mAP@50:-** Mean Average Precision calculated at an IoU (Intersection over Union) threshold of 0.50 (measures general detection accuracy).
- **mAP@50-95:-** Mean Average Precision calculated across a range of IoU thresholds from 0.50 to 0.95 (measures localization precision and boundary tightness).

### 5. **Inference and Output**

- The trained YOLOv8s-pose model was used for inference on the **test** split (1400 images) at `conf=0.25`.
- Each detected face was marked with a **bounding box**, **5 facial keypoints**, **label** and **confidence score**.
- Results showed the model successfully detected multiple faces and keypoints simultaneously in crowded scenes.

---

# **Confusion Matrix & Results data**

### Confusion Matrix (Normalized)

<img src="pose/train/confusion_matrix_normalized.png" />

> The **diagonal dark blue cells** represent correct predictions per class.

> Strong diagonal dominance for the **face** class confirms the model learned clear boundaries between faces and background.

### Training & Validation Curves

<img src="pose/train/results.png" />

> All **loss curves decrease smoothly** over 31 epochs with no overfitting.

> **Precision, Recall, mAP@50 and mAP@50-95** all rise steadily.

> Stable training and strong generalization on unseen data.

### Why These Results Are Strong

| Factor | Evidence |
|--------|---------|
| **High Precision (Pose) ~94%** | Very few false detections, model doesn't hallucinate faces |
| **High Recall (Pose) ~90%** | Catches almost all real faces in frame |
| **Diagonal Confusion Matrix** | Clean separation between the face class and background |
| **Smooth Loss Curves** | Stable training, no exploding/vanishing gradients |
| **mAP@50 (Pose) ~96%** | Reliable face and keypoint detection across varied real-world conditions |
| **mAP@50-95 (Pose) ~63%** | Good bounding box tightness, not just rough localization |
| **Val Loss ≈ Train Loss** | Model generalizes, not memorizing training data |

---

## **WORKFLOW:-**

```mermaid
flowchart TD
    A["🗄️ Dataset Source
    • Roboflow Universe
    • WFLW (5 Keypoints)
    • 6.4K+ Labeled Images"]

    B["📦 Dataset Preparation
    • Merge 3 Roboflow Subsets
    • YOLO-Pose Format Export
    • Train / Val / Test Split"]

    C["☁️ Kaggle Platform
    • Upload Dataset
    • Setup Dual T4 GPU
    • Install Ultralytics"]

    D["🧠 Pretrained Model
    • YOLOv8s-pose
    • Ultralytics
    • Transfer Learning"]

    E["⚙️ Model Training
    • Epochs: 100 | Batch: 64
    • Img Size: 640×640
    • Optimizer: AdamW | LR: 0.01"]

    F["🎨 Data Augmentation
    • HSV Color Jitter
    • Translation / Scale
    • Flip / Mosaic / Erasing"]

    G["📊 Model Validation
    • Precision: 94.3%
    • Recall: 90.1%
    • mAP@50: 96.5%"]

    H{"✅ Accuracy
    Acceptable?"}

    I["💾 Save Best Weights
    • best.pt
    • best.onnx
    • last.pt
    • Download via ZIP"]

    J["🔍 Inference & Testing
    • Test Dataset
    • Bounding Boxes
    • Keypoints + Confidence"]

    K["🎯 Real-Time Detection
    • Webcam / Video Feed
    • Live Face Labels
    • Keypoints + Confidence"]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    G --> H
    H -- No --> E
    H -- Yes --> I
    I --> J
    J --> K

    style A fill:#6100ee,color:#fff,stroke:#4a00b4,stroke-width:2px
    style B fill:#4A90D9,color:#fff,stroke:#2c6fad,stroke-width:2px
    style C fill:#20BEFF,color:#000,stroke:#0090cc,stroke-width:2px
    style D fill:#00CED1,color:#000,stroke:#009aa0,stroke-width:2px
    style E fill:#FF6B35,color:#fff,stroke:#cc4a10,stroke-width:2px
    style F fill:#E8A838,color:#000,stroke:#b87d10,stroke-width:2px
    style G fill:#2ECC71,color:#fff,stroke:#1a9e50,stroke-width:2px
    style H fill:#E74C3C,color:#fff,stroke:#b52a1c,stroke-width:2px
    style I fill:#8E44AD,color:#fff,stroke:#6a2585,stroke-width:2px
    style J fill:#3498DB,color:#fff,stroke:#1a6fad,stroke-width:2px
    style K fill:#27AE60,color:#fff,stroke:#1a7a40,stroke-width:2px
```

---

## **Project Structure**
```
├── face-wflw.ipynb                           # Main Jupyter Notebook
├── face-keypoints-web/                       # Browser demo app (React + LiteRT.js)
├── pose/
│   ├── train/                                # Training outputs & results
│   │   └── weights/                          # Saved model weights (best.pt, last.pt, best.onnx)
│   ├── val/                                  # Validation results & predictions
│   └── predict/                              # Testing images & outputs
├── WFLW/                                     # Dataset (train / valid / test) — local only, not in repo
├── outputs.zip                               # Zipped run outputs — local only, not in repo
├── LICENSE                                   # Project license
└── README.md                                 # Project documentation
```
---

## ⚙️ **Tech Stack**

- `Language`:- Python 3.10+
- `Frameworks/Libraries`:- Ultralytics (YOLO Framework)
- `Training Platform`:- Kaggle (Dual T4 GPU)
- `Dataset Platform`:- Roboflow Universe
- `Frontend`:- React + Vite + TypeScript, LiteRT.js (`@ultralytics/yolo`)
- `Deployment`:- HuggingFace Hub (model), Vercel (web app)

---

## Dataset

### Dataset Source:-

- [Roboflow Universe — WFLW Pose (0-29)](https://universe.roboflow.com/fullfacedetect/wflw_pose0_29)
- [Roboflow Universe — WFLW Pose (30-49)](https://universe.roboflow.com/fullfacedetect/wflw_pose30_49)
- [Roboflow Universe — WFLW Pose (50-61)](https://universe.roboflow.com/fullfacedetect/wflw_pose50_61)

> I merged all these 3 datasets into one single folder with train, test and valid splits and a corrected `data.yaml`, then zipped and uploaded it to Kaggle.

- 6,400+ labeled images (4,416 train / 612 val / 1,400 test)
- 1 class (face) with 5 facial keypoints per face
- Exported in YOLO-pose format with Train / Val / Test splits

---

## 📈 **Outputs**

<table border="0">
  <tr>
    <td align="center">
      <b>Parade</b><br><br>
      <img src="pose/predict/0_Parade_Parade_0_3_jpg.rf.d2a4a4be61eb5e8f3ac7e1630e2a28b1.jpg">
      <br>
    </td>
    <td align="center">
      <b>Handshaking</b><br><br>
      <img src="pose/predict/1_Handshaking_Handshaking_1_341_jpg.rf.08632db53e49df52f46400e7f88138dc.jpg">
      <br>
    </td>
    <td align="center">
      <b>People Marching</b><br><br>
      <img src="pose/predict/10_People_Marching_People_Marching_2_747_jpg.rf.c8b91be68cbba682c1c8c10bc8586b90.jpg">
      <br>
    </td>
    <td align="center">
      <b>Meeting</b><br><br>
      <img src="pose/predict/11_Meeting_Meeting_11_Meeting_Meeting_11_1016_jpg.rf.ca959428af8d3f8984e183601e0da17c.jpg">
      <br>
    </td>
  </tr>

  <tr>
    <td align="center">
      <b>Award Ceremony</b><br><br>
      <img src="pose/predict/16_Award_Ceremony_Awards_Ceremony_16_595_jpg.rf.16ff507e6c4a90848aed282e1365630c.jpg">
      <br>
    </td>
    <td align="center">
      <b>Interview</b><br><br>
      <img src="pose/predict/13_Interview_Interview_Sequences_13_59_jpg.rf.225d58fd8d321a8aa8d5268dbf508084.jpg">
      <br>
    </td>
    <td align="center">
      <b>Parade Marchingband</b><br><br>
      <img src="pose/predict/0_Parade_marchingband_1_1015_jpg.rf.cb330da621dbb4c241a797c2f3162727.jpg">
      <br>
    </td>
    <td align="center">
      <b>Group</b><br><br>
      <img src="pose/predict/12_Group_Group_12_Group_Group_12_10_jpg.rf.aa18cae149d849393a66ae1292434d1e.jpg">
      <br>
    </td>
  </tr>

  <tr>
    <td align="center">
      <b>Group Large</b><br><br>
      <img src="pose/predict/12_Group_Large_Group_12_Group_Large_Group_12_1013_jpg.rf.d50234c381baab4ddec8909e02ce7aea.jpg">
      <br>
    </td>
    <td align="center">
      <b>Group Team Organized</b><br><br>
      <img src="pose/predict/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_1_jpg.rf.05fd78f027f8a760bb5a3518f24d69a3.jpg">
      <br>
    </td>
    <td align="center">
      <b>Interview On Location</b><br><br>
      <img src="pose/predict/13_Interview_Interview_On_Location_13_100_jpg.rf.d1cb2739cba227cf32dee0ae1ce69103.jpg">
      <br>
    </td>
    <td align="center">
      <b>Traffic</b><br><br>
      <img src="pose/predict/14_Traffic_Traffic_14_1004_jpg.rf.bf38ab153d1607ff73bdf911fe1ef3ec.jpg">
      <br>
    </td>
  </tr>

  <tr>
    <td align="center">
      <b>Stock Market</b><br><br>
      <img src="pose/predict/15_Stock_Market_Stock_Market_15_1000_jpg.rf.bede5ae5fd19a7380f4c17fe873bcea0.jpg">
      <br>
    </td>
    <td align="center">
      <b>Ceremony</b><br><br>
      <img src="pose/predict/17_Ceremony_Ceremony_17_100_jpg.rf.a8cf69dca80a5869e893b53541fc4cbc.jpg">
      <br>
    </td>
    <td align="center">
      <b>Concerts</b><br><br>
      <img src="pose/predict/18_Concerts_Concerts_18_1006_jpg.rf.01855e0929d2b7a59f424573b32ae542.jpg">
      <br>
    </td>
    <td align="center">
      <b>Couple</b><br><br>
      <img src="pose/predict/19_Couple_Couple_19_231_jpg.rf.03ef445b91eac22717b0297d922f8866.jpg">
      <br>
    </td>
  </tr>

  <tr>
    <td align="center">
      <b>Family Group</b><br><br>
      <img src="pose/predict/20_Family_Group_Family_Group_20_1001_jpg.rf.415f0c7a915894209ee7410380f3b67c.jpg">
      <br>
    </td>
    <td align="center">
      <b>Festival</b><br><br>
      <img src="pose/predict/21_Festival_Festival_21_300_jpg.rf.5c87d106e6dea5c8a76d98d8483fe9b3.jpg">
      <br>
    </td>
    <td align="center">
      <b>Picnic</b><br><br>
      <img src="pose/predict/22_Picnic_Picnic_22_304_jpg.rf.192b6d6db111f256bb6db46939d2cfce.jpg">
      <br>
    </td>
    <td align="center">
      <b>Shoppers</b><br><br>
      <img src="pose/predict/23_Shoppers_Shoppers_23_237_jpg.rf.9783a551f164f41af991afd672c9b436.jpg">
      <br>
    </td>
  </tr>
</table>

---

## Results:

### Evaluation Metrics:- (Final Epoch)

| Metrics | Score |
|-------|--------|
|`Precision (Pose)`| ~ 94.3%|
|`Recall (Pose)`| ~ 90.1%|
|`mAP@50 (Pose)`| ~ 96.5%|
|`mAP@50–95 (Pose)`| ~ 62.6%|
|`mAP@50 (Box)`| ~ 97.4%|

---

## **License**

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the [LICENSE](LICENSE) file for the full license text.

This project uses the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. The Ultralytics software and related components are subject to their respective licensing terms.
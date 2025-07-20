# Dental X-ray Images Analysis Using Deep Learning (Segmentation Task) 🦷

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![GitHub stars](https://img.shields.io/github/stars/USERNAME/REPO.svg)](https://github.com/USERNAME/REPO/stargazers) [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=fff)](https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task)



![X-ray Example]([https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/1-s2.0-S0010482522010046-ga1.jpg](https://github.com/mohamedali020/AI-based-Dental-X-ray-Analysis-using-Custom-UNet-Segmentation-with-VGG19-Backbone/blob/main/Architicture%20model.png))



## 📑 Table of Contents
- [Abstract](#abstract)  
- [Challenges](#challenges)  
- [Project Overview](#project-overview)  
- [Dataset Details](#dataset-details)  
- [Methodology](#methodology)  
- [Model Architecture](#model-architecture)  
- [Results](#results)  
- [Final Output](#final-output)  
- [Future Work](#future-work)  
- [Installation & Usage](#installation--usage)  

---

## 🧾 Abstract

**Radiographic examinations** have a major role in assisting **dentists to analyse the early teeth complications diagnosis** such as infections, bone defects, and tumors. Unfortunately, relying only on the dentist’s opinion after 
a radiographic scan may lead to false-positive results, where it is proven that **3% of X-ray scan diagnoses are false resulting in psychological stress for the patients.** Researchers and doctors began using computer vision techniques to aid in diagnosing patients in the dentistry field because of the growing number of medical X-Ray images. In computer vision, various tasks are applied to digital images such as object detection, object tracking, and features recognition. **The most important computer vision technique is image segmentation, which is a **deep learning technology used in the medical sector to detect key features in medical radiographs**. Image segmentation works by dividing the **pixels of an image into numerous segments**, where each pixel is usually classified to belong to a specific class category in the image, this helps simplify the representation of the input image making the desired objects** 
easier to analyze by extracting the boundaries between objects to develop significant regions. There are numerous image segmentation algorithms with the goal to detect and extract the 
desired object from the image background. The **two main types of image segmentation are semantic segmentation and instance segmentation** where both techniques concatenate one another. **Semantic segmentation associates each pixel of the digital image with a class label** such as teeth in general, however, instance segmentation handles numerous objects of the same class independently.

---

## ⚠️ Challenges

- X-rays often have **noise**, requiring denoising, resizing, normalization, and scaling.
- **High variability** in tooth shape, size, and alignment across patients can lead to false positives.
- **Preprocessing** demands heavy computation and can cause runtime issues depending on available hardware.
- **Class imbalance**: background pixels heavily outnumber target pixels, requiring careful training strategy.

---

## 📋 Project Overview

- I built an **AI-powered system** that helps dentists analyze panoramic X-rays.
- They upload an X-ray → the model segments problem areas (e.g. caries, impacted or missing teeth).
- A **U-Net-based binary segmentation** identifies dental regions (target vs. background).
- I trained a **U-Net with VGG-16 encoder**, optimized to classify each pixel correctly.

---
 
## 📊 Dataset Details

- **Source:** Kaggle ([link](https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task)), originally from Roboflow.
- **Full code on Kaggle:** ([link](https://www.kaggle.com/code/mohamedali020/dental-x-ray-binary-segmentation-u-net-vgg16))
---

Dental-Xray-Segmentation/
│
├── train/
│ ├── train_images/
│ └── train_mask/
│
├── valid/
│ ├── valid_images/
│ └── valid_mask/
│
├── test/
│ ├── test_images/
│ └── test_mask/
│
├── train_annotations.coco.json
├── valid_annotations.coco.json
├── test_annotations.coco.json

---

- **Counts & Splits:**
  - Train: 4,772 images + masks  
  - Validation: 2,071 images + masks  
  - Test: 1,345 images + masks  
- **Resolution:** Mostly 640×640 px  
- **Classes:** {1,2,3,…,14} annotation IDs in masks  
- **Samples:**  
  ![Dataset Input Sample](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/0d220dea-Farasati_Simin_35yo_08062021_145847_jpg.rf.478a679c3667801fa26068e518dea362.jpg)  
  ![Dataset Mask Sample](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/00cf39c1-Karaptiyan_Robert_50yo_13032021_185908_jpg.rf.98b2e72cb9a26e75d40df97e04473ada.jpg_mask.png)

---

## ⚙️ Methodology

1. **Data Collection** → Gather and inspect the dataset.  
2. **Pre‑processing** → Resize, normalize, and ensure consistent formatting.  
3. **Data Augmentation** (via Albumentations):
   - Horizontal flip (70%)  
   - Rotation ≤ 5° (70%)  
   - Random brightness/contrast (limit 0.1, 30%)  
   - Shift/Scale/Rotate (10°, 50%)  
4. **Segmentation Model** → Build U-Net with VGG‑16 backbone.  
5. **Hyper‑tuning & Metrics** → Use Dice loss; track accuracy & Dice coefficient.

---

## 🧱 Model Architecture

- **U‑Net**: Encoder–decoder with skip connections, ideal for medical imaging.  
- **Encoder:** VGG‑16 (ImageNet pretrained) for robust feature extraction.  
- **Decoder:** Upsampling layers with skip‑connections to restore spatial details.  


---

## 📈 Results

After 8 epochs:

| Metric            | Training | Validation |
|------------------|----------|------------|
| Accuracy         | 0.9859   | 0.9878     |
| Dice Coefficient | 0.71     | 0.7123     |

![Training Curves](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/Screenshot_15.png)

---

## 🎯 Final Output

A binary mask highlighting target tooth regions.

![WhatsApp Image Example](https://github.com/mohamedali020/Dental-Panoramic-X-Ray-Segmentation-Using-U-Net-with-VGG-16-Backbone/raw/main/WhatsApp%20Image%202025-06-17%20at%2017.56.19_65fbd25a.jpg)

---

## 🔭 Future Work & Developments

- **Speed Optimizations:** Reduce computational cost via model trimming or quantization.  
- **Smart Imaging:** Automatically enhance low-quality X-rays (denoising, resolution upscaling).  
- **Multi-class Segmentation:** Use Mask R‑CNN or multiclass U-Net for identifying tooth types.  
- **Motion/Shadow Correction:** Handle real-world X-ray artifacts.  
- **Clinical Integration:** Incorporate dentist feedback into model refinement.

---

##  Conclusion

This project demonstrates the potential of deep learning—specifically U-Net with a VGG16 backbone—in the field of dental image analysis. Automating the segmentation of dental X-rays provides a foundation for developing intelligent tools to support dentists in diagnosis, treatment planning, and patient communication. With further improvements and integration into real-world systems, such models can contribute to faster, more accurate, and more accessible dental care.



## 🛠️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/USERNAME/REPO.git
cd REPO

# Install dependencies (Python 3.7+)
pip install -r requirements.txt

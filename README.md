# Dental X-ray Images Analysis Using Deep Learning (Segmentation Task) 🦷

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![GitHub stars](https://img.shields.io/github/stars/USERNAME/REPO.svg)](https://github.com/USERNAME/REPO/stargazers) [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=fff)](https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task)




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

**Project Title:** AI-based Dental X-ray Analysis using Custom UNet Segmentation with VGG19 Backbone

**Description:** As part of my graduation project, I developed the computer vision component of an AI Vision System integrated into a dental application. The goal was to assist dentists in diagnosing dental conditions by automatically segmenting problem regions from panoramic X-ray images sent by patients.

I built a custom semantic segmentation model based on the U-Net architecture, enhanced with a VGG19 backbone for feature extraction. The model was trained on a publicly available annotated dataset (14 dental condition classes) using preprocessing, augmentation, and advanced training techniques to improve accuracy and reduce overfitting.

**Key Contributions:**

Applied semantic segmentation for dental diagnostics

Customized U-Net with VGG19 as encoder (transfer learning)

Used Roboflow dataset with pixel-level annotations

Integrated preprocessing pipeline: generate masks through annotation files, data imbalance, resizing, normalization, thresholding, and augmentation, and data generation

**Model Evaluation:** To evaluate the performance of my segmentation model, I used several metrics, including:

Dice Coefficient, Jaccard Index (IoU), F1-Score, Precision, and Recall However, I focused mainly on the Dice Coefficient and IoU (Intersection over Union) as they are the most reliable and commonly used metrics in semantic segmentation tasks due to their effectiveness in measuring overlap between predicted masks and ground truth.

Final output: precise mask highlighting dental problem regions


---
 
## 📊 Dataset Details

- **Source:** Kaggle ([link](https://www.kaggle.com/datasets/mohamedali020/dental-x-raypanoramic-semanticsegmentation-task/data)), originally from Roboflow ([link](https://universe.roboflow.com/arshs-workspace-radio/vzrad2))
- **Full code on Kaggle:** ([link](https://www.kaggle.com/code/mohamedali020/ai-based-dental-x-ray-analysis-using-custom-unet-s)))
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
4. **Segmentation Model** → Build Custom segmentation U-Net with VGG‑19 backbone.  
5. **Hyper‑tuning & Metrics** → Use Dice loss; track Dice Coefficient, Jaccard Index (IoU), F1-Score, Precision, and Recall.

---

## 🧱 Model Architecture

**Architecture to ai vision system to this project**

**Here in my AI vision system architecture:**

**Part One:**

In preprocessing:
The dataset included annotations, which I used to generate the masks. I also focused on resizing both the image and masks to 256x256, which is a very suitable input size because, in the panoramic x-ray, the target zone is centered, so we don't need the rest of the image. This is because when I initially set the input size to 512*512, this made the model focus more on the pixels in the background. I applied data augmentation using techniques tailored to the nature of the dental x-ray input. I used a data generator, which gave me full control over the data during training, using a batch-by-batch approach. I also applied normalization and set a threshold for the masks, which is what the model needed for the segmentation output.

**Part Two:**
Regarding the Custom Segmentation U-Net with VGG19 as the backbone, I replaced the encoder with VGG19, a type of transfer learning model pre-trained on the ImageNet dataset, to leverage the power of feature extraction, reduce training time, and achieve higher accuracy.
I also developed the decoder with improvements, such as Batch Normalization and Dropout, to enhance model stability and prevent overfitting.

**Part Three:**
Finally, the model produces a mask that defines the problem regions.

![X-ray Example](https://github.com/mohamedali020/AI-based-Dental-X-ray-Analysis-using-Custom-UNet-Segmentation-with-VGG19-Backbone/blob/main/Architicture%20model.png?raw=true)



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

These are samples for the final output. Each image shows the original x-ray image and the truth mask it was used to generate it.
Here is a merge between the raw x-ray and the true mask to show exactly where the problem regions are that the model is supposed to recognize.
After that, the prediction output.

![Final output1](https://github.com/mohamedali020/AI-based-Dental-X-ray-Analysis-using-Custom-UNet-Segmentation-with-VGG19-Backbone/blob/main/Final%20output1.png)

![Final output2](https://github.com/mohamedali020/AI-based-Dental-X-ray-Analysis-using-Custom-UNet-Segmentation-with-VGG19-Backbone/blob/main/Final%20output2.png)

![Final output3](https://github.com/mohamedali020/AI-based-Dental-X-ray-Analysis-using-Custom-UNet-Segmentation-with-VGG19-Backbone/blob/main/Final%20output3.png)


---

## 🔭 Future Work & Developments


**1- Addressing Imbalanced Data**

To tackle data imbalance, I plan to acquire a well-annotated, balanced multi-class segmentation dataset with sufficient samples per class to enhance model performance. I will also apply advanced data augmentation techniques tailored for multi-class segmentation to boost minority class representation. Additionally, I aim to explore weighted loss functions or oversampling strategies to improve training on imbalanced data for my dental X-ray application.

**2- High-Quality Dental X-ray Images**

To improve segmentation accuracy, I will source a dataset with high-resolution dental X-ray images and precise annotations, ensuring clear details and minimal noise. I plan to use advanced imaging techniques, like high-definition X-ray systems, and implement quality control to validate image and annotation integrity before training. These steps will enhance the model's performance for dental X-ray segmentation.

**3- During practical evaluation**

I noticed that the model was able to identify the problem areas in general, but it was unable to separate each problem. Plus, I needed to achieve higher accuracy. Therefore, the next step is to experiment with instance segmentation, and the experiment will be with YOLOv11-seg, which combines object detection and instance segmentation.

---

##  Conclusion

This project demonstrates the potential of deep learning—specifically U-Net with a VGG19 backbone—in the field of dental image analysis. Automating the segmentation of dental X-rays provides a foundation for developing intelligent tools to support dentists in diagnosis, treatment planning, and patient communication. With further improvements and integration into real-world systems, such models can contribute to faster, more accurate, and more accessible dental care.




## 🛠️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/USERNAME/REPO.git
cd REPO

# Install dependencies (Python 3.7+)
pip install -r requirements.txt

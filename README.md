# 🌾 Rice Plant Disease Detection using DenseNet121

This deep learning project detects rice plant diseases from leaf images using a transfer learning approach with the **DenseNet121** model. Early detection enables farmers to take timely action, improving crop health and agricultural outcomes.

---

## 🧠 Model Overview

- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Classifier Head**: GlobalAveragePooling → Dense(128, ReLU) → Dense(9, Softmax)
- **Input Image Size**: 224 x 224 x 3
- **Framework**: TensorFlow / Keras
- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam (learning rate = 0.001)
- **Epochs**: 20
- **Batch Size**: 32

---

## 📁 Dataset Structure
Link for the dataset
https://drive.google.com/drive/folders/1xqp9GtXntHV6gQMLbd-yTdOOTgL3YKD9?usp=sharing

```
/content/
├── train/
│   ├── bacterial_leaf_blight/
│   ├── Brown_spot/
│   ├── Healthy/
│   ├── Hispa/
│   ├── leaf_blast/
│   ├── leaf_scald/
│   ├── Narrow_brown_spot/
│   ├── Shath Blight/
│   └── Tungro/
├── validation/
└── test/
```

- Each class folder contains respective labeled images.
- Ensure `train`, `validation`, and `test` directories are structured similarly.

---

## 🖼️ Disease Classes

1. **bacterial_leaf_blight**
2. **Brown_spot**
3. **Healthy**
4. **Hispa**
5. **leaf_blast**
6. **leaf_scald**
7. **Narrow_brown_spot**
8. **Shath Blight**
9. **Tungro**

---

## 🔄 Data Augmentation

Applied only to training set using `ImageDataGenerator`:

- Rescale: 1/255
- Rotation: ±20°
- Width/Height Shift: ±20%
- Shear, Zoom
- Horizontal Flip
- Fill Mode: `'nearest'`

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/rice-disease-detection.git
cd rice-disease-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Ensure your dataset is available in `/content/train`, `/content/validation`, and `/content/test`.

```bash
python train_model.py
```

> You can rename your training script accordingly.

### 4. Evaluate the Model

```bash
# Inside your training script or separately
model.evaluate(test_generator)
```

---

## 📈 Results

| Metric         | Value     |
|----------------|-----------|
| Validation Acc | ~87.06%      |
| Test Accuracy  | ~85.08%      |
| Epochs         | 20        |

_(Replace XX with your actual values after training.)_

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV (if using image preview)
- Matplotlib (for visualizing results)

Install with:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

---

## ✅ Future Enhancements

- Fine-tune all DenseNet121 layers
- Deploy using Streamlit or Flask
- Export as TFLite for Android apps
- Integrate Grad-CAM for model explainability

---

## 🙋‍♂️ Author

**CHANDRAGI SAIRAM**  
📧 sairamchandragi@gmail.com  
🔗 [[LinkedIn](https://linkedin.com/in/sairam](https://www.linkedin.com/in/sairam-chandragi))

---

## 📌 License

This project is open-source and free to use under the [MIT License](LICENSE).

# Deep-ensemble-learning-based-computer-aided-diagnostics-for-classifying-chest-X-ray-image

---

```markdown
# 🩺 Deep Ensemble Learning-Based Chest X-ray Classifier

![Logo](logo.jpeg)

This repository presents a deep learning-based diagnostic platform for classifying chest X-ray images into **Normal**, **Pneumonia**, and **Tuberculosis** using multiple CNN models. The final model is selected through performance comparison and deployed using **Streamlit**.

> 🚀 **Live Demo**: [https://deep-cxr-diagnosis.streamlit.app](https://deep-cxr-diagnosis.streamlit.app)

---

## 📌 Motivation

Respiratory illnesses like pneumonia and tuberculosis account for millions of deaths annually. Traditional diagnosis from chest X-rays requires trained radiologists and may still be error-prone. This project aims to bridge that gap using artificial intelligence and computer vision to build a reliable, real-time, automated diagnostic tool.

---

## 📁 Dataset Structure

The dataset is organized into 3 primary classes and split into three main directories:

```

/Normal
/PNEUMONIA
/Tuberculosis

````

Subdirectories:
- `train/`
- `validation/`
- `test/`

---

## 🧠 Models Trained

We compared several deep CNN architectures:

| Model           | Parameters | Train Time | Accuracy | Notes                        |
|----------------|------------|------------|----------|------------------------------|
| ResNet50        | ~25M       | ~3.5 hrs   | ✅ High  | Robust feature extraction    |
| DenseNet121     | ~8M        | ~2.8 hrs   | ✅ High  | Lightweight & efficient      |
| InceptionV3     | ~23M       | ~4 hrs     | ✅ High  | Good generalization          |
| MobileNetV2     | ~3.4M      | ~2 hrs     | ✅ Fast  | Great for deployment         |
| Custom CNN      | ~2M        | ~1.5 hrs   | ⚠️ Moderate | Basic structure              |
| **Ensemble**    | Mixed      | ~4.5 hrs   | 🏆 **Best** | Final model for deployment |

---

## 📊 Evaluation Metrics

| Model       | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|--------|----------|
| ResNet50    | 93.8%    | 94.2%     | 93.1%  | 93.6%    |
| DenseNet121 | 94.5%    | 95.1%     | 94.0%  | 94.5%    |
| InceptionV3 | 92.3%    | 92.5%     | 91.7%  | 92.1%    |
| MobileNetV2 | 91.8%    | 92.0%     | 91.0%  | 91.5%    |
| Custom CNN  | 88.2%    | 89.0%     | 87.5%  | 88.2%    |
| **Ensemble**| **96.1%**| **96.5%** | **95.8%**| **96.1%**|

---

## 📦 Installation & Setup

To run the app locally:

```bash
git clone https://github.com/haylisha/chest-xray-classifier.git
cd chest-xray-classifier
pip install -r requirements.txt
streamlit run app.py
````

---

## 💾 Handling Large Model Files (.h5)

To ensure deployment despite the `.h5` model’s large file size, we use **Google Drive** and `gdown` to fetch the model at runtime.

### 🔗 Downloading from Google Drive

We store the trained model on Google Drive and download it when needed.

#### Step-by-Step:

1. Upload your `.h5` model to Google Drive.
2. Set the sharing to **"Anyone with the link can view"**.
3. Copy the file ID from the link.

Example link:

```
[https://drive.google.com/file/d/1Q1rIjW-3kP0NED8n8vhvRYU4NkkfmYX9/view?usp=drive_link]
```

File ID: `1Q1rIjW-3kP0NED8n8vhvRYU4NkkfmYX9`

4. Use the following code in your app:

```python
import os
import gdown
from tensorflow.keras.models import load_model

model_path = "my_model.h5"
file_id = "1Q1rIjW-3kP0NED8n8vhvRYU4NkkfmYX9"
download_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    gdown.download(download_url, model_path, quiet=False)

model = load_model(model_path)
```

5. Be sure `gdown` is listed in `requirements.txt`:

```
gdown
```

---

## 🧪 Notebooks

| File              | Description                            |
| ----------------- | -------------------------------------- |
| `Resnet50.ipynb`  | ResNet50 model training                |
| `densenet.ipynb`  | DenseNet121 training                   |
| `inception.ipynb` | InceptionV3 training                   |
| `mobilenet.ipynb` | MobileNetV2 training                   |
| `RNN.ipynb`       | Experimental model (RNN layers)        |
| `ygg16.ipynb`     | VGG16 variant experiment               |
| `my_model.h5`     | Final trained model (loaded via Drive) |
| `streamlit/`      | Web app UI using Streamlit             |

---

## 🖼️ App Features

* Drag-and-drop X-ray image upload
* Real-time classification
* Class probability display
* Disease diagnosis output
* Grad-CAM heatmap visualization (optional future update)

---

## 🚧 Limitations

* Performance may drop on unseen domains (e.g., low-resolution scans)
* Does not handle multi-label disease detection
* Limited to 3-class classification: Normal, Pneumonia, Tuberculosis

---

## 🌟 Future Work

* Add support for COVID-19 detection
* Optimize model for edge/mobile deployment
* Federated learning for privacy-preserving diagnostics
* Expand to multi-label disease classification
* API integration with hospital information systems

---

## 👨‍💻 Author

**Haylisha** – [GitHub Profile](https://github.com/haylisha)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🙌 Acknowledgements

* TensorFlow & Keras
* Streamlit
* Google Drive for model storage
* NIH CXR datasets
* Open-source community ❤️

```


```

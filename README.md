# Gen-AI-Image-Caption-Generator

# 🖼️ Image Caption Generator with BLIP

This project uses the [Salesforce BLIP (Bootstrapped Language Image Pretraining)](https://huggingface.co/Salesforce/blip-image-captioning-base) model to generate image captions and descriptive narratives from any image. It also includes an option to produce "corrupted" captions by injecting random words for creative tasks or adversarial testing.

---

## 📌 Features

* ✅ Generates image captions using BLIP pre-trained model
* 🧠 Creates detailed, beam-search-based image descriptions
* 🔀 Adds random corruption to captions for robustness/experimentation
* 📷 Visual display with caption or description using `matplotlib`

---

## 🧰 Technologies Used

* Python 🐍
* Hugging Face Transformers 🤗
* PyTorch 🔥
* Matplotlib 📊
* PIL (Pillow) 🖼️
* torchvision

---

## 📦 Installation

1. **Clone this repository**:

   ```bash
   git clone https://github.com/your-username/image-caption-generator.git
   cd image-caption-generator
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   You can create a `requirements.txt` with:

   ```txt
   torch
   torchvision
   transformers
   matplotlib
   Pillow
   ```

---

## 🚀 Usage

### 🔹 Generate Basic Caption

```python
caption = generate_caption(image_path)
display_image_with_caption(image_path, caption)
```

### 🔹 Generate Corrupted Caption

```python
caption = existing_generate_caption(image_path)
display_image_with_caption(image_path, caption)
```

### 🔹 Generate Detailed Description

```python
description = generate_detailed_description(image_path)
display_image_with_description(image_path, description)
```

---

## 📂 Example Output

| Input Image                             | Caption/Description                                            |
| --------------------------------------- | -------------------------------------------------------------- |
| ![Sample](./sample_outputs/sample1.jpg) | *"A plate of delicious Indian shrikhand served with saffron."* |

---

## 🧠 Model

The model used is:

```text
Salesforce/blip-image-captioning-base
```

You can explore it on Hugging Face:
👉 [https://huggingface.co/Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

---

## 📌 Folder Structure (Optional Suggestion)

```text
.
├── blip_captioning.py
├── sample_outputs/
│   └── sample1.jpg
├── requirements.txt
└── README.md
```





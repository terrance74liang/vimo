# ViMo


**ViMo** is a GUI world model designed to predict the next state of a graphical user interface (GUI) in  **image** and **text** form.


## 📦 Installation

We recommend setting up a clean Python environment:

```bash
conda create -n vimo python=3.10 -y
conda activate vimo
pip install -q -U google-genai
```


## 🛠️ Usage

1. Set your Gemini API key inside the script.
2. Run the text prediction script:

```bash
python UI_text_prediction.py
```

## 📂 STR Data Processing

0. We provide the selected episode IDs (`epids`) for both **Android In The Wild (AITW)** and **AndroidControl**, along with their annotations in the data folder.

1. You need to download their images:
   - [AndroidControl](https://github.com/google-research/google-research/tree/master/android_control)
   - [Android In The Wild](https://github.com/google-research/google-research/tree/master/android_in_the_wild)

2. Organize the data as follows:
Image/
└── epid/
└── image_id.png

3. Run `generate_ocr.py` to detect OCR content in the GUI screenshots.

4. Run `check_fix.py` to identify static elements in the UI.
  
## Training Graphic prediction

Please refer to the [IP2P project](https://github.com/timothybrooks/instruct-pix2pix) for training guidance, and use our preprocessed data as input.

## ✅ To-Do List

- [x] Code for  UI text prediction
- [x ] Release the STR dataset process code
- [x] Release instruction for UI graphic prediction
- [ ] Release code for agent 


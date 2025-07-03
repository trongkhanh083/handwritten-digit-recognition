# ✍️ Handwritten Digit Recognition with LeNet-5

A computer vision project that recognizes handwritten digits (0-9) using a trained LeNet-5 CNN model on the MNIST dataset.

---

## 🚀 Features
- Real-time drawing canvas for digit input.
- Predicts digits with confidence scores.
- Preprocesses drawings to match MNIST standards.
- Simple, interactive UI.

## ⚙️ Installation
- Python 3.10+
   ```bash
   git clone https://github.com/trongkhanh083/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   pip install -r requirements.txt
   ```
## 🧠 Training the Model
Run the full pipeline (data preparation → training → evaluation → prediction):
  ```bash
  ./scripts/run_training.sh
  ```
## 🖼️ Web Demo
To run the web interface locally:
  ```bash
  streamlit run streamlit_app.py
  ```

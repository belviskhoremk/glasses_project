# 👓 AI-Powered Assistive Wearable for the Visually Impaired

This project is a smart glasses system designed to assist visually impaired individuals by converting visual information into real-time audio feedback. It features advanced functionalities such as **book reading**, **environmental description**, and **voice-triggered commands**, powered by cutting-edge AI technologies like **YOLOv8**, **ResNet50**, **LLMs (LLaMA, GPT)**, and **Google TTS**.

> ⚠️ **Note:** This repository represents a snapshot created shortly after a hackathon to showcase the project progress at that time.  
> 🔧 **Further development, optimization, and API integrations are ongoing** in a separate private repository.

---

## 🚀 Features

- 📖 **Book Reading Mode**: Captures text from a book and reads it aloud.
- 🌍 **Environment Description Mode**: Detects objects and their spatial context and narrates them.
- 🎤 **Voice Activation**: Activate modes using trigger words like `"read book"` and `"environment"`.
- 🌐 **Multilingual Support**: Select your preferred language during registration.
- ☁️ **Cloud Integration**: Uses Google Cloud APIs to process OCR and NLP tasks.
- 🧠 **LLM-based Text Refinement**: Improves extracted text with models like LLaMA 2.7 and GPT.

---

## 🧠 Tech Stack

| Category                 | Tools & Frameworks                                  |
|--------------------------|-----------------------------------------------------|
| Hardware                 | Raspberry Pi 4, ESP32-CAM, NodeMCU, USB Webcam      |
| OCR                     | Tesseract OCR                                        |
| NLP & Text Completion   | OpenAI API, LLaMA 2.7 via Hugging Face               |
| Object Detection        | YOLOv8                                               |
| Environment Classification | ResNet50                                         |
| Text to Speech          | Google TTS (gTTS)                                    |
| Backend APIs            | Django REST Framework (ongoing in separate repo)     |
| Hosting                 | Google Cloud Platform (GCP)                          |

---

## 🖥️ Setup Instructions (Ubuntu)

Install the required dependencies on Ubuntu:

```bash

sudo apt update
sudo apt install tesseract-ocr libtesseract-dev
sudo apt-get install portaudio19-dev python3-pyaudio
pip install --upgrade pydantic langchain-community
```
Additional libraries used in the code (not limited to):
```bash

pip install gTTS opencv-python requests numpy pillow
pip install openai langchain huggingface_hub
pip install torch torchvision torchaudio

```

## 🧪 How It Works
### 🗣️ Trigger Word System

- Say "book" → Triggers book reading flow:
- Image captured → Text extracted via Tesseract → Text refined using GPT/LLaMA → Played via gTTS
- Say "environment" → Triggers spatial awareness flow:
- Image captured → YOLOv8 detects objects → ResNet classifies environment → Description generated → Played via gTTS__

## 🔀 Related Repository
### 📁 Development Repository
Ongoing improvements, scalable backend APIs, advanced model orchestration, and optimized cloud functions are being maintained in a separate repository.
This current repo is mainly to showcase the MVP built during the hackathon and demonstrate key functional components.

## 📜 License

This project is for academic and research purposes. Contact the authors for commercial use or collaboration.
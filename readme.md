# 🚆 Railway Track Crack Monitoring System

![GitHub stars](https://img.shields.io/github/stars/your-repo?style=social)

## 📌 Overview
The **Railway Track Crack Monitoring System** is an AI-powered solution for detecting cracks and fractures in railway tracks using acoustic wave analysis. By leveraging Raspberry Pi sound sensors, cloud-based processing, and deep learning models, the system enables real-time defect detection, ensuring proactive railway maintenance.

## 🔥 Key Features
✅ **AI-Based Acoustic Monitoring** – Uses Raspberry Pi sound sensors to capture acoustic waves from railway tracks.  
✅ **Cloud-Based Processing** – Converts signals into spectrograms for deep learning analysis.  
✅ **Deep Learning Model** – CNN trained using k-fold cross-validation for high accuracy.  
✅ **Pre-Trained Models Available** – Includes a trained model in `.h5` format, the best model saved as `best_model.keras`.  
✅ **Real-Time Detection** – Instant defect analysis displayed on an Android app.  
✅ **Proactive Maintenance** – Helps prevent accidents by detecting track anomalies early.  

## 🏗 System Architecture
```
📡 Sound Sensors (Raspberry Pi)  →  🌩 Cloud Processing  →  🎛 Spectrogram Conversion  →  🧠 CNN Model  →  📱 Android App
```

## 🚀 How It Works
1️⃣ **Signal Collection:** Raspberry Pi sound sensors capture track vibrations.  
2️⃣ **Data Transmission:** Signals are sent to the cloud for preprocessing.  
3️⃣ **Spectrogram Generation:** Signals are transformed into spectrogram images.  
4️⃣ **AI Processing:** A Convolutional Neural Network (CNN) detects cracks.  
5️⃣ **Real-Time Updates:** Predictions are sent to the Android app for immediate action.  

## ⚙️ Technologies Used
- **Hardware:** Raspberry Pi, Sound Sensors  
- **Software:** Python, TensorFlow/Keras (CNN), OpenCV  
- **Cloud:** AWS/GCP for storage and processing  
- **Mobile App:** Android (Kotlin/Flutter)  

## 📲 Android App Interface
![App Screenshot](https://github.com/kishan0818/Track_Crack/blob/main/app_ss.png?raw=true)


## 🛠 Installation & Setup
### 🔧 Prerequisites
- Raspberry Pi with sound sensors configured
- Cloud storage and compute service
- Python environment with necessary libraries
- Pre-trained models (`model.h5`, `best_model.keras`)

### 🚀 Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/railway-track-monitoring.git
   cd railway-track-monitoring
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the signal processing script:
   ```bash
   python signal_processing.py
   ```
4. Start the Android app and connect to the cloud API.

## 👨‍💻 Contributors
- **Jayakishan B** - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/jayakishan-balagopal-978613300/) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/kishan0818)
- **Karishma S K** - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/karishma-sivakumar-25a3a4300/) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/karishma0624)
- **Nitish R** - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/nitish--rajendran/) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Nitish-Rajendran)
- **Shailesh S** - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/shailesh-s-671b65292/) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/shailesh-s-04)
  
## 📜 License
This project is licensed under the **MIT License**.

---
⭐ **Feel free to star this repo if you find it useful!** 🚀

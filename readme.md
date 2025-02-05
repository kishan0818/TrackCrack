# Railway Track Crack Monitoring System

## Overview
The **Railway Track Crack Monitoring System** is an AI-powered solution designed to detect cracks and fractures in railway tracks using acoustic wave analysis. By leveraging Raspberry Pi sound sensors, cloud-based processing, and deep learning models, the system provides real-time defect detection, enabling proactive railway maintenance and enhancing safety.

## Smart India Hackathon Experience
This project was developed as part of the intra-college **Smart India Hackathon**, where we aimed to create an innovative solution for railway safety. The competition challenged us to integrate hardware and AI-based analysis efficiently, ensuring real-time monitoring. Our participation provided hands-on experience in deploying AI at the edge, optimizing signal processing, and developing a seamless user interface. Through this journey, we collaborated with domain experts, fine-tuned our deep learning model, and overcame real-world implementation challenges.

## Key Features
- **AI-Based Acoustic Monitoring** – Uses Raspberry Pi sound sensors to capture acoustic waves from railway tracks.  
- **Cloud-Based Processing** – Converts signals into spectrograms for deep learning analysis.  
- **Deep Learning Model** – CNN trained using k-fold cross-validation for high accuracy.  
- **Pre-Trained Models Available** – Includes trained models in `.h5` and `best_model.keras` formats.  
- **Real-Time Detection** – Instant defect analysis displayed on an Android app.  
- **Proactive Maintenance** – Helps prevent accidents by detecting track anomalies early.  

## System Architecture
```
Sound Sensors (Raspberry Pi)  →  Cloud Processing  →  Spectrogram Conversion  →  CNN Model  →  Android App
```

## How It Works
1. **Signal Collection:** Raspberry Pi sound sensors capture track vibrations.  
2. **Data Transmission:** Signals are sent to the cloud for preprocessing.  
3. **Spectrogram Generation:** Signals are transformed into spectrogram images.  
4. **AI Processing:** A Convolutional Neural Network (CNN) detects cracks.  
5. **Real-Time Updates:** Predictions are sent to the Android app for immediate action.  

## Technologies Used
- **Hardware:** Raspberry Pi, Sound Sensors  
- **Software:** Python, TensorFlow/Keras (CNN), OpenCV  
- **Cloud:** Firebase for storage and processing  
- **Mobile App:** Android (Kotlin/Flutter)  

## Hardware Circuit Diagram
![Hardware Circuit Diagram](https://github.com/kishan0818/Track_Crack/blob/main/hardware_diagram.png?raw=true)

## Android App Interface
![App Screenshot](https://github.com/kishan0818/Track_Crack/blob/main/app_ss.png?raw=true)

## Installation & Setup
### Prerequisites
- Raspberry Pi with sound sensors configured
- Cloud storage and compute service
- Python environment with necessary libraries
- Pre-trained models (`signal_classification_model.h5`, `best_model.keras`)
- Android Studio installed for mobile app development

### Steps to Run Backend
1. Clone the repository:
   ```bash
   git clone https://github.com/kishan0818/Track_Crack.git
   cd Track_Crack
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the signal processing script:
   ```bash
   python signal_processing.py
   ```
4. Start the cloud server if required and connect the backend to the Android app.

### Steps to Run Android App
1. Open **Android Studio** and load the project from the `android_app` directory.
2. Configure an emulator or connect a physical device.
3. Build and run the application:
   ```bash
   ./gradlew assembleDebug
   ```
4. Start the app and connect it to the backend API.

## Contributors
- **Jayakishan B** - [LinkedIn](https://www.linkedin.com/in/jayakishan-balagopal-978613300/) | [GitHub](https://github.com/kishan0818)
- **Karishma S K** - [LinkedIn](https://www.linkedin.com/in/karishma-sivakumar-25a3a4300/) | [GitHub](https://github.com/karishma0624)
- **Nitish R** - [LinkedIn](https://www.linkedin.com/in/nitish--rajendran/) | [GitHub](https://github.com/Nitish-Rajendran)
- **Shailesh S** - [LinkedIn](https://www.linkedin.com/in/shailesh-s-671b65292/) | [GitHub](https://github.com/shailesh-s-04)
  
## License
This project is licensed under the **MIT License**.

Feel free to star this repository if you find it useful.


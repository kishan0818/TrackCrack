import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('signal_classification_model.h5')

input_shape = (128, 128)

# Spectrogram conversion
def wav_to_spectrogram(filepath, n_fft=2048, hop_length=512):
    y, sr = librosa.load(filepath, sr=None)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram

# Predict the signal class and return spectrogram for plotting
def predict_signal(filepath):
    spectrogram = wav_to_spectrogram(filepath)
    if spectrogram.shape[1] < input_shape[1]:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, input_shape[1] - spectrogram.shape[1])), mode='constant')
    else:
        spectrogram = spectrogram[:, :input_shape[1]]
    spectrogram_input = spectrogram[:input_shape[0], :input_shape[1]][np.newaxis, ..., np.newaxis]

    prediction = model.predict(spectrogram_input)[0]
    classes = ['Stable Surface', 'Surface Deformation', 'Stable Joint', 'Joint Deformation']
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    print(f"Predicted class: {classes[predicted_class]} with confidence")
    
    return spectrogram, classes[predicted_class]

# Plot the spectrograms of all signals together
def plot_all_spectrograms(filepaths, titles):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    
    for i, filepath in enumerate(filepaths):
        spectrogram, predicted_class = predict_signal(filepath)
        img = librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='log', cmap='viridis', ax=axs[i])
        axs[i].set_title(f"{titles[i]} (Predicted: {predicted_class})")
        fig.colorbar(img, ax=axs[i], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

# Paths for the signal files
filepaths = [
    'synthetic_stable_surface_signals/stable_surface_signal_0.wav',
    'synthetic_surface_deformation_signals/surface_deformation_signal_0.wav',
    'synthetic_stable_joint_signals/stable_joint_signal_0.wav',
    'synthetic_joint_deformation_signals/joint_deformation_signal_0.wav'
]

# Titles for the spectrograms
titles = ['Stable Surface Signal', 'Surface Deformation Signal', 'Stable Joint Signal', 'Joint Deformation Signal']

# Plot all spectrograms together
plot_all_spectrograms(filepaths, titles)

import os
import numpy as np
from scipy.io import wavfile
import librosa
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create directories for the new categories
stable_surface_dir = 'synthetic_stable_surface_signals'
surface_deformation_dir = 'synthetic_surface_deformation_signals'
stable_joint_dir = 'synthetic_stable_joint_signals'
joint_deformation_dir = 'synthetic_joint_deformation_signals'

os.makedirs(stable_surface_dir, exist_ok=True)
os.makedirs(surface_deformation_dir, exist_ok=True)
os.makedirs(stable_joint_dir, exist_ok=True)
os.makedirs(joint_deformation_dir, exist_ok=True)

# Signal Generation Functions
def generate_stable_surface_signal(filename, duration=2, sample_rate=22050, freq=440):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    wavfile.write(filename, sample_rate, (signal * 32767).astype(np.int16))

def generate_surface_deformation_signal(filename, duration=2, sample_rate=22050, freq=440, crack_intensity=0.1, crack_probability=0.02):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    cracks = np.random.rand(len(t)) < crack_probability
    noise = crack_intensity * np.random.randn(len(t)) * cracks
    deformed_signal = signal + noise
    wavfile.write(filename, sample_rate, (deformed_signal * 32767).astype(np.int16))

def generate_stable_joint_signal(filename, duration=2, sample_rate=22050, freq=330):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    wavfile.write(filename, sample_rate, (signal * 32767).astype(np.int16))

def generate_joint_deformation_signal(filename, duration=2, sample_rate=22050, freq=330, noise_factor=0.5, deform_intensity=0.4):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = noise_factor * np.random.randn(len(t))
    deformation = deform_intensity * np.sin(2 * np.pi * (freq + 10) * t)
    deformed_joint_signal = signal + noise + deformation
    wavfile.write(filename, sample_rate, (deformed_joint_signal * 32767).astype(np.int16))

# Spectrogram conversion
def wav_to_spectrogram(filepath, n_fft=2048, hop_length=512):
    y, sr = librosa.load(filepath, sr=None)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return log_spectrogram

# Load data from directories
def load_data(directory, label, input_shape=(128, 128)):
    X = []
    y = []
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            filepath = os.path.join(directory, file)
            spectrogram = wav_to_spectrogram(filepath)
            if spectrogram.shape[1] < input_shape[1]:
                spectrogram = np.pad(spectrogram, ((0, 0), (0, input_shape[1] - spectrogram.shape[1])), mode='constant')
            else:
                spectrogram = spectrogram[:, :input_shape[1]]
            X.append(spectrogram[:input_shape[0], :input_shape[1]])
            y.append(label)
    return np.array(X), np.array(y)

# Generate signals
for i in range(500):
    generate_stable_surface_signal(f'{stable_surface_dir}/stable_surface_signal_{i}.wav')
    generate_surface_deformation_signal(f'{surface_deformation_dir}/surface_deformation_signal_{i}.wav')
    generate_stable_joint_signal(f'{stable_joint_dir}/stable_joint_signal_{i}.wav')
    generate_joint_deformation_signal(f'{joint_deformation_dir}/joint_deformation_signal_{i}.wav')

# Load spectrograms and labels
input_shape = (128, 128)
X_stable_surface, y_stable_surface = load_data(stable_surface_dir, label=0, input_shape=input_shape)
X_surface_deformation, y_surface_deformation = load_data(surface_deformation_dir, label=1, input_shape=input_shape)
X_stable_joint, y_stable_joint = load_data(stable_joint_dir, label=2, input_shape=input_shape)
X_joint_deformation, y_joint_deformation = load_data(joint_deformation_dir, label=3, input_shape=input_shape)

# Concatenate data and reshape for CNN
X = np.concatenate((X_stable_surface, X_surface_deformation, X_stable_joint, X_joint_deformation), axis=0)
y = np.concatenate((y_stable_surface, y_surface_deformation, y_stable_joint, y_joint_deformation), axis=0)
X = X[..., np.newaxis]

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=4)

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build CNN model with adjusted architecture
def build_model():
    model = Sequential([
        Input(shape=(input_shape[0], input_shape[1], 1)),
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.4),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, save_weights_only=False)  # Changed to .keras

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in kf.split(X_train_final):
    X_train_cv, X_val_cv = X_train_final[train_index], X_train_final[val_index]
    y_train_cv, y_val_cv = y_train_final[train_index], y_train_final[val_index]
    
    model = build_model()
    history = model.fit(datagen.flow(X_train_cv, y_train_cv, batch_size=32), epochs=30, validation_data=(X_val_cv, y_val_cv), verbose=2, callbacks=[early_stopping, checkpoint])
    
    val_loss, val_accuracy = model.evaluate(X_val_cv, y_val_cv, verbose=0)
    cv_scores.append(val_accuracy)

print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores):.4f}")

# Final training on full dataset with data augmentation
model = build_model()
history = model.fit(datagen.flow(X_train_final, y_train_final, batch_size=32), epochs=30, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Save the final model
model.save('signal_classification_model.h5')

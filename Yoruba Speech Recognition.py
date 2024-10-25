# Import necessary libraries
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling1D # type: ignore
from sklearn.model_selection import train_test_split

# Load your dataset
#audio_dir = r"C:/Users/User/Downloads/yo_ng_female.wav"
transcripts_file = pd.read_csv(r"C:/Users/User/Downloads/yo_ng_female/line_index_female.tsv", sep='\t')
# Rename columns manually
transcripts_file.columns = ['audio_file', 'transcription']

# Check the columns
#print(transcripts_file.head(50))

audio_dir = r"C:/Users/User/Downloads/yo_ng_female"
audiofiles = os.listdir(audio_dir)
"""for file in audiofiles:
    if file.endswith('.wav'):  # Process only .wav files
        file_path = os.path.join(audio_dir, file)  # Create the full path to the file
        y, sr = librosa.load(file_path, sr=None)  # Load the audio file

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # n_mfcc: number of MFCCs to return

        # Display MFCCs using a heatmap
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'MFCC for {file}')
        plt.tight_layout()
        plt.show()"""

transcript_dict = {row['audio_file']: row['transcription'] for _, row in transcripts_file.iterrows()}

# Preprocess functions
def preprocess_text(transcripts):
    tokenizer = Tokenizer(char_level=True)  # Character-level tokenizer for transcription
    tokenizer.fit_on_texts(transcripts)  # Fit on all transcription text
    sequences = tokenizer.texts_to_sequences(transcripts)  # Convert to sequences
    return sequences, tokenizer

def load_and_preprocess_audio(audio_path):
    """Load audio file and compute its Mel spectrogram."""
    try:
        # Load audio file with librosa
        audio, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16kHz

        # Generate Mel-spectrogram using librosa's built-in function
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)

        # Convert to dB scale
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
      
# Prepare dataset
transcripts = transcripts_file['transcription'].tolist()  # Get all transcription texts
audio_files = transcripts_file['audio_file'].tolist()  # Get corresponding audio filenames
sequences, tokenizer = preprocess_text(transcripts)

X_train = []
y_train = []

for i, audio_file in enumerate(audio_files):
    audio_path = os.path.join(audio_dir, audio_file + '.wav')  # Add the .wav extension
    print(f"Processing audio file: {audio_file} | Audio path: {audio_path}")

    if os.path.exists(audio_path):
        mel_spectrogram = load_and_preprocess_audio(audio_path)  # Call updated function

        if mel_spectrogram is not None:
            X_train.append(mel_spectrogram.T)  # Transpose to match expected input shape
            y_train.append(sequences[i])  # Append corresponding sequence
            print(f"Added sequence for {audio_file}")
        else:
            print(f"Failed to process {audio_file} due to audio processing error.")
    else:
        print(f"Audio file {audio_file}.wav not found.")

# Final check on processed samples
#print(f"Total processed samples: {len(X_train)}, Total sequences: {len(y_train)}")


# Padding the input Mel spectrograms (X_train)
max_len_audio = max([x.shape[0] for x in X_train])  # Get the maximum length of the audio samples
X_train_padded = pad_sequences(X_train, maxlen=max_len_audio, padding='post', dtype='float32')


# Padding the output sequences (y_train)
max_len_text = max([len(seq) for seq in y_train])  # Get the maximum length of the transcription sequences
y_train_padded = pad_sequences(y_train, maxlen=max_len_text, padding='post')

X_train_padded = pad_sequences(X_train, maxlen=max_len_text, padding='post', dtype='float32')


# Convert labels to categorical format (one-hot encoding)
num_classes = len(tokenizer.word_index) + 1  # +1 for the padding token
y_train_one_hot = to_categorical(y_train_padded, num_classes=num_classes)

# Verify shapes
print(f"X_train shape: {X_train_padded.shape}")
print(f"y_train shape: {y_train_one_hot.shape}")

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_padded, y_train_one_hot, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val_split)}")

model = Sequential()

# Add LSTM layer
model.add(LSTM(256, input_shape=(X_train_padded.shape[1], X_train_padded.shape[2]), return_sequences=True))

# Add dropout for regularization
model.add(Dropout(0.3))


# Add time-distributed dense layer
model.add(TimeDistributed(Dense(128, activation='relu')))

# Output layer
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=6, batch_size=32
)

val_loss, val_acc = model.evaluate(X_val_split, y_val_split)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

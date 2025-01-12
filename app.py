import os
import streamlit as st
import numpy as np
import librosa
import torch
from  keras.models import load_model
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, AutoProcessor, AutoModelForAudioClassification
from torch import nn


# Define the LSTM model for PyTorch
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_labels, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.reshape = nn.Linear(input_size, input_size)  # Reshape input
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.reshape(x)  # Reshape to add a sequence dimension of 1
        x = x[:, None, :]    # Reshape to (batch_size, 1, input_size)
        _, (hidden, _) = self.lstm(x)  # hidden shape: (num_layers, batch_size, hidden_size)
        hidden = hidden[-1]
        output = self.dropout(hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output


# Dynamic model loader for TensorFlow, PyTorch, and Hugging Face
@st.cache_resource
def load_model_dynamic(model_info):
    model_path = model_info["path"]
    model_type = model_info["type"]

    if model_type == "h5":
        return load_model(model_path)  # Load TensorFlow model
    elif model_type == "pth":
        return load_pytorch_model(model_path)  # Load PyTorch model
    elif model_type == "hf":
        # Load Hugging Face model and feature extractor
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        return model, feature_extractor
    elif model_type =="safetensors":
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        return model, processor
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Load PyTorch LSTM model
def load_pytorch_model(model_path):
    input_size = 16000  # Match the feature dimension of your preprocessed data
    hidden_size = 256
    num_layers = 2
    num_labels = 10  # Number of genres
    dropout = 0.3

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_labels, dropout)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Audio preprocessing for Hugging Face models
def preprocess_audio_hf(file_path, feature_extractor, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    inputs = feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    return inputs.input_values


# Preprocess audio for other models
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def preprocess_format_audio(file_path, sampling_rate, max_dura, processor):
    try:
        audio, _ = librosa.load(file_path, sr=sampling_rate)
        audio =  audio[:max_dura]
        np.savetxt('aaa.txt', audio, delimiter=',')
        inputs = processor(audio,sampling_rate= sampling_rate, return_tensors="pt").input_values.squeeze(0)
    except Exception as e:
        print(f'Failed to process {file_path} : {e}')
    return inputs

# Pad or truncate embeddings
def pad_or_truncate(embedding, target_length=16000):
    if len(embedding) > target_length:
        return embedding[:target_length]
    else:
        padding = target_length - len(embedding)
        return np.pad(embedding, (0, padding), mode='constant')


# Dynamically scan for Hugging Face model directories
def discover_hf_models(base_path):
    hf_models = {}
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path) and all(
            file in os.listdir(folder_path) for file in ["model.safetensors", "config.json", "preprocessor_config.json"]
        ):
            hf_models[folder] = {"path": folder_path, "type": "hf"}
    return hf_models


# Define genres
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Static model paths
MODEL_PATHS = {
    # "ANN Model": {
    #     "path": 'Inference_Models\\trained_model.h5',
    #     "type": "h5",
    # },
    "LSTM PyTorch Model": {
        "path": 'Inference Models\lstm_model.pth',
        "type": "pth",
    },
    "Wav2Vec 5 Seconds" : {
        "path" : 'Inference Models\model-5secs-checkpoint-4100',
        "type" : "safetensors"
    },
    "Wav2Vec 10 Seconds" : {
    "path" : 'Inference Models\model-10secs-checkpoint-5900',
    "type" : "safetensors"
    },
    "Wav2Vec 20 Seconds" : {
    "path" : 'Inference Models\model-20secs-checkpoint-5400',
    "type" : "safetensors"
    },
}

# Streamlit UI
st.title("Music Genre Classification")
st.write("Upload an audio file to classify its genre.")

# Model selection dropdown
selected_model_name = st.selectbox("Select a model:", list(MODEL_PATHS.keys()))
selected_model_info = MODEL_PATHS[selected_model_name]

# Load the selected model
model_data = load_model_dynamic(selected_model_info)

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    with open(f'temp_audio.{file_extension}', 'wb')as f:
        f.write(uploaded_file.read())
    # Display the uploaded audio file in the app    
    st.audio(f'temp_audio.{file_extension}', format=f"audio/{file_extension}")
    st.write("Audio uploaded successfully. Click 'Predict' to classify the genre.")
    
if st.button("Predict"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if uploaded_file is None:
        st.error("Please upload an audio file before clicking 'Predict'.")
    else:
        try:
            model_type = selected_model_info["type"]
            if model_type == "hf":
                model, feature_extractor = model_data
                inputs = preprocess_audio_hf("temp_audio.wav", feature_extractor)
                with torch.no_grad():
                    logits = model(inputs).logits
                    probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            elif model_type == "safetensors":
                model, processor = model_data
                sr = 16000
                max_duration = sr * 20 
                inputs = preprocess_format_audio("temp_audio.wav", sampling_rate=sr, max_dura= max_duration, processor= processor )
                with torch.no_grad():
                    logits = model(inputs.detach().unsqueeze(0).to(device)).logits
                    probabilities = torch.softmax(logits, dim=1)[0]
            else:
                embedding = preprocess_audio("temp_audio.wav")
                input_data = pad_or_truncate(embedding, target_length=16000)
                input_data = np.array(input_data).reshape(1, -1)

                if model_type == "h5":
                    probabilities = model_data.predict(input_data)[0]   
                elif model_type == "pth":
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                    probabilities = model_data(input_tensor).detach().numpy()[0]

            predicted_genre_index = np.argmax(probabilities)
            predicted_genre = GENRES[predicted_genre_index]
            predicted_confidence = probabilities[predicted_genre_index] * 100

            st.write(f"**Model Used:** {selected_model_name}")
            st.write(f"**Predicted Genre:** {predicted_genre}")
            st.write(f"**Confidence:** {predicted_confidence:.2f}%")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

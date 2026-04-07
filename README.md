# EchoSense: Speech Emotion Recognition

EchoSense is an AI-powered web application for real-time speech emotion recognition. Using advanced machine learning techniques, it analyzes audio input to detect emotions such as happiness, sadness, anger, fear, and more through MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.

## Features

- **Real-time Emotion Detection**: Upload audio files or record directly from your microphone
- **MFCC Feature Extraction**: Advanced audio processing for accurate emotion classification
- **Random Forest Classifier**: Robust machine learning model trained on the RAVDESS dataset
- **Interactive Web Interface**: Built with Streamlit for easy access and visualization
- **Audio Visualization**: Waveform and MFCC spectrogram displays
- **Dark Theme UI**: Modern, professional interface design

## Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which contains:
- 24 professional actors (12 female, 12 male)
- 8 emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- Audio-only files in WAV format
- 1440 audio files total

The dataset is organized in the `data/RAVDESS/` directory with actor folders.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd EchoSense
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the RAVDESS dataset** (if not already present):
   - Visit the [RAVDESS website](https://zenodo.org/record/1188976)
   - Download the audio-only files
   - Extract to `data/RAVDESS/audio_speech_actors_01-24/`

## Training the Model

To train the emotion recognition model:

1. Ensure the dataset is properly placed in `data/RAVDESS/`
2. Run the training script:
   ```bash
   python scripts/train_model.py
   ```
   This will create a `models/model.pkl` file containing the trained Random Forest classifier.

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:
   - Open your browser to `http://localhost:8501`
   - Choose between uploading an audio file or recording directly
   - View the predicted emotion and audio visualizations

## Project Structure

```
EchoSense/
├── app.py                 # Main Streamlit application
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── data/
│   └── RAVDESS/          # RAVDESS dataset
├── src/
│   ├── __init__.py
│   ├── dataset.py        # Dataset loading utilities
│   ├── emotion_labels.py # Emotion label extraction
│   ├── feature_extraction.py # MFCC feature extraction
│   └── predict.py        # Emotion prediction functions
├── scripts/
│   └── train_model.py    # Model training script
├── models/
│   └── model.pkl         # Trained machine learning model
├── experimenting/         # Experimental code and alternative models
│   ├── dataset.py
│   ├── emotion_labels.py
│   ├── feature_extraction.py
│   ├── KNN.py
│   ├── RandomForest.py
│   └── SVM_models.py
└── tests/                 # Unit tests and validation scripts
```

## How It Works

1. **Audio Input**: User uploads a WAV file or records audio
2. **Feature Extraction**: MFCC coefficients are extracted from the audio signal
3. **Model Prediction**: The trained Random Forest classifier predicts the emotion
4. **Visualization**: Results are displayed with waveform and MFCC spectrograms

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib**: Data visualization
- **NumPy**: Numerical computing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RAVDESS dataset creators and contributors
- Librosa library developers
- Streamlit community
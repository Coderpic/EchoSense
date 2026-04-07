import pickle
from .feature_extraction import extract_features
from glob import glob

def predict_emotion(file):
    model = pickle.load(open("models/model.pkl","rb"))

    features = extract_features(file)
    features = features.reshape(1,-1)

    return model.predict(features)[0]

# testing the predict_emotion function
# audio_files = glob('./data/RAVDESS/*/*.wav')
# print(predict_emotion(audio_files[5]))

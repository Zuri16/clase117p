import tensorflow 
import keras.api._v2.keras as keras 
import numpy as np 
import pandas as pd 
from keras.preprocessing.text import Tokenizer 
from keras_preprocessing.sequence import pad_sequences 
from keras.models import load_model

# datos de entrenamiento
train_data = pd.read_csv("./static/data_files/tweet_emotions.csv")    
training_sentences = []

for i in range(len(train_data)):
    sentence = train_data.loc[i, "content"]
    training_sentences.append(sentence)

# cargar modelo
model = load_model("./static/model_files/Tweet_Emotion.h5")

vocab_size = 40000
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# asignar emoticones para diferentes emociones
emo_code_url = {
    "empty": [0, "./static/emoticons/Empty.png"],
    "sadness": [1,"./static/emoticons/Sadness.png" ],
    "enthusiasm": [2, "./static/emoticons/Enthusiasm.png"],
    "neutral": [3, "./static/emoticons/Neutral.png"],
    "worry": [4, "./static/emoticons/Worry.png"],
    "surprise": [5, "./static/emoticons/Surprise.png"],
    "love": [6, "./static/emoticons/Love.png"],
    "fun": [7, "./static/emoticons/fun.png"],
    "hate": [8, "./static/emoticons/hate.png"],
    "happiness": [9, "./static/emoticons/happiness.png"],
    "boredom": [10, "./static/emoticons/boredom.png"],
    "relief": [11, "./static/emoticons/relief.png"],
    "anger": [12, "./static/emoticons/anger.png"]
    
    }
# escribir la función para predecir la emoción
def predict(text):
    predicted_emotion=''
    predicted_emotion_emoticon=''
    if(text != ''):
        sentence=[]
        sentence.append(text)
        sequences=tokenizer.texts_to_sequences(sentence)
        padded=pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
        result=model.predict(padded)
        prediccion_final=np.argmax(result,axis=1)
        print(prediccion_final)
        for key,value in emo_code_url.items():
            if (value[0]==prediccion_final):
                predicted_emotion_emoticon=value[1]
                predicted_emotion=key
        return predicted_emotion,predicted_emotion_emoticon

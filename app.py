from flask import Flask, render_template, url_for, request, jsonify
from text_sentiment_prediction import *

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/predict-emotion', methods=["POST"])
def predict_emotion():
    
    # Obtener el texto ingresado del requerimiento POST.
    input_text=request.json.get('text')
    
    if not input_text:
        # Respuesta para enviar si input_text está indefinido.
        response={
            'status':'error',
            'message':'ingresa texto',
        }
        return jsonify(response)
        # Respuesta para enviar si input_text no está indefinido.
        
        # Enviar respuesta.         
    else:
        predicted_emotion,predicted_emotion_emoticon=predict(input_text)   
        response={
            'status':'succes',
            'data':{
                'predicted_emotion':predicted_emotion,
                'predicted_emotion_emoticon':predicted_emotion_emoticon
            }
        }
        return jsonify (response)
       
app.run(debug=True)



    
//Crear la variable date - (fecha).
var date=new Date()
var display_date='fecha'+date.toLocaleDateString()

//Cargar HTML DOM.
$(document).ready(function(){
    $('#display_date').html(display_date)
})

//Definir la variable para almacenar la emoción predecida.
var predicted_emotion

//HTML-->JavaScript--->Flask.
//Flask--->JavaScript--->HTML.

//Selector jQuery y la acción click.

$(function () {
    $("#predict_button").click(function () {
        var input_data={
            'text':$('#text').val()
        }
        console.log(input_data)
        //Llamada a AJAX 

        $.ajax({
            type:'POST',url:'/predict-emotion',data:JSON.stringify(input_data),dataType:'json',contentType:'aplication/json',
            
            success:function(result){
                
                // Resultado recibido de Flask ----->JavaScript
                predicted_emotion=result.data.predicted_emotion
                emo_url = result.data.predicted_emotion_emoticon

                // Mostrar resultado usando JavaScript----->HTML
                $('#prediction').html(predicted_emotion)
                $('#prediction').css('display','block')
                $('#emo_img_url').attr('src',emo_url)
                $('#emo_img_url').css('display','block')
                
            },
            //Función error 
            error:function(result){
                alert(result.responseJSON.mesagge)
            }
        });
    });
})


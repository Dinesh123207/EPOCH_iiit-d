from flask import Flask , render_template, request, redirect, url_for

app=Flask(__name__,template_folder='templates')
sample_text = "This is the sample text from the Python code."
@app.route('/')
def index():
    sample_text1 = "Sample text from Python"
    sample_text2 = "Sample text from Python"
    return render_template('index.html', sample_text1=sample_text1,sample_text2=sample_text2)
    

@app.route('/process_audio', methods=['POST'])
def process_audio():
    
    if 'audio_file' in request.files:
        # Handle uploaded audio file
        audio_file = request.files['audio_file']
        audio_file.save('uploads/uploaded_audio.wav')
        return "Uploaded audio file received successfully!"
    elif 'audio_blob' in request.form:
        # Handle recorded audio blob
        audio_blob = request.form['audio_blob']
        audio_data = audio_blob.split(",")[1]
        audio_binary = bytes(audio_data, 'utf-8')
        with open('uploads/recorded_audio.wav', 'wb') as f:
            f.write(audio_binary)
        print("abc")
        return "Recorded audio blob received successfully!"
    elif request.headers['Content-Type'].startswith('audio/'):
        # Handle recorded audio sent via fetch request
        audio_blob = request.data
        with open('uploads/recorded_audio.wav', 'wb') as f:
            f.write(audio_blob)
        return "Recorded audio from fetch request received successfully!"
    else:
        return redirect(url_for('index'))
    
if __name__=='__main__':
    app.run(debug=True,port=5002)
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, session, flash
import speech_recognition as sr
import pyttsx3
import os
from pathlib import Path
from g4f.client import Client
import io
import time
import re
from pydub import AudioSegment
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import json
import bcrypt

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key

# Initialize g4f client and EasyOCR reader
client = Client()
reader = easyocr.Reader(['en'], gpu=False)

# JSON database file
USERS_DB = 'users.json'

# Initialize users.json if it doesn't exist
if not os.path.exists(USERS_DB):
    with open(USERS_DB, 'w') as f:
        json.dump({}, f)

# Helper functions for user management
def load_users():
    try:
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Preprocess image for OCR
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = img.filter(ImageFilter.MedianFilter())
        preprocessed_path = os.path.join('temp', f"preprocessed_{int(time.time())}.jpg")
        img.save(preprocessed_path)
        return preprocessed_path
    except Exception as e:
        return f"Error preprocessing image: {str(e)}"

# Enhance text with GPT
def enhance_text_with_gpt(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI that enhances unclear transcriptions into clear, accurate text for dyslexia detection while preserving original letter case and sequence where possible. Improve the following transcription."},
                {"role": "user", "content": f"Enhance this transcription: {text}"}
            ],
            web_search=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error enhancing text with GPT: {str(e)}"

# Calculate dyslexia risk score
def calculate_dyslexia_risk(text):
    if not text or "Error" in text:
        return 0, "No valid text to analyze"
    
    score = 0
    details = []
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    for i in range(len(words) - 1):
        if words[i].lower() == words[i + 1].lower():
            score += 10
            details.append(f"Repeated word: '{words[i]}'")
    
    swap_patterns = [
        (r'b', r'd'), (r'd', r'b'),
        (r'p', r'q'), (r'q', r'p'),
        (r'w', r'm'), (r'm', r'w')
    ]
    for word in words:
        for orig, swap in swap_patterns:
            if re.search(orig, word.lower()) and re.search(swap, word.lower()):
                score += 15
                details.append(f"Possible letter swap in '{word}' (e.g., {orig}/{swap})")
                break
    
    avg_words_per_sentence = sum(len(s.split()) for s in sentences if s.strip()) / max(len([s for s in sentences if s.strip()]), 1)
    if avg_words_per_sentence < 4:
        score += 15
        details.append(f"Short phrasing detected (avg {avg_words_per_sentence:.1f} words/sentence)")
    
    word_lengths = [len(word) for word in words]
    if word_lengths and max(word_lengths) - min(word_lengths) > 5:
        score += 10
        details.append("High variability in word lengths")
    
    filler_words = ['uh', 'um', 'like', 'er', 'ah']
    filler_count = sum(word.lower() in filler_words for word in words)
    if filler_count > 0:
        score += 5 * filler_count
        details.append(f"Found {filler_count} filler word(s) (e.g., 'uh', 'um')")
    
    if sum(1 for c in text if c.isupper()) > len(sentences) * 2:
        score += 10
        details.append("Inconsistent capitalization detected")
    
    score = min(score, 100)
    
    if score == 0:
        details.append("No significant dyslexia indicators found")
    
    return score, "; ".join(details)

# Calculate eye movement risk
def calculate_eye_movement_risk(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, "Error: Could not open video file"
    
    frame_count = 0
    movement_score = 0
    prev_frame = None
    details = []

    while frame_count < 20 * 30:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            movement = np.sum(diff) / (diff.size * 255)
            if movement > 0.05:
                movement_score += 10
                details.append(f"Significant movement detected at frame {frame_count}")
        
        prev_frame = gray
        frame_count += 1
    
    cap.release()
    movement_score = min(movement_score, 100)
    
    if movement_score == 0:
        details.append("No significant eye movement detected")
    else:
        details.append(f"Total movement score: {movement_score}")
    
    return movement_score, "; ".join(details)

# Generate quiz
import json
import openai  # Make sure this is properly configured

def generate_quiz(risk_score):
    try:
        # Determine difficulty level based on risk score
        if risk_score <= 33:
            difficulty = "low"
        elif risk_score <= 66:
            difficulty = "medium"
        else:
            difficulty = "high"

        # Define prompt with better structure and difficulty hint
        prompt = f"""
        Generate a 10-question multiple-choice reading quiz for a Dyslexia Risk Score of {risk_score}/100 (difficulty: {difficulty}).

        Instructions:
        - Tailor the questions based on difficulty level:
          - Low: simple words, short sentences.
          - Medium: moderate vocabulary and reasoning.
          - High: complex sentences, inference, abstract thinking.
        - Each question should have:
          - 4 options: a, b, c, d
          - One correct answer
        - Do not include a reading passage.
        - Return ONLY valid JSON in the following format:

        {{
            "risk_score": {risk_score},
            "difficulty": "{difficulty}",
            "questions": [
                {{
                    "question": "text",
                    "options": {{
                        "a": "text",
                        "b": "text",
                        "c": "text",
                        "d": "text"
                    }},
                    "answer": "a" | "b" | "c" | "d"
                }},
                ...
            ]
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an educational assistant creating dyslexia-friendly assessments."},
                {"role": "user", "content": prompt}
            ]
        )

        quiz_json = response.choices[0].message.content.strip()

        # Clean up JSON if needed
        json_start = quiz_json.find('{')
        json_end = quiz_json.rfind('}') + 1
        quiz_cleaned = quiz_json[json_start:json_end]

        quiz = json.loads(quiz_cleaned)

        # Validate structure
        if not isinstance(quiz, dict) or 'questions' not in quiz:
            raise ValueError("Invalid quiz structure")

        return quiz

    except Exception as e:
        print(f"Error generating quiz: {str(e)}. Returning fallback quiz.")

        return {
            "risk_score": risk_score,
            "difficulty": "low",
            "questions": [
                {
                    "question": "What is the capital of France?",
                    "options": {
                        "a": "Florida",
                        "b": "France",
                        "c": "Paris",
                        "d": "London"
                    },
                    "answer": "c"
                }
            ] * 10
        }


# Process image
def process_image(image_file):
    try:
        temp_file_path = os.path.join('temp', f"image_{int(time.time())}.jpg")
        os.makedirs('temp', exist_ok=True)
        image_file.save(temp_file_path)
        
        preprocessed_path = preprocess_image(temp_file_path)
        if isinstance(preprocessed_path, str) and preprocessed_path.startswith("Error"):
            os.remove(temp_file_path)
            return preprocessed_path
        
        result = reader.readtext(preprocessed_path, detail=1, paragraph=True)
        
        extracted_lines = []
        for item in result:
            if len(item) == 3:
                bbox, text, _ = item
            elif len(item) == 2:
                bbox, text = item
            else:
                continue
            extracted_lines.append((bbox, text))
        
        sorted_lines = sorted(extracted_lines, key=lambda x: (x[0][0][1], x[0][0][0]))
        extracted_text = "\n".join([text for (_, text) in sorted_lines])
        
        os.remove(temp_file_path)
        os.remove(preprocessed_path)
        
        if not extracted_text.strip():
            return "Error: No text detected in the image"
        
        return extracted_text
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Process audio bytes
def process_audio_bytes(audio_data):
    temp_file_path = os.path.join('temp', f"realtime_{int(time.time())}.wav")
    temp_raw_path = os.path.join('temp', f"realtime_raw_{int(time.time())}.webm")
    os.makedirs('temp', exist_ok=True)
    
    try:
        with open(temp_raw_path, 'wb') as f:
            f.write(audio_data)
        audio = AudioSegment.from_file(temp_raw_path)
        audio.export(temp_file_path, format="wav")
        result = audio_to_text(temp_file_path)
    except Exception as e:
        return f"Error processing audio bytes: {str(e)}"
    finally:
        for path in [temp_raw_path, temp_file_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    time.sleep(1)
                    os.remove(path)
    
    return result

# Process video bytes
def process_video_bytes(video_data):
    temp_file_path = os.path.join('temp', f"eye_movement_{int(time.time())}.webm")
    os.makedirs('temp', exist_ok=True)
    
    try:
        with open(temp_file_path, 'wb') as f:
            f.write(video_data)
        score, details = calculate_eye_movement_risk(temp_file_path)
    except Exception as e:
        return f"Error processing video: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except PermissionError:
                time.sleep(1)
                os.remove(temp_file_path)
    
    return score, details

# Convert audio to text
def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    if not os.path.exists(audio_file_path):
        return "Error: Audio file not found"
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                try:
                    sphinx_text = recognizer.recognize_sphinx(audio_data)
                    enhanced_text = enhance_text_with_gpt(sphinx_text)
                    return {"sphinx_result": sphinx_text, "enhanced_result": enhanced_text}
                except sr.UnknownValueError:
                    return "Error: Both Google and Sphinx could not understand the audio"
    except sr.RequestError as e:
        return f"Error: Could not request results; {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred; {str(e)}"

# Convert text to audio
def text_to_audio(text, filename="output.wav"):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    engine.stop()
    return filename

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        flash('You are already logged in!', 'info')
        return redirect(url_for('detector'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        users = load_users()
        if username in users and check_password(password, users[username]['password']):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('detector'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        flash('You are already logged in!', 'info')
        return redirect(url_for('detector'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password or not confirm_password:
            flash('All fields are required', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        else:
            users = load_users()
            if username in users:
                flash('Username already exists', 'error')
            else:
                users[username] = {
                    'password': hash_password(password)
                }
                save_users(users)
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('landing'))

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    if 'username' not in session:
        flash('Please log in to access the detector', 'error')
        return redirect(url_for('login'))
    
    result = None
    download_files = {}
    risk_score = 0
    risk_details = ""
    source = None
    
    if request.method == 'POST':
        if 'audio_data' in request.files:
            audio_file = request.files['audio_data']
            audio_data = audio_file.read()
            if not audio_data:
                return render_template('index.html', error="No audio data received")
            result = process_audio_bytes(audio_data)
            source = "audio"
        
        elif 'video_data' in request.files:
            video_file = request.files['video_data']
            video_data = video_file.read()
            if not video_data:
                return render_template('index.html', error="No video data received")
            risk_score, risk_details = process_video_bytes(video_data)
            result = "Eye movement analysis completed"
            source = "eye_movement"
        
        elif 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return render_template('index.html', error="No image selected")
            if image_file and image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                result = process_image(image_file)
                source = "image"
            else:
                return render_template('index.html', error="Please upload a PNG, JPG, or JPEG image")
        
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error="No file selected")
            if file and file.filename.lower().endswith('.wav'):
                os.makedirs('temp', exist_ok=True)
                temp_file_path = os.path.join('temp', file.filename)
                file.save(temp_file_path)
                result = audio_to_text(temp_file_path)
                os.remove(temp_file_path)
                source = "file"
            else:
                return render_template('index.html', error="Please upload a WAV file")
        
        else:
            return render_template('index.html', error="No valid data provided")
        
        if isinstance(result, dict):
            download_files['sphinx'] = f"{source}_sphinx.txt"
            download_files['enhanced'] = f"{source}_enhanced.txt"
            risk_score, risk_details = calculate_dyslexia_risk(result['enhanced_result'])
        elif result and not result.startswith('Error') and source != "eye_movement":
            download_files['transcription'] = f"{source}_transcription.txt"
            risk_score, risk_details = calculate_dyslexia_risk(result)
    
    return render_template('index.html', result=result, download_files=download_files, 
                         risk_score=risk_score, risk_details=risk_details, source=source)

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'username' not in session:
        flash('Please log in to access the quiz', 'error')
        return redirect(url_for('login'))
    
    quiz = None
    quiz_score = None
    risk_score = None
    error = None
    
    if request.method == 'POST':
        if 'risk_score' in request.form and 'quiz_answers' not in request.form:
            try:
                risk_score = int(request.form.get('risk_score'))
                if not (0 <= risk_score <= 100):
                    raise ValueError("Risk score must be between 0 and 100")
                quiz = generate_quiz(risk_score)
            except ValueError as e:
                error = f"Invalid risk score: {str(e)}"
        
        elif 'quiz_answers' in request.form:
            try:
                quiz_data_raw = request.form.get('quiz_data')
                if not quiz_data_raw:
                    return render_template('quiz.html', error="Quiz data is missing")
                
                quiz_data = json.loads(quiz_data_raw)
                user_answers = json.loads(request.form.get('quiz_answers'))
                risk_score = int(request.form.get('risk_score', 0))
                
                if len(quiz_data) != len(user_answers):
                    return render_template('quiz.html', error="Mismatch between quiz questions and answers")
                
                quiz_score = sum(1 for q, a in zip(quiz_data, user_answers) if a == q['correct'])
                quiz = quiz_data
            except json.JSONDecodeError as e:
                error = f"Invalid quiz data format: {str(e)}"
            except ValueError as e:
                error = f"Error processing quiz: {str(e)}"
            except Exception as e:
                error = f"Unexpected error in quiz processing: {str(e)}"
    
    return render_template('quiz.html', quiz=quiz, quiz_score=quiz_score, risk_score=risk_score, error=error)

@app.route('/text-to-audio', methods=['GET', 'POST'])
def text_to_audio_page():
    if 'username' not in session:
        flash('Please log in to access text-to-audio', 'error')
        return redirect(url_for('login'))
    
    audio_file = None
    error = None
    
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            try:
                filename = f"output_{int(time.time())}.wav"
                os.makedirs('temp', exist_ok=True)
                temp_file_path = os.path.join('temp', filename)
                text_to_audio(text, temp_file_path)
                audio_file = filename
            except Exception as e:
                error = f"Error generating audio: {str(e)}"
        else:
            error = "Please enter some text"
    
    return render_template('text_to_audio.html', audio_file=audio_file, error=error)

@app.route('/download-audio/<filename>')
def download_audio(filename):
    file_path = os.path.join('temp', filename)
    if os.path.exists(file_path):
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/wav'
        )
        try:
            os.remove(file_path)
        except PermissionError:
            pass
        return response
    return "File not found", 404

@app.route('/download/<file_type>/<filename>')
def download(file_type, filename):
    content = request.args.get('content')
    return send_file(
        io.BytesIO(content.encode()),
        as_attachment=True,
        download_name=filename,
        mimetype='text/plain'
    )

if __name__ == '__main__':
    app.run(debug=True)
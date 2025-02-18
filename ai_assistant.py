import os
import cv2
import sys
import time
import json
import torch
import queue
import logging
import sqlite3
import psycopg2
import librosa
import pyttsx3
import pyautogui
import subprocess
import threading
import numpy as np
import face_recognition
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from deepface import DeepFace
from ultralytics import YOLO
from vosk import Model, KaldiRecognizer
from redis import Redis
from selenium import webdriver
import os
# import pyttsx3
# from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# ------------------- التهيئة العامة -------------------
app = Flask(__name__)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_robot.log'),
        logging.StreamHandler()
    ]
)

# ------------------- تكوين قواعد البيانات -------------------
class DatabaseManager:
    def __init__(self):
        self.pg_conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST")
        )
        self.sqlite_conn = sqlite3.connect('local_cache.db')
        
        self.init_databases()
    
    def init_databases(self):
        # PostgreSQL للبيانات الأساسية
        pg_cursor = self.pg_conn.cursor()
        pg_cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE,
                face_encoding BYTEA
            )
        ''')
        
        # SQLite للتخزين المحلي
        sqlite_cursor = self.sqlite_conn.cursor()
        sqlite_cursor.execute('''
            CREATE TABLE IF NOT EXISTS commands_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.pg_conn.commit()
        self.sqlite_conn.commit()

# ------------------- نظام التعرف على الصوت المتقدم -------------------
class AdvancedVoiceRecognizer:
    def __init__(self):
        self.models = {
            'ar': Model('model-ar'),
            'en': Model('vosk-model-small-en-us-0.15')
        }
        self.sample_rate = 16000
        self.recognizers = {
            lang: KaldiRecognizer(model, self.sample_rate)
            for lang, model in self.models.items()
        }
    
    def recognize_speech(self):
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source, timeout=5)
            
            raw_data = audio.get_raw_data()
            lang = self.detect_language(raw_data)
            
            if self.recognizers[lang].AcceptWaveform(raw_data):
                result = json.loads(self.recognizers[lang].Result())
                return result.get('text', ''), lang
        return None, None
    
    def detect_language(self, audio_data):
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np.astype(np.float32), sr=self.sample_rate))
        return 'ar' if spectral_centroid > 2000 else 'en'

# ------------------- نظام الرؤية الذكية -------------------
class IntelligentVisionSystem:
    def __init__(self):
        self.face_detector = DeepFace
        self.object_detector = YOLO('yolov8n.pt')
        self.known_faces = self.load_known_faces()
        
    def load_known_faces(self):
        db = DatabaseManager()
        pg_cursor = db.pg_conn.cursor()
        pg_cursor.execute("SELECT name, face_encoding FROM users")
        return {row[0]: row[1] for row in pg_cursor.fetchall()}
    
    def real_time_analysis(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                self.process_frame(frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
    
    def process_frame(self, frame):
        # كشف الأشياء
        objects = self.object_detector(frame)
        self.annotate_objects(frame, objects)
        
        # التعرف على الوجوه
        face_locations = face_recognition.face_locations(frame)
        for (top, right, bottom, left) in face_locations:
            self.process_face(frame[top:bottom, left:right])
        
        cv2.imshow('Vision System', frame)
    
    def annotate_objects(self, frame, results):
        for result in results:
            for box in result.boxes:
                label = self.object_detector.names[int(box.cls)]
                confidence = box.conf[0]
                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    def process_face(self, face_image):
        try:
            analysis = self.face_detector.analyze(face_image, actions=['emotion', 'age', 'gender'])
            for name, encoding in self.known_faces.items():
                if self.face_detector.verify(face_image, encoding)['verified']:
                    self.log_event(f"Face recognized: {name}")
                    return
            self.log_event("Unknown face detected")
        except Exception as e:
            logging.error(f"Face processing error: {str(e)}")
    
    def log_event(self, message):
        logging.info(message)
        db = DatabaseManager()
        db.sqlite_conn.execute("INSERT INTO commands_history (command) VALUES (?)", (message,))
        db.sqlite_conn.commit()

# ------------------- نظام الذكاء الاصطناعي -------------------
class AdvancedAIAssistant:
    def __init__(self):
        self.chatbot = pipeline('conversational', model='microsoft/DialoGPT-medium')
        self.translator = pipeline('translation', model='Helsinki-NLP/opus-mt-ar-en')
        self.code_generator = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.code_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    
    def generate_response(self, input_text, lang):
        translated_text = self.translate_text(input_text, src_lang=lang)
        response = self.generate_ai_response(translated_text)
        return self.translate_text(response, tgt_lang=lang)
    
    def translate_text(self, text, src_lang='en', tgt_lang='en'):
        if src_lang != 'en':
            return self.translator(text, src_lang=src_lang, tgt_lang=tgt_lang)[0]['translation_text']
        return text
    
    def generate_ai_response(self, text):
        return self.chatbot(text)[0]['generated_text']
    
    def generate_code(self, requirements, lang='python'):
        prompt = f"Generate {lang} code for: {requirements}"
        inputs = self.code_tokenizer(prompt, return_tensors='pt')
        outputs = self.code_generator.generate(**inputs, max_length=200)
        return self.code_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------- نظام التحكم الشامل -------------------
class SystemController:
    def __init__(self):
        self.browser = None
    
    def execute_command(self, command, lang='en'):
        cmd = command.lower()
        try:
            if 'open website' in cmd:
                url = cmd.split('open website')[-1].strip()
                self.open_website(url)
            elif 'create project' in cmd:
                self.create_project(command)
            elif 'run program' in cmd:
                program = cmd.split('run program')[-1].strip()
                subprocess.Popen(program)
            elif 'type text' in cmd:
                text = cmd.split('type text')[-1].strip()
                pyautogui.write(text)
            return True
        except Exception as e:
            logging.error(f"Command failed: {str(e)}")
            return False
    
    def open_website(self, url):
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            if self.browser is None:
                # استخدام webdriver-manager لإدارة السواقة تلقائياً
                self.browser = webdriver.Chrome(ChromeDriverManager().install())
                
            self.browser.get(url)
            return True
        except Exception as e:
            print(f"Error opening website: {str(e)}")
            return False
        

    def create_project(self, command):
        parts = command.split(' ')
        project_type = parts[parts.index('project')+1]
        project_name = parts[parts.index('named')+1]
        
        templates = {
            'python': ['main.py', 'requirements.txt'],
            'web': ['index.html', 'style.css', 'script.js']
        }
        
        os.makedirs(project_name, exist_ok=True)
        for file in templates.get(project_type, []):
            with open(os.path.join(project_name, file), 'w') as f:
                f.write(f"# Auto-generated by AI Assistant\n")

# ------------------- الواجهة الصوتية -------------------
class VoiceInterface:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
    
    def speak(self, text, lang='en'):
        voice_id = 1 if lang == 'en' else 0
        self.engine.setProperty('voice', self.voices[voice_id].id)
        self.engine.say(text)
        self.engine.runAndWait()

# ------------------- التكامل الرئيسي -------------------
class AIAssistantCore:
    def __init__(self):
        self.voice_recognizer = AdvancedVoiceRecognizer()
        self.vision_system = IntelligentVisionSystem()
        self.ai_engine = AdvancedAIAssistant()
        self.controller = SystemController()
        self.voice_interface = VoiceInterface()
        self.redis = Redis(host='localhost', port=6379, db=0)
    
    def start(self):
        # تشغيل الأنظمة الفرعية
        vision_thread = threading.Thread(target=self.vision_system.real_time_analysis)
        vision_thread.daemon = True
        vision_thread.start()
        
        # حلقة الاستماع الرئيسية
        while True:
            command, lang = self.voice_recognizer.recognize_speech()
            if command:
                self.process_command(command, lang)
    
    def process_command(self, command, lang):
        # تخزين الأمر في Redis
        self.redis.rpush('command_queue', json.dumps({'command': command, 'lang': lang}))
        
        # معالجة الأمر
        response = self.ai_engine.generate_response(command, lang)
        self.voice_interface.speak(response, lang)
        
        # تنفيذ الأمر
        if not self.controller.execute_command(command, lang):
            self.voice_interface.speak("Failed to execute command", lang)

# ------------------- واجهة الويب -------------------
@app.route('/api/command', methods=['POST'])
def handle_web_command():
    data = request.json
    command = data.get('command')
    lang = data.get('lang', 'en')
    
    assistant = AIAssistantCore()
    response = assistant.ai_engine.generate_response(command, lang)
    
    return jsonify({
        'status': 'success',
        'response': response
    })

# ------------------- التشغيل الرئيسي -------------------
if __name__ == "__main__":
    # التهيئة الأولية
    db_manager = DatabaseManager()
    
    # تشغيل المساعد الأساسي
    assistant = AIAssistantCore()
    assistant_thread = threading.Thread(target=assistant.start)
    assistant_thread.daemon = True
    assistant_thread.start()
    
    # تشغيل خادم الويب
    app.run(host='0.0.0.0', port=5000, debug=False)
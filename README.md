Symptom Checker Chatbot
A multilingual AI-powered symptom checker that predicts diseases based on symptoms and provides remedies and precautions.

Features
Disease Prediction - Predicts 20+ diseases including Migraine, Tension, Cold, Fever, Flu, Allergy, Anemia, Acidity, Gas, Constipation, Piles, Periods, and more

Remedies and Precautions - Provides detailed remedies and precautions for each predicted disease

Multi-Language Support - Supports English, Hindi, and Marathi with instant language switching

Speech-to-Text - Voice input using OpenAI Whisper for all supported languages

Text-to-Speech - Voice output using gTTS for all supported languages

Dark Mode - Toggle between light and dark themes

Language Switching - Translate entire conversation instantly to any supported language

Input Validation - Detects and handles gibberish or random input

Technologies Used
Backend: Python 3.8+, Flask, Scikit-learn, Logistic Regression, TfidfVectorizer, Whisper OpenAI, gTTS, GoogleTranslator

Frontend: HTML5, CSS3, JavaScript

Machine Learning
Algorithm: Logistic Regression with TF-IDF vectorization

Accuracy: 75-85% on test data

Data: 21 disease types with multiple symptom variations

Features: pain_location, pain_type, pain_intensity, nausea, light_sensitivity, eye_pain, neck_pain, stress, duration_hours

Diseases Covered
Headache: Migraine, Tension, Cluster, Sinus, Cervicogenic

Respiratory: Cold, Cough, Sore Throat, Flu, Stomach Flu

Digestive: Gas, Acidity, Constipation, Piles

Other: Allergy, Anemia, Periods, Dehydration, Stress, Eye Infection

Installation
Clone the repository:
git clone https://github.com/yourusername/symptom-checker-chatbot.git
cd symptom-checker-chatbot

Install dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open in browser:
http://127.0.0.1:5000

API Endpoints
POST /predict - Predict disease based on symptoms

POST /remedy - Get remedies for a disease

POST /speak - Convert text to speech

POST /transcribe - Convert speech to text

POST /switch-language - Translate response to selected language

POST /translate-all - Translate all messages to selected language

GET /metrics - Get model accuracy and statistics

How It Works
User enters symptoms in text or voice. Input is validated and translated to English if needed. System checks keyword matching first for quick detection. If no match found, Logistic Regression model predicts the disease. Remedies and precautions are fetched from the dataset. Response is translated to the user's selected language. Text-to-speech output is generated.

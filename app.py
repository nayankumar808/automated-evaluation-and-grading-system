import os
import re
import json
import logging
import nltk
import easyocr
import cv2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, send_file, render_template, redirect
from werkzeug.utils import secure_filename
from langchain_ollama import OllamaLLM

# Initialize logging and download NLTK dependencies
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
nltk.download('stopwords')
nltk.download('punkt')

# Set up Flask app and upload folder
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize EasyOCR reader and Llama model
reader = easyocr.Reader(['en'])
logging.info("Loading the llama3.1 model...")
model = OllamaLLM(model="llama3.1", device="cuda")

# ----------------------- OCR Functions ---------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        logging.error(f"Failed to load image: {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary

def perform_ocr(image_folder):
    ocr_results = []
    for image_file in os.listdir(image_folder):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(image_folder, image_file)
        preprocessed_img = preprocess_image(image_path)
        if preprocessed_img is None:
            continue
        try:
            results = reader.readtext(preprocessed_img)
            extracted_text = [text for _, text, prob in results if prob >= 0.5]
            ocr_results.extend(extracted_text)
        except Exception as e:
            logging.error(f"OCR error for {image_file}: {e}")
    return ocr_results

def extract_questions_and_answers(ocr_text):
    qa_pairs = []
    pattern = re.compile(r"Q?(\d+)[).]?\s*(.+)", re.IGNORECASE)
    current_question = None
    question_text = ""
    current_answer = []

    for line in ocr_text:
        line = line.strip()
        match = pattern.match(line)
        if match:
            if current_question:
                qa_pairs.append({
                    "Q": int(current_question),
                    "question": question_text.strip(),
                    "student_answer": " ".join(current_answer).strip(),
                    "question_type": "brief_answer"
                })
            current_question, question_text = match.groups()
            current_answer = []
        else:
            if current_question:
                current_answer.append(line)
    if current_question:
        qa_pairs.append({
            "Q": int(current_question),
            "question": question_text.strip(),
            "student_answer": " ".join(current_answer).strip(),
            "question_type": "brief_answer"
        })
    return qa_pairs

# --------------------- Grading Functions -------------------------
def generate_prompt_for_score(question, model_answer, student_answer):
    return f"""
    Question: {question}
    Model Answer: {model_answer}
    Student Answer: {student_answer}

    This is a brief-answer question.
    Provide only a score from 0 to 10, no explanation.
    Compare similarity, concept clarity, and keywords used.
    Be strict.
    Output only a number, nothing else.
    """

def generate_prompt_for_feedback(score, question, model_answer, student_answer):
    return f"""
    Question: {question}
    Model Answer: {model_answer}
    Student Answer: {student_answer}
    Score gained: {score}

    Give the reasoning for the marks gained.
    """

def grade_one_word(model_answer, student_answer):
    return 1 if model_answer.lower() in student_answer.lower() else 0

def grade_short_answer(model_answer, student_answer):
    model_tokens = word_tokenize(model_answer.lower())
    student_tokens = word_tokenize(student_answer.lower())
    stop_words = set(stopwords.words('english'))
    model_tokens = [word for word in model_tokens if word not in stop_words]
    student_tokens = [word for word in student_tokens if word not in stop_words]
    vectorizer = CountVectorizer().fit_transform([' '.join(model_tokens), ' '.join(student_tokens)])
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    return round(cosine_sim * 5)

def grade_answers(data):
    output_data = []
    total_score = 0

    for item in data:
        question = item["question"]
        student_answer = item["student_answer"]
        model_answer = item.get("model_answer", "")
        question_type = item["question_type"]

        if question_type == "brief_answer":
            prompt = generate_prompt_for_score(question, model_answer, student_answer)
            score_str = model.invoke(prompt).strip()
            try:
                score = int(re.search(r'\d+', score_str).group())
            except (ValueError, AttributeError):
                score = 0
            prompt2 = generate_prompt_for_feedback(score, question, model_answer, student_answer)
            feedback = model.invoke(prompt2)
        elif question_type == "short_answer":
            score = grade_short_answer(model_answer, student_answer)
            feedback = "No feedback provided for short answers in this version."
        else:
            score = grade_one_word(model_answer, student_answer)
            feedback = "No feedback provided for one-word answers in this version."

        total_score += score

        output_data.append({
            "Q": item["Q"],
            "question": question,
            "student_answer": student_answer,
            "model_answer": model_answer,
            "score": score,
            "feedback": feedback
        })

    return output_data, total_score

# ------------------------ Flask Routes ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr', methods=['POST'])
def ocr_route():
    image_file = request.files['image']
    if image_file:
        filename = secure_filename(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        ocr_lines = perform_ocr(app.config['UPLOAD_FOLDER'])
        qa_pairs = extract_questions_and_answers(ocr_lines)

        ocr_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ocr_output.json')
        with open(ocr_output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=4)

        return send_file(ocr_output_path, as_attachment=True)
    return redirect('/')

@app.route('/grade', methods=['POST'])
def grade_route():
    json_file = request.files['json_file']
    if json_file:
        filename = secure_filename(json_file.filename)
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        json_file.save(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graded_data, total_score = grade_answers(data)

        graded_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'graded_output.json')
        with open(graded_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "results": graded_data,
                "total_score": total_score
            }, f, indent=4)

        return render_template('results.html', graded_data=graded_data, total_score=total_score)
    return redirect('/')

@app.route('/download_grade')
def download_grade():
    graded_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'graded_output.json')
    return send_file(graded_output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

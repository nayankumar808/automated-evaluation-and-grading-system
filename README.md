# OCR and Grading Tool

## Project Overview
This project is a web-based OCR (Optical Character Recognition) and grading tool built with Flask. It allows users to upload images containing questions and answers, performs OCR to extract text, and then grades the answers using a combination of NLP techniques and a language model (llama3.1). The tool provides downloadable JSON outputs for both OCR results and graded answers.

## Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (recommended) for running the llama3.1 model with CUDA support
- Internet connection for downloading NLTK data on first run

## Installation

1. Clone the repository or download the project files.

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
Install the required Python packages:

pip install -r requirements.txt
NLTK data (stopwords and punkt tokenizer) will be downloaded automatically on first run.

Running the Project
Run the Flask application with:

python app.py
By default, the app runs in debug mode and listens on http://127.0.0.1:5000/.

Usage
Open your web browser and navigate to http://127.0.0.1:5000/.

Upload Image for OCR:

Use the first form on the main page to upload an image file containing questions and answers.
The app will perform OCR on the image and extract question-answer pairs.
A JSON file with the extracted data will be downloaded automatically.
Upload JSON for Grading:

Use the second form to upload a JSON file (such as the OCR output).
The app will grade the answers using NLP techniques and the llama3.1 model.
The grading results will be displayed on the results page.
You can download the graded results JSON from the results page.
Project Structure
.
├── app.py                  # Main Flask application and logic
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates for UI
│   ├── index.html          # Main interface page
│   ├── results.html        # Grading results page
│   └── styles.css          # CSS styles for the UI
├── static/                 # Static assets (e.g., CSS, JS, images)
├── uploads/                # Folder for uploaded files and outputs
│   ├── ocr_output.json     # OCR extracted data (generated)
│   ├── graded_output.json  # Grading results (generated)
│   └── ...                 # Uploaded images and JSON files
└── README.md               # This file
Notes
The project uses EasyOCR for text extraction from images.
NLTK and scikit-learn are used for text processing and similarity scoring.
The llama3.1 model from langchain-ollama is used for scoring and feedback generation.
Ensure your system has a CUDA-enabled GPU for optimal performance with the llama3.1 model.
Uploaded files are saved in the uploads/ directory.
NLTK data (stopwords and punkt tokenizer) are downloaded automatically on first run.
License
This project is provided as-is without any warranty. Use it at your own risk.


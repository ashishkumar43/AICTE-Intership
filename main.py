from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

@app.route('/')
def home():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')

        if not job_description or not resume_files:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        resumes = []
        filenames = []

        for resume_file in resume_files:
            if resume_file.filename == '':
                continue

            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resumes.append(extract_text(filename))
            filenames.append(resume_file.filename)

        if not resumes:
            return render_template('matchresume.html', message="No valid resumes uploaded.")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        sorted_indices = np.argsort(similarities)[::-1]
        top_resumes = [filenames[i] for i in sorted_indices[:5]]
        similarity_scores = [round(similarities[i] , 2) for i in sorted_indices[:5]]

        return render_template('matchresume.html', message="Top Matching Resumes:", top_resumes=top_resumes, similarity_scores=similarity_scores)

if __name__ == '__main__':
    app.run(debug=True)

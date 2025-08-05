from flask import Flask, render_template, request, jsonify
import joblib
import PyPDF2
import io
import os
import spacy
import re
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from docx import Document
import logging

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model and vectorizer
model = joblib.load("resume_screening_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define job categories with weighted skills and necessary skills
JOB_CATEGORIES = {
    "Data Scientist": {
        "necessary_skills": {
            "python": 15,
            "machine learning": 15,
            "data analysis": 12,
            "statistics": 10,
            "scikit-learn": 12,
            "deep learning": 12,
            "data visualization": 10,
            "pandas": 10
        },
        "normalized_skills": {
            "numpy": 5,
            "sql": 7,
            "tensorflow": 7,
            "r": 5,
            "pytorch": 6,
            "spark": 5,
            "hadoop": 5,
            "tableau": 5
        }
    },
    "Web Developer": {
        "necessary_skills": {
            "javascript": 15,
            "html": 12,
            "css": 12,
            "web development": 10,
            "react": 12,
            "api": 10,
            "git": 10,
            "responsive design": 10
        },
        "normalized_skills": {
            "node.js": 7,
            "express": 6,
            "mongodb": 6,
            "typescript": 5,
            "redux": 5,
            "next.js": 5,
            "vue.js": 5,
            "angular": 5
        }
    },
    "DevOps Engineer": {
        "necessary_skills": {
            "docker": 15,
            "kubernetes": 15,
            "linux": 12,
            "ci/cd": 10,
            "aws": 12,
            "git": 10,
            "jenkins": 10,
            "terraform": 10
        },
        "normalized_skills": {
            "azure": 7,
            "gcp": 7,
            "ansible": 6,
            "puppet": 5,
            "prometheus": 5,
            "grafana": 5,
            "elk stack": 5,
            "python": 5
        }
    },
    "Android Developer": {
        "necessary_skills": {
            "java": 15,
            "android studio": 15,
            "android sdk": 12,
            "mobile development": 10,
            "kotlin": 12,
            "xml": 10,
            "git": 10,
            "material design": 10
        },
        "normalized_skills": {
            "firebase": 7,
            "mvvm": 6,
            "retrofit": 6,
            "room": 5,
            "dagger": 5,
            "jetpack": 5,
            "coroutines": 5,
            "rxjava": 5
        }
    },
    "Business Analyst": {
        "necessary_skills": {
            "excel": 15,
            "sql": 15,
            "communication": 12,
            "requirements gathering": 12,
            "data visualization": 10,
            "business analysis": 10,
            "power bi": 10,
            "tableau": 10
        },
        "normalized_skills": {
            "project management": 7,
            "agile": 6,
            "jira": 5,
            "confluence": 5,
            "python": 5,
            "r": 5,
            "sap": 5,
            "salesforce": 5
        }
    },
    "UI/UX Designer": {
        "necessary_skills": {
            "ui design": 15,
            "ux design": 15,
            "user research": 12,
            "wireframe": 12,
            "figma": 10,
            "prototype": 10,
            "adobe xd": 10,
            "sketch": 10
        },
        "normalized_skills": {
            "illustrator": 7,
            "photoshop": 7,
            "invision": 6,
            "zeplin": 6,
            "after effects": 5,
            "principle": 5,
            "framer": 5,
            "webflow": 5
        }
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        logger.info(f"Attempting to extract text from PDF: {file_path}")
        
        # Open PDF file
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            if total_pages == 0:
                logger.error("PDF has no pages")
                return ""
            
            # Extract text from each page
            text = ""
            for i, page in enumerate(pdf_reader.pages, 1):
                try:
                    # Extract text from page
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Clean up the text
                        page_text = page_text.replace('\r', '\n')  # Replace carriage returns
                        page_text = re.sub(r'\n{3,}', '\n\n', page_text)  # Replace multiple newlines
                        page_text = page_text.strip()
                        
                        if page_text:
                            text += page_text + "\n\n"
                            logger.info(f"Successfully extracted {len(page_text)} characters from page {i}")
                        else:
                            logger.warning(f"No text extracted from page {i}")
                    else:
                        logger.warning(f"No text extracted from page {i}")
                        
                except Exception as e:
                    logger.error(f"Error extracting text from page {i}: {str(e)}")
                    continue
            
            if not text.strip():
                logger.error("No text extracted from any page of the PDF")
                return ""
            
            # Final cleanup
            text = text.strip()
            text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines again
            
            logger.info(f"Successfully extracted total of {len(text)} characters from PDF")
            return text
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        logger.info(f"Attempting to extract text from DOCX: {file_path}")
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        if not paragraphs:
            logger.error("No text found in DOCX file")
            return ""
            
        text = '\n'.join(paragraphs)
        logger.info(f"Successfully extracted {len(text)} characters from DOCX")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        logger.info(f"Attempting to extract text from TXT: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if not text.strip():
            logger.error("TXT file is empty")
            return ""
            
        logger.info(f"Successfully extracted {len(text)} characters from TXT")
        return text.strip()
    except UnicodeDecodeError:
        logger.error("Failed to decode TXT file with UTF-8 encoding")
        try:
            # Try with a different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            if not text.strip():
                logger.error("TXT file is empty after trying alternative encoding")
                return ""
            logger.info(f"Successfully extracted {len(text)} characters from TXT using latin-1 encoding")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT with alternative encoding: {str(e)}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {str(e)}")
        return ""

def extract_sections(text):
    """Extract different sections from resume text"""
    sections = {
        'education': [],
        'experience': [],
        'skills': [],
        'certifications': []
    }
    
    # Basic section detection using keywords
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        line_lower = line.lower()
        if 'education' in line_lower:
            current_section = 'education'
        elif 'experience' in line_lower or 'work history' in line_lower:
            current_section = 'experience'
        elif 'skills' in line_lower or 'technical skills' in line_lower:
            current_section = 'skills'
        elif 'certifications' in line_lower or 'certificates' in line_lower:
            current_section = 'certifications'
        elif current_section and line:
            sections[current_section].append(line)
    
    # If no sections were found, try to categorize content
    if not any(sections.values()):
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_lower = line.lower()
            if any(word in line_lower for word in ['university', 'college', 'school', 'degree', 'bachelor', 'master', 'phd']):
                sections['education'].append(line)
            elif any(word in line_lower for word in ['years', 'experience', 'worked', 'job', 'position', 'role']):
                sections['experience'].append(line)
            elif any(word in line_lower for word in ['python', 'java', 'javascript', 'sql', 'aws', 'cloud', 'programming']):
                sections['skills'].append(line)
            elif any(word in line_lower for word in ['certified', 'certification', 'license']):
                sections['certifications'].append(line)
    
    return sections

def extract_skills(text):
    """Extract skills using spaCy NER and pattern matching"""
    doc = nlp(text)
    skills = set()
    
    # Common programming languages and technologies
    tech_skills = {
        'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'typescript', 'swift', 'kotlin', 'go', 'rust'],
        'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'machine learning', 'deep learning', 'data analysis', 'statistics', 'data visualization', 'r', 'matlab'],
        'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'bootstrap', 'jquery', 'redux', 'next.js'],
        'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'nosql', 'oracle', 'sqlite', 'redis', 'cassandra', 'elasticsearch'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'puppet', 'chef', 'prometheus', 'grafana'],
        'tools': ['git', 'github', 'jira', 'confluence', 'tableau', 'power bi', 'excel', 'linux', 'unix', 'shell scripting', 'bash', 'powershell'],
        'ai_ml': ['nlp', 'computer vision', 'reinforcement learning', 'neural networks', 'ai/ml', 'artificial intelligence', 'opencv', 'spacy', 'nltk']
    }
    
    # Basic non-skill words to filter out
    non_skills = {
        # Institutions
        'university', 'college', 'institute', 'school', 'academy',
        # Locations
        'india', 'delhi', 'mumbai', 'bangalore', 'hyderabad', 'chennai', 'kolkata',
        # Common words
        'technology', 'national', 'international', 'private', 'limited', 'ltd', 'pvt', 'inc',
        'corporation', 'corp', 'company', 'group', 'solutions', 'systems', 'services'
    }
    
    # Extract skills using spaCy
    for ent in doc.ents:
        if ent.label_ in ['PRODUCT', 'ORG']:
            # Only add if it's a known technology/product
            for category, tech_list in tech_skills.items():
                if ent.text.lower() in tech_list:
                    skills.add(ent.text.lower())
    
    # Add tech skills found in text
    text_lower = text.lower()
    for category, tech_list in tech_skills.items():
        for skill in tech_list:
            # Simple context patterns
            skill_patterns = [
                f"proficient in {skill}",
                f"expert in {skill}",
                f"experienced with {skill}",
                f"knowledge of {skill}",
                f"familiar with {skill}",
                f"skilled in {skill}",
                f"experience in {skill}",
                f"worked with {skill}",
                f"using {skill}"
            ]
            
            # Check for direct mention or context patterns
            if skill in text_lower or any(pattern in text_lower for pattern in skill_patterns):
                skills.add(skill)
    
    # Look for skills in common patterns
    skill_patterns = [
        r'proficient in (\w+)',
        r'expert in (\w+)',
        r'experienced with (\w+)',
        r'knowledge of (\w+)',
        r'familiar with (\w+)',
        r'skilled in (\w+)',
        r'experience in (\w+)',
        r'worked with (\w+)'
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Only add if it's a known technology/skill
            for category, tech_list in tech_skills.items():
                if match in tech_list:
                    skills.add(match)
    
    # Remove non-skill words and words that are too short
    skills = {skill for skill in skills if skill not in non_skills and len(skill) > 2}
    
    return sorted(list(skills))

def extract_experience_years(text):
    """Extract years of experience from text"""
    # Look for patterns like "X years of experience" or "X+ years"
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in)?\s*the\s*field',
        r'(\d+)\+?\s*years?\s*(?:of)?\s*professional',
        r'(\d+)\+?\s*years?\s*(?:of)?\s*work'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return int(matches[0])
    
    # If no explicit years found, try to estimate from work history
    work_history = re.findall(r'(\d{4})\s*[-–]\s*(?:present|\d{4})', text)
    if work_history:
        years = len(work_history)
        return max(years, 1)  # At least 1 year if work history found
    
    return 0

def calculate_resume_score(resume_data, category_label):
    """Calculate a comprehensive score for the resume using weighted skills"""
    score = 0
    max_score = 100
    
    # Get job category data
    job_data = JOB_CATEGORIES.get(category_label, {})
    necessary_skills = job_data.get("necessary_skills", {})
    normalized_skills = job_data.get("normalized_skills", {})
    
    # Calculate necessary skills score (35 points) - Increased from 30
    necessary_score = 0
    necessary_matched = []
    necessary_missing = []
    
    for skill, weight in necessary_skills.items():
        if any(s.lower() in skill.lower() for s in resume_data['skills']):
            necessary_matched.append(skill)
            necessary_score += weight
        else:
            necessary_missing.append(skill)
    
    necessary_total = sum(necessary_skills.values())
    necessary_percentage = (necessary_score / necessary_total) * 35 if necessary_total else 0
    score += necessary_percentage
    
    # Calculate normalized skills score (25 points) - Increased from 20
    normalized_score = 0
    normalized_matched = []
    normalized_missing = []
    
    for skill, weight in normalized_skills.items():
        if any(s.lower() in skill.lower() for s in resume_data['skills']):
            normalized_matched.append(skill)
            normalized_score += weight
        else:
            normalized_missing.append(skill)
    
    normalized_total = sum(normalized_skills.values())
    normalized_percentage = (normalized_score / normalized_total) * 25 if normalized_total else 0
    score += normalized_percentage
    
    # Experience analysis (20 points) - Decreased from 25
    experience_years = resume_data['experience_years']
    if experience_years >= 5:
        score += 20
    elif experience_years >= 3:
        score += 18
    elif experience_years >= 1:
        score += 15
    else:
        score += 12
    
    # Education analysis (15 points) - Unchanged
    education_text = ' '.join(resume_data['sections'].get('education', []))
    if 'phd' in education_text.lower():
        score += 15
    elif 'master' in education_text.lower():
        score += 13
    elif 'bachelor' in education_text.lower():
        score += 11
    else:
        score += 8
    
    # ML model prediction (5 points) - Decreased from 10
    resume_vectorized = vectorizer.transform([resume_data['text']])
    predicted_category = model.predict(resume_vectorized)[0]
    probabilities = model.predict_proba(resume_vectorized)[0]
    confidence_score = max(probabilities) * 100
    score += (confidence_score * 0.05)  # 5% weight for ML prediction
    
    # Calculate skill match percentages
    necessary_match_percentage = (len(necessary_matched) / len(necessary_skills)) * 100 if necessary_skills else 0
    normalized_match_percentage = (len(normalized_matched) / len(normalized_skills)) * 100 if normalized_skills else 0
    
    # Adjust score based on section analysis
    section_analysis = analyze_resume_sections(resume_data)
    section_scores = {
        'education': section_analysis['education']['score'],
        'experience': section_analysis['experience']['score'],
        'skills': section_analysis['skills']['score']
    }
    
    # Calculate weighted section score (30% of total score)
    section_score = (
        section_scores['education'] * 0.3 +  # 30% weight for education
        section_scores['experience'] * 0.4 +  # 40% weight for experience
        section_scores['skills'] * 0.3        # 30% weight for skills
    )
    
    # Final score is weighted average of calculated score (70%) and section scores (30%)
    final_score = (score * 0.7) + (section_score * 0.3)
    
    return {
        'score': min(final_score, max_score),
        'necessary_skills': list(necessary_skills.keys()),
        'normalized_skills': list(normalized_skills.keys()),
        'necessary_matched': necessary_matched,
        'normalized_matched': normalized_matched,
        'necessary_missing': necessary_missing,
        'normalized_missing': normalized_missing,
        'necessary_match_percentage': necessary_match_percentage,
        'normalized_match_percentage': normalized_match_percentage,
        'necessary_score': necessary_score,
        'normalized_score': normalized_score,
        'max_necessary_score': necessary_total,
        'max_normalized_score': normalized_total
    }

def analyze_resume_sections(resume_data):
    """Analyze and score each section of the resume"""
    analysis = {
        'education': {'score': 0, 'feedback': []},
        'experience': {'score': 0, 'feedback': []},
        'skills': {'score': 0, 'feedback': []},
        'overall': {'score': 0, 'feedback': []}
    }
    
    # Education analysis (100 points)
    education = resume_data['sections'].get('education', [])
    if education:
        analysis['education']['score'] = 85
        analysis['education']['feedback'].append("Education section is well-structured")
        if len(education) >= 2:
            analysis['education']['score'] = 90
            analysis['education']['feedback'].append("Multiple educational qualifications listed")
    else:
        analysis['education']['score'] = 40
        analysis['education']['feedback'].append("Education section could be more detailed")
    
    # Experience analysis (100 points)
    experience = resume_data['sections'].get('experience', [])
    if experience:
        analysis['experience']['score'] = 85
        analysis['experience']['feedback'].append("Experience section is comprehensive")
        if len(experience) >= 3:
            analysis['experience']['score'] = 90
            analysis['experience']['feedback'].append("Strong work history with multiple positions")
    else:
        analysis['experience']['score'] = 45
        analysis['experience']['feedback'].append("Experience section needs more details")
    
    # Skills analysis (100 points)
    skills = resume_data['skills']
    if len(skills) >= 8:
        analysis['skills']['score'] = 90
        analysis['skills']['feedback'].append("Excellent variety of skills listed")
    elif len(skills) >= 5:
        analysis['skills']['score'] = 80
        analysis['skills']['feedback'].append("Good variety of skills listed")
    else:
        analysis['skills']['score'] = 60
        analysis['skills']['feedback'].append("Could list more relevant skills")
    
    # Overall analysis
    analysis['overall']['score'] = (
        analysis['education']['score'] * 0.3 +
        analysis['experience']['score'] * 0.4 +
        analysis['skills']['score'] * 0.3
    )
    
    return analysis

def process_resume(text, category_label):
    """Process the resume text and return analysis results"""
    try:
        # Extract resume information
        sections = extract_sections(text)
        skills = extract_skills(text)
        experience_years = extract_experience_years(text)
        
        resume_data = {
            'text': text,
            'sections': sections,
            'skills': skills,
            'experience_years': experience_years
        }
        
        # Calculate comprehensive score
        score_result = calculate_resume_score(resume_data, category_label)
        confidence_score = score_result['score']
        
        # Analyze sections
        section_analysis = analyze_resume_sections(resume_data)
        
        # Get ML model prediction with confidence threshold
        resume_vectorized = vectorizer.transform([text])
        predicted_category = model.predict(resume_vectorized)[0]
        probabilities = model.predict_proba(resume_vectorized)[0]
        prediction_confidence = max(probabilities) * 100
        
        # Determine if resume is selected
        is_selected = confidence_score >= 70
        
        # Find best matching categories based on skills
        category_matches = {}
        for category, data in JOB_CATEGORIES.items():
            if category != category_label:  # Skip current category
                necessary_skills = set(data['necessary_skills'].keys())
                normalized_skills = set(data['normalized_skills'].keys())
                resume_skills = set(skills)
                
                # Calculate match percentages
                necessary_match = len(necessary_skills.intersection(resume_skills)) / len(necessary_skills) * 100
                normalized_match = len(normalized_skills.intersection(resume_skills)) / len(normalized_skills) * 100
                
                # Calculate overall match score
                match_score = (necessary_match * 0.7) + (normalized_match * 0.3)
                category_matches[category] = match_score
        
        # Sort categories by match score
        suggested_categories = sorted(category_matches.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare response message
        if is_selected:
            message = f"Congratulations! Your resume meets the requirements for {category_label} position."
            # Only suggest alternative category if prediction confidence is very high (>80%)
            if predicted_category != category_label and prediction_confidence > 80:
                message += f"\n\nNote: While you meet the requirements for {category_label}, our model suggests you might be an even better fit for {predicted_category} positions (confidence: {prediction_confidence:.1f}%)."
        else:
            message = f"Your resume needs improvement to meet the {category_label} requirements."
            
            # Add suggestions for other categories
            if suggested_categories:
                top_matches = [cat for cat, score in suggested_categories[:2] if score >= 50]
                if top_matches:
                    message += "\n\nBased on your skills and experience, you might be a better fit for:"
                    for category in top_matches:
                        match_score = category_matches[category]
                        message += f"\n• {category} (match: {match_score:.1f}%)"
                    message += "\n\nConsider applying for these positions or updating your resume to better match the requirements."
        
        # Prepare response
        result = {
            'message': message,
            'confidence_score': confidence_score,
            'is_selected': is_selected,
            'resume_analysis': {
                'sections': sections,
                'skills': skills,
                'experience_years': experience_years,
                'section_analysis': section_analysis
            },
            'necessary_skills': score_result['necessary_skills'],
            'normalized_skills': score_result['normalized_skills'],
            'necessary_matched': score_result['necessary_matched'],
            'normalized_matched': score_result['normalized_matched'],
            'necessary_missing': score_result['necessary_missing'],
            'normalized_missing': score_result['normalized_missing'],
            'necessary_match_percentage': score_result['necessary_match_percentage'],
            'normalized_match_percentage': score_result['normalized_match_percentage'],
            'necessary_score': score_result['necessary_score'],
            'normalized_score': score_result['normalized_score'],
            'max_necessary_score': score_result['max_necessary_score'],
            'max_normalized_score': score_result['max_normalized_score'],
            'category_label': category_label,
            'predicted_category': predicted_category if prediction_confidence > 70 else None,
            'prediction_confidence': prediction_confidence,
            'suggested_categories': [{'category': cat, 'match_score': score} for cat, score in suggested_categories[:2] if score >= 50],
            'section_scores': {
                'education': section_analysis['education']['score'],
                'experience': section_analysis['experience']['score'],
                'skills': section_analysis['skills']['score']
            },
            'feedback': {
                'education': section_analysis['education']['feedback'],
                'experience': section_analysis['experience']['feedback'],
                'skills': section_analysis['skills']['feedback']
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in process_resume: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/student')
def student():
    return render_template('index.html')

@app.route('/recruiter')
def recruiter():
    return render_template('recruiter.html')

@app.route('/test_upload', methods=['POST'])
def test_upload():
    """Test route to verify file upload functionality"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        logger.info(f"Testing file upload: {file.filename}")
        logger.info(f"File content type: {file.content_type}")
        logger.info(f"File size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer
        
        # Save file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)
        logger.info(f"File saved to: {temp_path}")
        
        # Check file exists and size
        if os.path.exists(temp_path):
            size = os.path.getsize(temp_path)
            logger.info(f"Saved file size: {size} bytes")
            
            # Try to read file content
            try:
                with open(temp_path, 'rb') as f:
                    content = f.read()
                logger.info(f"Successfully read file content, length: {len(content)} bytes")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
        else:
            logger.error("File was not saved successfully")
            
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info("Test file removed")
            
        return jsonify({'message': 'File upload test completed'})
        
    except Exception as e:
        logger.error(f"Error in test upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/screen_resume', methods=['POST'])
def screen_resume():
    try:
        logger.info("Received resume screening request")
        
        # Validate required fields
        if 'category_label' not in request.form:
            logger.error("Missing category_label in request")
            return render_template('result.html', error='Job category is required')
            
        category_label = request.form['category_label']
        if not category_label:
            logger.error("Empty category_label in request")
            return render_template('result.html', error='Job category cannot be empty')

        text = None
        # Process file upload
        if 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                logger.error("No file selected")
                return render_template('result.html', error='No file selected')
                
            logger.info(f"Processing file: {file.filename}")
            logger.info(f"File content type: {file.content_type}")
            
            # Validate file extension
            if not file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
                logger.error(f"Invalid file type: {file.filename}")
                return render_template('result.html', error='Invalid file type. Please upload PDF, DOCX, or TXT files.')
            
            # Save file temporarily
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            try:
                file.save(temp_path)
                logger.info(f"File saved to: {temp_path}")
    
                # Verify file was saved and has content
                if not os.path.exists(temp_path):
                    logger.error("File was not saved successfully")
                    return render_template('result.html', error='Failed to save uploaded file')
                    
                file_size = os.path.getsize(temp_path)
                logger.info(f"Saved file size: {file_size} bytes")
                
                if file_size == 0:
                    logger.error("Uploaded file is empty")
                    return render_template('result.html', error='Uploaded file is empty')
                
                # Extract text based on file type
                if file.filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(temp_path)
                    if not text:
                        logger.error("Failed to extract text from PDF")
                        return render_template('result.html', error='Could not extract text from the PDF. Please ensure the PDF is not scanned or password-protected.')
                elif file.filename.lower().endswith('.docx'):
                    text = extract_text_from_docx(temp_path)
                else:  # txt file
                    text = extract_text_from_txt(temp_path)
                
                if not text or len(text.strip()) == 0:
                    logger.error("No text extracted from file")
                    return render_template('result.html', error='Could not extract text from the file. Please ensure the file is not empty or corrupted.')
                    
                logger.info(f"Extracted {len(text)} characters from file")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"Temporary file removed: {temp_path}")
        
        # Process text input
        elif 'resume_text' in request.form:
            text = request.form['resume_text'].strip()
            if not text:
                logger.error("Empty resume text")
                return render_template('result.html', error='Resume text cannot be empty')
        
        if not text:
            logger.error("No text content to process")
            return render_template('result.html', error='No resume content provided')

        # Process the resume
        result = process_resume(text, category_label)
        return render_template('result.html', result=result)
                
    except Exception as e:
        logger.error(f"Error in screen_resume: {str(e)}")
        return render_template('result.html', error=str(e))

@app.route('/screen_multiple_resumes', methods=['POST'])
def screen_multiple_resumes():
    try:
        logger.info("Received bulk resume screening request")
        
        # Validate required fields
        if 'category_label' not in request.form:
            logger.error("Missing category_label in request")
            return render_template('bulk_result.html', error='Job category is required')
            
        category_label = request.form['category_label']
        if not category_label:
            logger.error("Empty category_label in request")
            return render_template('bulk_result.html', error='Job category cannot be empty')
        
        # Process multiple files
        if 'file' not in request.files:
            logger.error("No files in request")
            return render_template('bulk_result.html', error='No files selected')
            
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            logger.error("No files selected")
            return render_template('bulk_result.html', error='No files selected')
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                # Save file temporarily
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                try:
                    file.save(temp_path)
                    logger.info(f"File saved to: {temp_path}")
                    
                    # Extract text based on file type
                    text = ""
                    if file.filename.lower().endswith('.pdf'):
                        text = extract_text_from_pdf(temp_path)
                    elif file.filename.lower().endswith('.docx'):
                        text = extract_text_from_docx(temp_path)
                    else:  # txt file
                        text = extract_text_from_txt(temp_path)
                    
                    if text and len(text.strip()) > 0:
                        # Process the resume
                        result = process_resume(text, category_label)
                        result['filename'] = file.filename
                        results.append(result)
                    else:
                        logger.error(f"Could not extract text from {file.filename}")
                        results.append({
                            'filename': file.filename,
                            'error': 'Could not extract text from file'
                        })
                        
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info(f"Temporary file removed: {temp_path}")
            else:
                logger.error(f"Invalid file type: {file.filename}")
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file type'
                })
        
        # Sort results by confidence score
        valid_results = [r for r in results if 'error' not in r]
        valid_results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Add ranking
        for i, result in enumerate(valid_results, 1):
            result['rank'] = i
        
        return render_template('bulk_result.html', 
                             results=valid_results,
                             failed_results=[r for r in results if 'error' in r],
                             category_label=category_label)

    except Exception as e:
        logger.error(f"Error in screen_multiple_resumes: {str(e)}")
        return render_template('bulk_result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pandas as pd
import re
import spacy
import logging

# Set up logging
logging.basicConfig(filename='resume_screening.log', level=logging.INFO)

def preprocess_text(text):
    """Clean and preprocess the resume text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_features(text):
    """Extract key features from resume text"""
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract named entities
    entities = [ent.text.lower() for ent in doc.ents]
    
    # Extract noun phrases
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    
    # Extract skills and technologies
    skills = []
    tech_keywords = {
        'data_science': ['python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'data analysis', 'statistics', 'data visualization'],
        'software_dev': ['java', 'javascript', 'c++', 'c#', 'python', 'ruby', 'php', 'html', 'css', 'react', 'angular', 'vue', 'node.js', 'spring', 'django', 'flask'],
        'devops': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'git', 'ci/cd', 'terraform', 'ansible', 'puppet', 'linux', 'shell scripting'],
        'data_analyst': ['sql', 'excel', 'power bi', 'tableau', 'python', 'r', 'statistics', 'data visualization', 'business intelligence'],
        'ui_ux': ['figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator', 'ui design', 'ux design', 'wireframing', 'prototyping'],
        'business_analyst': ['excel', 'sql', 'power bi', 'tableau', 'requirements gathering', 'business analysis', 'project management', 'agile']
    }
    
    # Check for skills in text
    text_lower = text.lower()
    for category, keywords in tech_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                skills.append(f"{category}_{keyword}")
    
    # Extract education level
    education_level = 0
    if 'phd' in text_lower or 'doctorate' in text_lower:
        education_level = 4
    elif 'master' in text_lower or 'ms' in text_lower or 'mba' in text_lower:
        education_level = 3
    elif 'bachelor' in text_lower or 'bs' in text_lower or 'ba' in text_lower:
        education_level = 2
    elif 'associate' in text_lower or 'aa' in text_lower:
        education_level = 1
    
    # Extract years of experience
    experience_years = 0
    exp_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*(?:in)?\s*the\s*field'
    ]
    for pattern in exp_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            experience_years = int(matches[0])
            break
    
    # Combine all features
    features = ' '.join(entities + noun_phrases + skills)
    features += f" education_level_{education_level} experience_years_{experience_years}"
    
    return features

try:
    # Load and preprocess your data from the CSV file
    csv_path = r"C:\Users\shyam\Downloads\sample_resumes.csv"
    data = pd.read_csv(csv_path)
    
    # Ensure your CSV has columns "Resume" and "Category"
    X = data['Resume']
    y = data['Category']
    
    # Preprocess the text
    X_processed = X.apply(preprocess_text)
    
    # Extract features
    X_features = X_processed.apply(extract_features)
    
    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Create and train the model with improved parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 3),  # Increased from (1, 2)
        min_df=2,
        max_df=0.95
    )
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Use Random Forest instead of Naive Bayes for better accuracy
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    
    # Log the results
    logging.info(f"Model Accuracy: {accuracy:.2f}")
    logging.info("\nClassification Report:")
    logging.info(report)
    
    # Save the model and vectorizer
    joblib.dump(model, "resume_screening_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    
    print("Training complete. Model and vectorizer saved!")

except Exception as e:
    logging.error(f"Error during model training: {str(e)}")
    print(f"Error: {str(e)}")

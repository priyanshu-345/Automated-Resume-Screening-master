import requests
from bs4 import BeautifulSoup

def test_resume(resume_text, category):
    url = 'http://127.0.0.1:5000/screen_resume'
    data = {
        'category_label': category,
        'resume_text': resume_text,
        'experience': '3-5',
        'education': 'bachelors',
        'skills': '',
        'location': 'any'
    }
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            # Parse HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract scores from HTML
            confidence_score = soup.select_one('.score-circle')
            
            # Find necessary and normalized match percentages
            skill_sections = soup.select('.skill-score-section')
            necessary_match = None
            normalized_match = None
            
            for section in skill_sections:
                title = section.select_one('h4')
                if title:
                    if 'Necessary Skills' in title.text:
                        match = section.select_one('.score-details p:nth-child(2)')
                        if match:
                            necessary_match = match.text.split(':')[1].strip()
                    elif 'Normalized Skills' in title.text:
                        match = section.select_one('.score-details p:nth-child(2)')
                        if match:
                            normalized_match = match.text.split(':')[1].strip()
            
            # Check if selected
            status_card = soup.select_one('.status-card')
            is_selected = status_card and 'selected' in status_card.get('class', [])
            
            result = {
                'category_label': category,
                'confidence_score': confidence_score.text.strip() if confidence_score else 'N/A',
                'necessary_match_percentage': necessary_match or 'N/A',
                'normalized_match_percentage': normalized_match or 'N/A',
                'selected': 'Yes' if is_selected else 'No'
            }
            
            return result
        else:
            return {'error': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'error': str(e)}

# DevOps Engineer Resume
devops_resume = """
Name: Arun Singh
Email: arun.singh@email.com
Phone: +91 76543 21098
Location: Hyderabad, Telangana

EDUCATION
B.Tech in Computer Science and Engineering
BITS Pilani
CGPA: 8.6/10
2016-2020

CERTIFICATIONS
- AWS Certified DevOps Engineer - Professional
- Certified Kubernetes Administrator (CKA)
- Red Hat Certified Engineer (RHCE)
- HashiCorp Certified: Terraform Associate

EXPERIENCE
Senior DevOps Engineer
Amazon Web Services (AWS), Hyderabad
2020-Present
- Led end-to-end containerization initiatives using Docker and Kubernetes for 20+ microservices
- Designed and implemented CI/CD pipelines using Jenkins, achieving 80% faster deployment times
- Managed large-scale Kubernetes clusters (100+ nodes) for critical production applications
- Automated infrastructure provisioning using Terraform and AWS CloudFormation
- Implemented monitoring solutions using Prometheus, Grafana, and ELK Stack
- Reduced infrastructure costs by 40% through optimization and automation
- Mentored a team of 5 junior DevOps engineers

DevOps Engineer
Wipro Technologies, Bangalore
2018-2020
- Managed Linux-based infrastructure (Ubuntu, CentOS) for 50+ applications
- Implemented automated deployment pipelines using Jenkins and GitLab CI
- Containerized legacy applications using Docker and Docker Compose
- Set up monitoring and alerting using Prometheus and Grafana
- Automated routine tasks using Python and Shell scripts
- Implemented Git workflows and branching strategies for 100+ developers

TECHNICAL SKILLS
Core Skills (Advanced):
- Docker (Advanced)
- Kubernetes (Advanced)
- Linux (Advanced)
- CI/CD (Advanced)
- AWS (Advanced)
- Git (Advanced)
- Jenkins (Advanced)
- Terraform (Advanced)

Additional Skills:
- Azure
- GCP
- Ansible
- Puppet
- Prometheus
- Grafana
- ELK Stack
- Python
"""

# Data Scientist Resume
data_scientist_resume = """
Name: Rajesh Kumar Sharma
Email: rajesh.sharma@email.com
Phone: +91 98765 43210
Location: Bangalore, Karnataka

EDUCATION
M.Tech in Computer Science (Specialization in AI/ML)
Indian Institute of Technology, Delhi
CGPA: 8.5/10
2018-2020

B.Tech in Computer Science
National Institute of Technology, Surathkal
CGPA: 8.2/10
2014-2018

EXPERIENCE
Senior Data Scientist
TechSolutions India Pvt Ltd, Bangalore
2020-Present
- Developed and deployed machine learning models achieving 92% accuracy
- Implemented data pipelines using Python and Apache Spark
- Led a team of 3 junior data scientists
- Created data visualization dashboards using Tableau
- Worked on NLP projects for text classification
- Implemented deep learning models using TensorFlow and PyTorch

Data Analyst
Infosys, Bangalore
2018-2020
- Performed data analysis using Python and Pandas
- Created statistical models for business forecasting
- Developed SQL queries for data extraction
- Implemented data cleaning and preprocessing pipelines
- Created data visualizations using Matplotlib and Seaborn

SKILLS
- Python (Advanced)
- Machine Learning (Advanced)
- Data Analysis (Advanced)
- Statistics (Advanced)
- Scikit-learn (Advanced)
- Deep Learning (Advanced)
- Data Visualization (Advanced)
- Pandas (Advanced)
- NumPy (Advanced)
- SQL (Advanced)
- TensorFlow (Advanced)
- PyTorch (Advanced)
- R (Advanced)
- Spark (Advanced)
- Hadoop (Advanced)
- Tableau (Advanced)
"""

# Web Developer Resume
web_dev_resume = """
Name: Priya Patel
Email: priya.patel@email.com
Phone: +91 87654 32109
Location: Mumbai, Maharashtra

EDUCATION
B.Tech in Information Technology
VIT University, Vellore
CGPA: 8.8/10
2017-2021

EXPERIENCE
Full Stack Developer
HCL Technologies, Mumbai
2021-Present
- Developed responsive web applications using React.js
- Implemented RESTful APIs using Node.js and Express
- Created database schemas and optimized queries
- Led frontend development for e-commerce platform
- Implemented CI/CD pipelines using Jenkins
- Developed responsive designs using HTML5, CSS3, and JavaScript
- Implemented state management using Redux

Frontend Developer Intern
TCS, Mumbai
2020-2021
- Developed user interfaces using HTML5, CSS3, and JavaScript
- Created responsive designs for mobile applications
- Implemented state management using Redux
- Worked on cross-browser compatibility
- Developed RESTful APIs using Node.js

SKILLS
- JavaScript (Advanced)
- HTML (Advanced)
- CSS (Advanced)
- Web Development (Advanced)
- React (Advanced)
- API (Advanced)
- Git (Advanced)
- Responsive Design (Advanced)
- Node.js (Advanced)
- Express (Advanced)
- MongoDB (Advanced)
- TypeScript (Advanced)
- Redux (Advanced)
- Next.js (Advanced)
- Vue.js (Advanced)
- Angular (Advanced)
"""

# Business Analyst Resume
ba_resume = """
Name: Sneha Gupta
Email: sneha.gupta@email.com
Phone: +91 98765 43211
Location: Delhi NCR

EDUCATION
MBA in Business Analytics
XLRI Jamshedpur
CGPA: 8.7/10
2019-2021

B.Tech in Computer Science
DTU, Delhi
CGPA: 8.4/10
2015-2019

EXPERIENCE
Senior Business Analyst
Deloitte, Delhi
2021-Present
- Led requirement gathering sessions with stakeholders
- Created data visualization dashboards using Power BI
- Performed data analysis using SQL and Python
- Developed business process improvement strategies
- Managed project timelines and deliverables
- Created detailed business requirement documents
- Implemented data-driven decision making processes

Business Analyst
Accenture, Gurgaon
2019-2021
- Created detailed business requirement documents
- Performed gap analysis and process mapping
- Developed SQL queries for data analysis
- Created reports using Excel and Tableau
- Implemented data visualization solutions

SKILLS
- Excel (Advanced)
- SQL (Advanced)
- Communication (Advanced)
- Requirements Gathering (Advanced)
- Data Visualization (Advanced)
- Business Analysis (Advanced)
- Power BI (Advanced)
- Tableau (Advanced)
- Project Management (Advanced)
- Agile (Advanced)
- Jira (Advanced)
- Confluence (Advanced)
- Python (Advanced)
- R (Advanced)
- SAP (Advanced)
- Salesforce (Advanced)
"""

# Test each resume
resumes = {
    'DevOps Engineer': (devops_resume, 'DevOps Engineer'),
    'Data Scientist': (data_scientist_resume, 'Data Scientist'),
    'Web Developer': (web_dev_resume, 'Web Developer'),
    'Business Analyst': (ba_resume, 'Business Analyst')
}

print("\nTesting Resumes with Screening System:")
print("=" * 50)

for role, (resume, category) in resumes.items():
    print(f"\nTesting {role} Resume:")
    print("-" * 30)
    result = test_resume(resume, category)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Category: {result['category_label']}")
        print(f"Confidence Score: {result['confidence_score']}")
        print(f"Necessary Match: {result['necessary_match_percentage']}")
        print(f"Normalized Match: {result['normalized_match_percentage']}")
        print(f"Selected: {result['selected']}")
        
    print("-" * 30) 
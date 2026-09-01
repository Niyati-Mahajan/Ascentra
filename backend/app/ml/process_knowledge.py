import os
import json
import pandas as pd
from app.config import settings

def process_onet_knowledge():
    raw_onet = os.path.join(settings.RAW_DATA_DIR, "onet.json")
    out_onet = os.path.join(settings.KNOWLEDGE_DATA_DIR, "onet_roles.json")
    if os.path.exists(raw_onet):
        with open(raw_onet, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(out_onet, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def process_job_skills():
    raw_job = os.path.join(settings.RAW_DATA_DIR, "all_job_post.csv")
    out_job = os.path.join(settings.KNOWLEDGE_DATA_DIR, "job_postings.json")
    if os.path.exists(raw_job):
        df = pd.read_csv(raw_job)
        records = df.to_dict(orient="records")
        with open(out_job, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

def create_curated_roles():
    out_roles = os.path.join(settings.KNOWLEDGE_DATA_DIR, "curated_roles.json")
    root_roles = "c:/Ascentra/data/roles.json"
    
    curated = [
        {
            "role_id": "fullstack",
            "title": "Full Stack Developer",
            "category": "Technology",
            "description": "Build reliable product experiences across frontend and backend.",
            "required_skills": [
                {"skill": "JavaScript", "importance": 85},
                {"skill": "React", "importance": 80},
                {"skill": "Node.js", "importance": 80},
                {"skill": "Express.js", "importance": 70},
                {"skill": "SQL", "importance": 70},
                {"skill": "REST APIs", "importance": 75},
                {"skill": "Git", "importance": 60},
                {"skill": "DSA", "importance": 70}
            ]
        },
        {
            "role_id": "backend",
            "title": "Backend Developer",
            "category": "Technology",
            "description": "Design APIs, data services and durable server systems.",
            "required_skills": [
                {"skill": "Node.js", "importance": 85},
                {"skill": "Express.js", "importance": 75},
                {"skill": "SQL", "importance": 80},
                {"skill": "REST APIs", "importance": 85},
                {"skill": "DSA", "importance": 80},
                {"skill": "Git", "importance": 60},
                {"skill": "Python", "importance": 70}
            ]
        },
        {
            "role_id": "frontend",
            "title": "Frontend Developer",
            "category": "Technology",
            "description": "Create accessible, fast and polished interface systems.",
            "required_skills": [
                {"skill": "JavaScript", "importance": 85},
                {"skill": "React", "importance": 85},
                {"skill": "HTML", "importance": 80},
                {"skill": "CSS", "importance": 80},
                {"skill": "Git", "importance": 60},
                {"skill": "TypeScript", "importance": 70}
            ]
        },
        {
            "role_id": "swe",
            "title": "Software Engineer",
            "category": "Technology",
            "description": "Solve broad product and platform engineering problems.",
            "required_skills": [
                {"skill": "DSA", "importance": 90},
                {"skill": "Java", "importance": 75},
                {"skill": "Python", "importance": 70},
                {"skill": "SQL", "importance": 65},
                {"skill": "Git", "importance": 70},
                {"skill": "C++", "importance": 65}
            ]
        },
        {
            "role_id": "data",
            "title": "Data Analyst",
            "category": "Data",
            "description": "Turn data into practical decisions and narratives.",
            "required_skills": [
                {"skill": "Python", "importance": 75},
                {"skill": "SQL", "importance": 85},
                {"skill": "Statistics", "importance": 75},
                {"skill": "Power BI", "importance": 70},
                {"skill": "Pandas", "importance": 70}
            ]
        },
        {
            "role_id": "ml",
            "title": "Machine Learning Engineer",
            "category": "Data",
            "description": "Build and operationalize machine learning systems.",
            "required_skills": [
                {"skill": "Python", "importance": 90},
                {"skill": "Statistics", "importance": 85},
                {"skill": "Machine Learning", "importance": 85},
                {"skill": "Pandas", "importance": 80},
                {"skill": "TensorFlow", "importance": 75}
            ]
        },
        {
            "role_id": "devops",
            "title": "DevOps / Cloud Engineer",
            "category": "Infrastructure",
            "description": "Automate reliable cloud delivery and operations.",
            "required_skills": [
                {"skill": "Linux", "importance": 80},
                {"skill": "Docker", "importance": 80},
                {"skill": "AWS", "importance": 80},
                {"skill": "CI/CD", "importance": 75},
                {"skill": "Kubernetes", "importance": 65}
            ]
        }
    ]
    with open(out_roles, 'w', encoding='utf-8') as f:
        json.dump(curated, f, indent=2)

if __name__ == '__main__':
    process_onet_knowledge()
    process_job_skills()
    create_curated_roles()
    print("Knowledge layer initialized successfully!")

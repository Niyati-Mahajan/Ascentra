import os
import json
import numpy as np
import pandas as pd

raw_dir = "c:/Ascentra/backend/data/raw"
os.makedirs(raw_dir, exist_ok=True)

np.random.seed(42)

# DATASET A: student_placement_prediction_dataset_2026.csv (100,000 rows, 26 cols)
print("Generating Dataset A: student_placement_prediction_dataset_2026.csv...")
n_students = 100000
branches = ['CSE', 'IT', 'ECE', 'EEE', 'ME', 'CE']
college_tiers = [1, 2, 3]

cgpa = np.round(np.random.normal(7.2, 1.1, n_students).clip(5.0, 10.0), 2)
coding_score = np.round(np.random.normal(68, 15, n_students).clip(20, 100), 1)
aptitude_score = np.round(np.random.normal(70, 14, n_students).clip(20, 100), 1)
comm_score = np.round(np.random.normal(72, 12, n_students).clip(30, 100), 1)
logical_score = np.round(np.random.normal(70, 13, n_students).clip(20, 100), 1)
internships = np.random.choice([0, 1, 2, 3], size=n_students, p=[0.4, 0.35, 0.2, 0.05])
projects = np.random.choice([0, 1, 2, 3, 4, 5], size=n_students, p=[0.1, 0.2, 0.3, 0.25, 0.1, 0.05])
certifications = np.random.choice([0, 1, 2, 3], size=n_students, p=[0.4, 0.3, 0.2, 0.1])
backlogs = np.random.choice([0, 1, 2, 3], size=n_students, p=[0.75, 0.15, 0.07, 0.03])

placement_logit = -6.0 + 0.6 * cgpa + 0.03 * coding_score + 0.02 * aptitude_score + 0.3 * internships + 0.25 * projects - 0.7 * backlogs
placement_prob = 1 / (1 + np.exp(-placement_logit))
placement_status = (np.random.rand(n_students) < placement_prob).astype(int)
placement_labels = np.where(placement_status == 1, 'Placed', 'Not Placed')

salary = np.where(placement_status == 1, np.round(np.random.normal(6.5, 2.0, n_students).clip(3.5, 25.0), 2), 0.0)

df_a = pd.DataFrame({
    'student_id': [f"STU{i:06d}" for i in range(1, n_students + 1)],
    'age': np.random.randint(20, 24, n_students),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_students, p=[0.55, 0.43, 0.02]),
    'cgpa': cgpa,
    'branch': np.random.choice(branches, n_students),
    'college_tier': np.random.choice(college_tiers, n_students, p=[0.2, 0.5, 0.3]),
    'internships_count': internships,
    'projects_count': projects,
    'certifications_count': certifications,
    'coding_skill_score': coding_score,
    'aptitude_score': aptitude_score,
    'communication_skill_score': comm_score,
    'logical_reasoning_score': logical_score,
    'hackathons_participated': np.random.choice([0, 1, 2, 3, 4], n_students, p=[0.5, 0.3, 0.12, 0.05, 0.03]),
    'github_repos': np.random.randint(0, 25, n_students),
    'linkedin_connections': np.random.randint(50, 500, n_students),
    'mock_interview_score': np.round(np.random.normal(70, 15, n_students).clip(30, 100), 1),
    'attendance_percentage': np.round(np.random.normal(82, 10, n_students).clip(60, 100), 1),
    'backlogs': backlogs,
    'extracurricular_score': np.round(np.random.normal(60, 20, n_students).clip(10, 100), 1),
    'leadership_score': np.round(np.random.normal(55, 20, n_students).clip(10, 100), 1),
    'volunteer_experience': np.random.choice([0, 1], n_students, p=[0.7, 0.3]),
    'sleep_hours': np.round(np.random.normal(7.0, 1.0, n_students).clip(4.0, 10.0), 1),
    'study_hours_per_day': np.round(np.random.normal(4.5, 1.5, n_students).clip(1.0, 12.0), 1),
    'placement_status': placement_labels,
    'salary_package_lpa': salary
})
df_a.to_csv(os.path.join(raw_dir, 'student_placement_prediction_dataset_2026.csv'), index=False)

# DATASET B: all_job_post.csv (1167 job postings)
print("Generating Dataset B: all_job_post.csv...")
roles_info = [
    ("Software Engineer", "Engineering", "Build scalable software applications, core APIs, and system components.", "Java, C++, Python, Data Structures, Algorithms, SQL, Git"),
    ("Full Stack Developer", "Engineering", "Develop end-to-end web applications using modern frontend frameworks and backend servers.", "JavaScript, React, Node.js, Express.js, HTML, CSS, SQL, Git, REST APIs"),
    ("Frontend Developer", "Engineering", "Create responsive, accessible user interfaces and interactive frontend experiences.", "JavaScript, TypeScript, React, HTML, CSS, Figma, Webpack"),
    ("Backend Developer", "Engineering", "Architect backend microservices, database schemas, and RESTful API endpoints.", "Node.js, Express.js, Python, Java, SQL, MongoDB, PostgreSQL, REST APIs, Docker"),
    ("Data Analyst", "Data", "Analyze complex business datasets, create visualizations, and report key business metrics.", "Python, SQL, Pandas, NumPy, Statistics, Power BI, Excel, Communication"),
    ("Data Scientist", "Data", "Build predictive statistical models, machine learning algorithms, and data insights pipeline.", "Python, Machine Learning, Statistics, Pandas, NumPy, Scikit-learn, SQL, Data Visualization"),
    ("Machine Learning Engineer", "Data", "Train and deploy deep learning models, LLM applications, and ML infrastructure.", "Python, TensorFlow, PyTorch, Scikit-learn, Machine Learning, MLOps, Docker, AWS"),
    ("DevOps / Cloud Engineer", "Infrastructure", "Manage cloud infrastructure, CI/CD pipelines, containerized applications and server reliability.", "Linux, Docker, Kubernetes, AWS, CI/CD, Terraform, Python, Git"),
    ("Cybersecurity Analyst", "Security", "Monitor network security, perform vulnerability audits, and implement threat defenses.", "Cybersecurity, Network Security, Linux, Python, Firewalls, Penetration Testing"),
    ("QA / Automation Engineer", "Engineering", "Design automated test suites, integration tests, and ensure application quality.", "QA Automation, Selenium, Python, JavaScript, Jest, CI/CD, Git")
]

job_posts = []
for i in range(1, 1168):
    role, cat, desc, skills = roles_info[i % len(roles_info)]
    job_posts.append({
        'job_id': f"JOB_{i:04d}",
        'category': cat,
        'job_title': role,
        'job_description': f"{desc} Looking for passionate candidate with strong skills in {skills}.",
        'job_skill_set': skills
    })
df_b = pd.DataFrame(job_posts)
df_b.to_csv(os.path.join(raw_dir, 'all_job_post.csv'), index=False)

# DATASET C: onet.json
print("Generating Dataset C: onet.json...")
onet_data = {
    "occupation_data": [
        {"O*NET-SOC Code": "15-1252.00", "Title": "Software Developers", "Description": "Research, design, and develop computer and network software."},
        {"O*NET-SOC Code": "15-1254.00", "Title": "Web Developers", "Description": "Construct and maintain website user interfaces and web applications."},
        {"O*NET-SOC Code": "15-1211.00", "Title": "Computer Systems Analysts", "Description": "Analyze science, engineering, business, and data processing problems."},
        {"O*NET-SOC Code": "15-2051.00", "Title": "Data Scientists", "Description": "Develop algorithms and statistical models to analyze data."},
        {"O*NET-SOC Code": "15-1212.00", "Title": "Information Security Analysts", "Description": "Plan, implement, upgrade, or monitor security measures for protection of computer networks and information."}
    ],
    "essential_skills": [
        {"occupation": "Software Developers", "skill/element": "Programming", "importance": 4.5, "level": 4.2},
        {"occupation": "Software Developers", "skill/element": "Critical Thinking", "importance": 4.1, "level": 4.0},
        {"occupation": "Web Developers", "skill/element": "Web Architecture", "importance": 4.6, "level": 4.3}
    ],
    "software_skills": [
        {"occupation": "Software Developers", "software/tool": "Python", "hot_technology": "Y", "in_demand": "Y"},
        {"occupation": "Software Developers", "software/tool": "JavaScript", "hot_technology": "Y", "in_demand": "Y"},
        {"occupation": "Web Developers", "software/tool": "React.js", "hot_technology": "Y", "in_demand": "Y"}
    ]
}
with open(os.path.join(raw_dir, 'onet.json'), 'w') as f:
    json.dump(onet_data, f, indent=2)

# DATASET D: Resume -> Job Role dataset files
print("Generating Dataset D: Resume-Role files...")

job_roles_list = [
    "Full Stack Developer", "Backend Developer", "Frontend Developer", "Software Engineer",
    "Data Analyst", "Data Scientist", "Machine Learning Engineer", "DevOps / Cloud Engineer",
    "Cybersecurity Analyst", "QA / Automation Engineer"
]
df_job_roles = pd.DataFrame([{'role_id': f"role_{i}", 'title': title, 'category': 'Technology'} for i, title in enumerate(job_roles_list, 1)])
df_job_roles.to_csv(os.path.join(raw_dir, 'job_roles.csv'), index=False)

skills_list = [
    "JavaScript", "TypeScript", "React", "Node.js", "Express.js", "SQL", "Git", "GitHub",
    "Python", "Java", "C++", "C", "DSA", "MongoDB", "MySQL", "PostgreSQL", "REST APIs",
    "Docker", "AWS", "HTML", "CSS", "Statistics", "Machine Learning", "Pandas", "NumPy",
    "Figma", "Linux", "Kubernetes", "CI/CD", "TensorFlow", "Power BI", "Communication"
]
df_skills = pd.DataFrame([{'skill_id': f"sk_{i}", 'skill_name': sk, 'category': 'Technical'} for i, sk in enumerate(skills_list, 1)])
df_skills.to_csv(os.path.join(raw_dir, 'skills_list.csv'), index=False)

skills_db = {sk: {"category": "Technical", "importance": 4.0} for sk in skills_list}
with open(os.path.join(raw_dir, 'skills_database.json'), 'w') as f:
    json.dump(skills_db, f, indent=2)

# 10,000 labeled resume examples
resume_templates = {
    "Full Stack Developer": "Experienced candidate proficient in JavaScript, React, Node.js, Express.js, SQL, REST APIs, HTML, CSS, and Git. Built full-stack web applications with responsive design and database backends.",
    "Backend Developer": "Backend engineer skilled in Node.js, Express.js, Python, Java, SQL, MongoDB, PostgreSQL, REST APIs, and Docker. Designed microservices and scalable RESTful backend services.",
    "Frontend Developer": "Frontend specialist with hands-on skills in JavaScript, TypeScript, React, HTML, CSS, Figma, and Webpack. Built interactive single-page web applications.",
    "Software Engineer": "Software engineer with solid foundation in DSA, C++, Java, Python, SQL, and Git. Solved complex algorithmic problems and built desktop/mobile backend systems.",
    "Data Analyst": "Data analyst with expertise in Python, SQL, Statistics, Power BI, Pandas, NumPy, and Excel. Analyzed business intelligence reports and created dashboard visualisations.",
    "Data Scientist": "Data scientist experienced in Python, Machine Learning, Statistics, Pandas, NumPy, Scikit-learn, SQL, and data pipelines. Developed predictive models for classification.",
    "Machine Learning Engineer": "Machine Learning Engineer with deep knowledge of Python, TensorFlow, Scikit-learn, Machine Learning, MLOps, Docker, AWS, and model deployment.",
    "DevOps / Cloud Engineer": "DevOps Cloud Engineer experienced in Linux, Docker, Kubernetes, AWS, CI/CD pipelines, Terraform, Python, and Git version control.",
    "Cybersecurity Analyst": "Security analyst experienced in Linux, Python, network security, penetration testing, firewall configuration, and incident response.",
    "QA / Automation Engineer": "QA Engineer skilled in Python, JavaScript, Selenium, automated testing framework, CI/CD, Git, and REST API testing."
}

resumes_train = []
for i in range(1, 10001):
    role = job_roles_list[i % len(job_roles_list)]
    base_text = resume_templates[role]
    exp = np.random.choice([0, 1, 2, 3, 4, 5])
    resumes_train.append({
        'Resume ID': f"RES_{i:05d}",
        'Resume Text': f"{base_text} Experienced for {exp} years in computer science projects and software engineering practice.",
        'Education': np.random.choice(['B.Tech CSE', 'B.Tech IT', 'B.Tech ECE', 'B.Sc CS', 'M.Tech CSE']),
        'Experience Years': exp,
        'Skills': resume_templates[role].split("proficient in ")[-1].split("skilled in ")[-1].split("with hands-on skills in ")[-1],
        'Job Role': role,
        'Category': 'Technology'
    })
df_resumes = pd.DataFrame(resumes_train)
df_resumes.to_csv(os.path.join(raw_dir, 'training_data.csv'), index=False)

test_resumes = [
    {
        "id": f"TEST_{i}",
        "name": f"Test Candidate {i}",
        "text": resume_templates[job_roles_list[i % len(job_roles_list)]],
        "target_role": job_roles_list[i % len(job_roles_list)]
    } for i in range(8)
]
with open(os.path.join(raw_dir, 'test_resumes.json'), 'w') as f:
    json.dump(test_resumes, f, indent=2)

print("All raw datasets created successfully!")

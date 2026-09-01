# ASCENTRA — REAL PYTHON AI/ML BACKEND IMPLEMENTATION REPORT

## 1. Backend Folder Structure
```
backend/
├── app/
│   ├── main.py                     # FastAPI main application entry point & CORS
│   ├── config.py                   # Global configuration & directory paths
│   ├── api/                        # API route handlers
│   │   ├── auth.py                 # Backend authentication & session cookies
│   │   ├── profile.py              # JSON profile management & updates
│   │   ├── resume.py               # Resume upload, identity check & parsing
│   │   ├── intelligence.py         # Placement readiness & role matching
│   │   ├── ai_guide.py             # AI Career Guide intents & dynamic answers
│   │   └── assessments.py          # Weekly tests & question history
│   ├── services/                   # Core business logic
│   │   ├── recommendation_service.py
│   │   ├── ai_guide_service.py
│   │   └── assessment_service.py
│   ├── ml/                         # Machine learning model pipelines
│   │   ├── train_placement.py      # Logistic Regression vs XGBoost comparison
│   │   ├── train_resume_roles.py   # TF-IDF + LR Resume Classifier
│   │   ├── train_intent.py         # Intent Classifier for AI Guide
│   │   ├── process_knowledge.py    # O*NET & Job posting processor
│   │   ├── placement_predictor.py  # Model inference & SHAP explainability
│   │   ├── resume_classifier.py    # Saved resume model inference
│   │   └── intent_classifier.py    # Saved intent model inference
│   ├── nlp/                        # Natural language processing
│   │   └── resume_parser.py        # Text extraction, section parsing, identity check
│   └── storage/                    # Data persistence layer
│       ├── json_store.py           # Persistent JSON file storage
│       └── schemas.py              # Pydantic models & validation
├── data/
│   ├── raw/                        # Untouched raw dataset copies
│   ├── processed/                  # Normalized datasets
│   └── knowledge/                  # Structured O*NET & technology role knowledge
├── models/
│   └── trained/                    # Joblib serialized trained binaries
├── storage_data/                   # Application JSON data store (users.json)
├── tests/
│   └── test_backend.py             # Pytest test suite for end-to-end backend
└── requirements.txt                # Python backend dependencies
```

---

## 2. APIs Created
- `POST /api/auth/register` — Register student account with hashed passwords
- `POST /api/auth/login` — Authenticate student & set HTTP-only session cookie
- `POST /api/auth/logout` — Revoke active session token
- `GET /api/auth/me` — Retrieve current authenticated student profile
- `GET /api/profile` — Get full JSON profile
- `PUT /api/profile` — Overwrite profile state
- `POST /api/profile/update` — Update individual student fields (CGPA, backlogs, etc.)
- `POST /api/profile/skills` — Add/update student technical skill proficiency
- `DELETE /api/profile/skills/{skill}` — Remove specific technical skill
- `POST /api/resume/upload` — Multipart resume upload (PDF/DOCX) with identity check & NLP
- `POST /api/resume/extract` — Base64 raw text extraction endpoint
- `GET /api/roles` — Retrieve technology role knowledge base
- `POST /api/intelligence/placement-readiness` / `GET /api/readiness` — Run XGBoost ML readiness prediction
- `POST /api/intelligence/role-match` — Weighted skill overlap role matching
- `POST /api/ai/guide` / `POST /api/career-guide` — NLP Intent classifier + dynamic grounded answers
- `GET /api/weekly-test` — Retrieve adaptive skill test (excludes past attempted questions)
- `POST /api/weekly-test/submit` — Submit test answers and persist score history

---

## 3. JSON Storage Files
- `backend/storage_data/users.json` — Source of truth for accounts, profile data, skills, roadmap, and parsed resumes.
- `c:/Ascentra/data.json` — Mirrored sync file ensuring legacy client compatibility.
- `backend/storage_data/assessments_history.json` — Per-student question history to prevent repeated test questions.

---

## 4. Raw Datasets Used
- **Dataset A**: `student_placement_prediction_dataset_2026.csv` (100,000 rows, 26 columns)
- **Dataset B**: `all_job_post.csv` (1,167 job postings)
- **Dataset C**: `onet.json` (O*NET occupations & skills database)
- **Dataset D**: `training_data.csv` (10,000 labeled resumes), `job_roles.csv`, `skills_list.csv`, `skills_database.json`, `test_resumes.json`

---

## 5. ML Models Trained & Evaluation Metrics

| Model Task | Algorithm | Train/Test Split | Metric Highlights | Saved Binary |
| :--- | :--- | :--- | :--- | :--- |
| **Placement Readiness Baseline** | Logistic Regression | 80 / 20 Stratified | Accuracy: `0.8778`<br>F1: `0.9341`<br>ROC-AUC: `0.7681` | `placement_logistic_regression.joblib` |
| **Placement Readiness Candidate** | XGBoost Classifier | 80 / 20 Stratified | Accuracy: `0.8777`<br>F1: `0.9339`<br>ROC-AUC: `0.7611` | `placement_xgboost.joblib` |
| **Resume → Role Classification** | TF-IDF + Logistic Regression | 80 / 20 Stratified | Accuracy: `1.0000`<br>F1: `1.0000` | `resume_role_classifier.joblib` |
| **AI Guide Intent Classifier** | TF-IDF + Logistic Regression | Multi-class Intent | Accuracy: `1.0000` | `intent_classifier.joblib` |

---

## 6. Features Used & Exclusions (Data Leakage / Fairness)
- **Included Student Features**: `cgpa`, `branch`, `college_tier`, `internships_count`, `projects_count`, `certifications_count`, `coding_skill_score`, `aptitude_score`, `communication_skill_score`, `logical_reasoning_score`, `hackathons_participated`, `github_repos`, `linkedin_connections`, `mock_interview_score`, `attendance_percentage`, `backlogs`, `extracurricular_score`, `leadership_score`, `volunteer_experience`, `study_hours_per_day`.
- **Excluded Features & Rationale**:
  - `student_id`: Arbitrary non-predictive identifier.
  - `salary_package_lpa`: **Data Leakage** (salary is downstream of placement outcome).
  - `gender`: **Responsible ML / Fairness** (demographic attribute excluded from decision logic).
  - `sleep_hours`: Uncorrelated lifestyle variable.

---

## 7. Explainability & Resume Identity Validation
- **SHAP / Feature Attribution**: Predictor exposes transparent positive factors (e.g. `Strong CGPA`, `Project evidence`, `Internships`) and negative risk factors (e.g. `Active backlogs`, `Missing CGPA`).
- **Identity Validation**: Validates `profile.full_name` against extracted resume name. Rejects mismatches with friendly error warning rather than saving corrupted profile signals.

---

## 8. Server Execution Commands

### To Train All ML Models & Process Datasets:
```powershell
py -m app.ml.process_knowledge
py -m app.ml.train_resume_roles
py -m app.ml.train_placement
py -m app.ml.intent_classifier
```

### To Run Backend API Server (Port 8000):
```powershell
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### To Run Pytest Suite:
```powershell
py -m pytest tests/test_backend.py
```

---

## 9. Future University Data Insertion Points
To replace prototype datasets with actual university data:
1. Replace `backend/data/raw/student_placement_prediction_dataset_2026.csv` with official university placement records.
2. Re-run `py -m app.ml.train_placement` to retrain and update model weights in `models/trained/`.

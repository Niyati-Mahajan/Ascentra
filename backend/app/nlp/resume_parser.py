import os
import re
from typing import List, Dict, Any, Tuple

# Base technology skills list
TECHNICAL_SKILLS = [
    "JavaScript", "TypeScript", "React", "Node.js", "Express.js", "SQL", "Git", "GitHub",
    "Python", "Java", "C++", "C", "DSA", "MongoDB", "MySQL", "PostgreSQL", "REST APIs",
    "Docker", "AWS", "HTML", "CSS", "Statistics", "Machine Learning", "Pandas", "NumPy",
    "Figma", "Linux", "Kubernetes", "CI/CD", "TensorFlow", "Power BI", "Communication"
]

SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving", "adaptability",
    "collaboration", "time management"
]

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    return re.sub(r'\n+', '\n', text).strip()

def normalize_skill(skill: str) -> str:
    return re.sub(r'[^a-z0-9+#.]', '', skill.lower())

def extract_skills(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for skill in TECHNICAL_SKILLS:
        pattern = r'(?:^|[^a-z0-9])' + re.escape(skill.lower()) + r'(?:[^a-z0-9]|$)'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found

def extract_soft_skills(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for skill in SOFT_SKILLS:
        if skill in text_lower:
            found.append(skill)
    return found

def extract_sections(text: str) -> Dict[str, bool]:
    sections = ["projects", "experience", "internships", "education", "certifications", "training", "skills"]
    res = {}
    for s in sections:
        res[s] = bool(re.search(r'(?:^|\n)\s*' + s, text, re.IGNORECASE))
    return res

# Resume header / section keywords to skip when extracting name
_RESUME_KEYWORDS = re.compile(
    r'(skills|education|projects|experience|internships|certifications|curriculum|'
    r'resume|cv|contact|objective|summary|profile|address|phone|email|linkedin|github|'
    r'mobile|tel|website|portfolio|gpa|cgpa|university|college|institute|department|'
    r'bachelor|master|degree|b\.?tech|m\.?tech|b\.?sc|m\.?sc)',
    re.IGNORECASE
)

def extract_identity_name(text: str) -> str:
    """Extract the candidate's name from a resume text.
    
    Strategy:
    1. Scan the first 25 non-empty lines.
    2. Skip lines that contain resume section keywords, numbers, or special chars.
    3. Accept the first line that looks like a human name (2–4 capitalized words).
    4. Fallback: try the very first non-empty line even if it has minor issues.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Pattern: 2–4 words that are each capitalized (or all-caps), no digits/special chars
    name_pattern = re.compile(r'^[A-Z][a-zA-Z]+(?:\s[A-Za-z]+){1,4}$')
    
    for line in lines[:25]:
        # Skip if it's too short or too long
        if len(line) < 3 or len(line) > 70:
            continue
        # Skip lines with digits (phone numbers, years, grades etc.)
        if re.search(r'\d', line):
            continue
        # Skip lines with @ (emails), : (labels), | / \\ (separators)
        if re.search(r'[@:|/\\]', line):
            continue
        # Skip known resume section keywords
        if _RESUME_KEYWORDS.search(line):
            continue
        # Accept lines that match a name pattern
        if name_pattern.match(line):
            return line
        # Also accept if all parts look like name tokens (purely alphabetic, 2-5 words)
        parts = line.split()
        if 2 <= len(parts) <= 5 and all(re.match(r'^[A-Za-z]+$', p) for p in parts):
            return line
    
    # Last resort: return the first non-empty, all-alpha line (even 1 word)
    for line in lines[:5]:
        if line.strip() and re.match(r'^[A-Za-z][A-Za-z .\'-]{2,60}$', line.strip()):
            if not _RESUME_KEYWORDS.search(line):
                return line.strip()
    
    return ""


def _tokenize_name(s: str) -> List[str]:
    """Split a name string into lowercase alphabetic tokens of length >= 2."""
    return [w.lower() for w in re.findall(r'[a-zA-Z]+', s) if len(w) >= 2]


def validate_resume_identity(profile_full_name: str, resume_name: str) -> Tuple[bool, str]:
    """Validate that the resume belongs to the logged-in student.
    
    Matching rules (any one is sufficient to pass):
      1. Shared token(s): at least 1 non-trivial word in common (e.g. surname).
      2. Substring containment: profile name contains resume name or vice-versa.
      3. Initials match: resume initials appear in profile name tokens.
    
    A missing resume_name (extraction failure) is not penalised — it passes.
    """
    # If we couldn't extract a name from the resume, don't penalise
    if not resume_name:
        return True, "unverified"
    # If profile name is empty, skip validation too
    if not profile_full_name:
        return True, "unverified"
    
    # Very short or single-word names extracted are unreliable — skip
    resume_words = resume_name.strip().split()
    if len(resume_words) < 2:
        return True, "unverified"
    
    prof_tokens = _tokenize_name(profile_full_name)
    res_tokens  = _tokenize_name(resume_name)
    
    if not prof_tokens or not res_tokens:
        return True, "unverified"
    
    # Common stopwords / trivial tokens to exclude from shared-token check
    _TRIVIAL = {"mr", "ms", "mrs", "dr", "sr", "jr", "ii", "iii", "iv"}
    meaningful_prof = [t for t in prof_tokens if t not in _TRIVIAL]
    meaningful_res  = [t for t in res_tokens  if t not in _TRIVIAL]
    
    # Rule 1: shared tokens (at least 1 meaningful surname / given-name word in common)
    shared = set(meaningful_prof).intersection(set(meaningful_res))
    if shared:
        return True, "verified"
    
    # Rule 2: substring — does the profile name contain the resume name or vice-versa?
    prof_flat = profile_full_name.lower().replace(" ", "")
    res_flat  = resume_name.lower().replace(" ", "")
    if res_flat in prof_flat or prof_flat in res_flat:
        return True, "verified"
    
    # Rule 3: initials — each initial in resume_name matches a token in prof_tokens
    initials = [w[0].lower() for w in meaningful_res if w]
    if initials and all(any(t.startswith(i) for t in meaningful_prof) for i in initials):
        return True, "verified"
    
    return False, "mismatch"

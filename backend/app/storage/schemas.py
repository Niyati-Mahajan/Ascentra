from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Optional, Any

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    name: Optional[str] = None
    remember: Optional[bool] = False

class UserLogin(BaseModel):
    email: str
    password: str
    remember: Optional[bool] = False

class StudentProject(BaseModel):
    id: Optional[str] = None
    name: str
    detail: str

class StudentProfile(BaseModel):
    full_name: str = ""
    degree: Optional[str] = ""
    branch: Optional[str] = ""
    university: Optional[str] = ""
    year: Optional[int] = None
    semester: Optional[int] = None
    cgpa: Optional[float] = None
    backlogs: int = 0
    technical_skills: Dict[str, int] = Field(default_factory=dict)
    soft_skills: List[str] = Field(default_factory=list)
    projects: List[StudentProject] = Field(default_factory=list)
    internships: int = 0
    certifications: List[str] = Field(default_factory=list)
    career_interests: List[str] = Field(default_factory=list)
    target_role: Optional[str] = None
    roadmap: Dict[str, bool] = Field(default_factory=dict)
    resume: Dict[str, Any] = Field(default_factory=dict)
    assessment: Dict[str, Any] = Field(default_factory=dict)

class UserRecord(BaseModel):
    id: str
    username: str
    email: str
    password_hash: str
    profile: StudentProfile
    has_logged_in: bool = True

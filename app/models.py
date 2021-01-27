from pydantic import BaseModel, ValidationError, validator
from typing import Dict, List, Optional

class Company(BaseModel):
    url: str
    name: str
    founded: Optional[int]
    headquarters: Optional[str]
    description: str
    
    @validator('*', pre=True)
    def blank_str(cls, v):
        if v == "":
            return None
        return v
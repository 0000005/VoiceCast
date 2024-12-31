from pydantic import BaseModel
from typing import Optional
import os


class Settings(BaseModel):
    # AI Settings
    baseUrl: str = ""
    model: str = ""
    apiKey: str = ""

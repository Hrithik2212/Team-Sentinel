# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional,List
from enum import Enum
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: str
    password:str



class LoginUser(BaseModel):
    email:str
    password:str


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str
    id: int
    name: str
    role: str


class User(BaseModel):
    id: int
    name: str
    email: str


    class Config:
        from_attributes = True
        


class LoginRequest(BaseModel):
    email:str
    password:str

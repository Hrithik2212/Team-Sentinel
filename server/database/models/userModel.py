# app/models/user.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

try : 
    from server.database.database import Base
except :
    from database import Base 



class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True) #,autoincrement=True,)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password=Column(String)

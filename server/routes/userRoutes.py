# app/routes/user.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from server.controllers.auth import verify_password, authenticate_request
from server.database.database import get_db
from server.database.schemas import User, UserCreate, Token, LoginRequest
from server.controllers import user_controller
from fastapi.security import OAuth2PasswordBearer
from typing import List

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()


@router.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    return user_controller.create_user(db=db, user=user)


@router.get("/getuser/", response_model=User)
@authenticate_request
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    user_email = request.state.user.get("sub")
    user = user_controller.get_user(db, email=user_email)
    return user


@router.get("/user/all", response_model=List[User])
async def get_current_user(db: Session = Depends(get_db)):
    return user_controller.get_users(db)


@router.post("/token", response_model=Token)
def login_user(form_data: LoginRequest, db: Session = Depends(get_db)):

    db_user = user_controller.get_user(db, email=form_data.email)
    print(db_user)
    if db_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not verify_password(form_data.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_controller.login(db_user)
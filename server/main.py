from fastapi import FastAPI
from server.routes import userRoutes
import pytz
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

kolkata_tz = pytz.timezone('Asia/Kolkata')
app.include_router(userRoutes.router)











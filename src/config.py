from dotenv import load_dotenv
import os

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

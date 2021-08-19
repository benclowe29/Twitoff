from .app import create_app
from dotenv import load_dotenv

load_dotenv()
APP = create_app()


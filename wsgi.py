"""
WSGI entry point for the Accent Detection System.
Used for production deployment with Gunicorn.
"""

from src.main import app

if __name__ == "__main__":
    app.run()

import sys
import os

# Add root to path so "from main import app" works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# THIS LINE IS CRITICAL FOR HF SPACES + OPENENV
main = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:main",   # Important: reference :main
        host="0.0.0.0",
        port=7860
    )

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# ← Yeh line HF OpenEnv validator ke liye sabse zaroori hai
main = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,                    # yahan app pass kar rahe hain
        host="0.0.0.0",
        port=7860
    )

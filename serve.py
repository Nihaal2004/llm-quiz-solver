# serve.py
import os, uvicorn
from main import app
uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

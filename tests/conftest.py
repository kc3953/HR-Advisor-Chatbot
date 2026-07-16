import os

# app.main initializes a Vertex AI client at import time and requires this env
# var to be set, even though these tests never make a real Gemini call.
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

import os

UPLOAD_FOLDER = 'docs'
ALLOWED_EXTENSIONS = {'pdf'}

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_BASE="vb"
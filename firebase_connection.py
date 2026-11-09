import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Firebase credentials path
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not credentials_path:
    raise ValueError("Firebase credentials path not found in .env file")

# Initialize Firebase
cred = credentials.Certificate(credentials_path)

firebase_admin.initialize_app(cred)

# Connect to Firestore
db = firestore.client()

print("Firebase connected successfully!")

# Test Firestore connection
test_data = {
    "message": "Day 27 Firebase connection successful",
    "status": "Connected"
}

db.collection("connection_tests").add(test_data)

print("Test document uploaded to Firestore!")
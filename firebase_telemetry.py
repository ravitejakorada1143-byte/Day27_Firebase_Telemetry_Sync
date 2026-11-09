import os
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase
if not firebase_admin._apps:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path:
        raise ValueError("Firebase credentials path not found")

    cred = credentials.Certificate(credentials_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def send_proctor_alert(
    candidate_id,
    session_id,
    alerts,
    severity_level,
    latency_ms
):
    """
    Upload a proctoring alert to Firebase Firestore.
    """

    telemetry_data = {
        "candidate_id": candidate_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc),
        "alerts": alerts,
        "severity_level": severity_level,
        "latency_ms": latency_ms
    }

    try:
        document_ref = (
            db.collection("proctor_tracking")
            .add(telemetry_data)
        )

        print("Proctor alert uploaded successfully!")
        print("Document ID:", document_ref[1].id)

        return True

    except Exception as error:
        print("Firebase upload failed:", error)
        return False
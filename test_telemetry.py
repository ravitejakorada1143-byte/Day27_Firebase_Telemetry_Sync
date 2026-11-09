from firebase_telemetry import send_proctor_alert

sample_alerts = {
    "gaze_deviation": {
        "triggered": True,
        "duration_seconds": 2.4,
        "vector_shift": {
            "x": -0.35,
            "y": 0.12
        }
    },
    "acoustic_anomaly": {
        "triggered": True,
        "avg_logprob": -1.34,
        "flag_type": "BACKGROUND_WHISPER"
    }
}

send_proctor_alert(
    candidate_id="CAND001",
    session_id="SESSION001",
    alerts=sample_alerts,
    severity_level="CRITICAL",
    latency_ms=1120
)
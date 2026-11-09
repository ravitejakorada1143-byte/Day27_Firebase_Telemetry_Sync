import os
import cv2
import mediapipe as mp
import time
import json
import queue
import threading
import tempfile
import wave
import pyaudio

from firebase_telemetry import send_proctor_alert
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=GROQ_API_KEY)

MODEL_PATH = "face_landmarker.task"
OUTPUT_FILE = "proctor_alerts.json"

# Local video and audio recording
RECORDINGS_DIR = "recordings"

# Fixed filenames: each new run replaces the previous recording.
VIDEO_FILE = os.path.join(
    RECORDINGS_DIR,
    "webcam.mp4"
)

AUDIO_FILE = os.path.join(
    RECORDINGS_DIR,
    "microphone.wav"
)

CANDIDATE_ID = "CANDIDATE_001"
SESSION_ID = "SESSION_001"

GAZE_THRESHOLD = 0.08
GAZE_ALERT_DURATION = 2.0

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 1024
CHUNK_SECONDS = 5

# Shared event queue
event_queue = queue.Queue()

# Stop signal
stop_event = threading.Event()

# Lock for shared data
data_lock = threading.Lock()


# ==================================================
# 2. SHARED ALERT STATES
# ==================================================

latest_gaze = {
    "triggered": False,
    "duration_seconds": 0.0,
    "vector_shift": {
        "x": 0.0,
        "y": 0.0
    }
}

latest_acoustic = {
    "triggered": False,
    "avg_logprob": None,
    "flag_type": None
}


# ==================================================
# 3. CENTRAL ALERT ENGINE
# ==================================================

def generate_proctor_alert(
    gaze_alert,
    acoustic_alert,
    latency_ms
):

    active_alerts = sum([
        gaze_alert["triggered"],
        acoustic_alert["triggered"]
    ])

    if active_alerts == 0:
        severity = "NORMAL"

    elif active_alerts == 1:
        severity = "WARNING"

    else:
        severity = "CRITICAL"

    payload = {

        "candidate_id": CANDIDATE_ID,

        "session_id": SESSION_ID,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "alerts": {

            "gaze_deviation": gaze_alert,

            "acoustic_anomaly": acoustic_alert

        },

        "severity_level": severity,

        "latency_ms": round(latency_ms, 2)

    }

    return payload


# ==================================================
# 4. SAVE ALERT TO JSON
# ==================================================

def save_alert(payload):

    try:

        records = []

        if os.path.exists(OUTPUT_FILE):

            with open(
                OUTPUT_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                try:

                    records = json.load(file)

                    if not isinstance(records, list):
                        records = []

                except json.JSONDecodeError:

                    records = []

        records.append(payload)

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                indent=4
            )

        print(
            "\nALERT SAVED:",
            payload["severity_level"]
        )

    except Exception as error:

        print("JSON save error:", error)


# ==================================================
# 5. SEVERITY TESTING
# ==================================================

def test_severity_levels():

    normal_gaze = {
        "triggered": False,
        "duration_seconds": 0.0,
        "vector_shift": {
            "x": 0.0,
            "y": 0.0
        }
    }

    normal_audio = {
        "triggered": False,
        "avg_logprob": -0.4,
        "flag_type": None
    }

    alert_gaze = {
        "triggered": True,
        "duration_seconds": 2.5,
        "vector_shift": {
            "x": 0.15,
            "y": 0.0
        }
    }

    alert_audio = {
        "triggered": True,
        "avg_logprob": -1.2,
        "flag_type": "BACKGROUND_WHISPER"
    }

    tests = [

        (normal_gaze, normal_audio, "NORMAL"),

        (alert_gaze, normal_audio, "WARNING"),

        (normal_gaze, alert_audio, "WARNING"),

        (alert_gaze, alert_audio, "CRITICAL")

    ]

    print("\nSEVERITY TEST RESULTS")

    for gaze, audio, expected in tests:

        result = generate_proctor_alert(
            gaze,
            audio,
            100
        )

        actual = result["severity_level"]

        status = (
            "PASSED"
            if actual == expected
            else "FAILED"
        )

        print(
            f"Expected: {expected} | "
            f"Actual: {actual} | {status}"
        )


# ==================================================
# 6. GAZE CALCULATION
# ==================================================

def calculate_gaze_ratio(landmarks):

    left_eye = [33, 133]
    right_eye = [362, 263]

    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]

    def iris_ratio(eye_indices, iris_indices):

        eye_left = landmarks[eye_indices[0]].x

        eye_right = landmarks[eye_indices[1]].x

        iris_x = sum(
            landmarks[index].x
            for index in iris_indices
        ) / len(iris_indices)

        eye_width = abs(
            eye_right - eye_left
        )

        if eye_width == 0:
            return 0.5

        return (
            abs(iris_x - min(eye_left, eye_right))
            / eye_width
        )

    left_ratio = iris_ratio(
        left_eye,
        left_iris
    )

    right_ratio = iris_ratio(
        right_eye,
        right_iris
    )

    return (
        left_ratio + right_ratio
    ) / 2


# ==================================================
# 7. WEBCAM THREAD
# ==================================================

def webcam_worker():

    global latest_gaze

    if not os.path.exists(MODEL_PATH):

        event_queue.put({
            "type": "worker_error",
            "message": "face_landmarker.task not found"
        })

        return

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        event_queue.put({
            "type": "worker_error",
            "message": "Could not open webcam"
        })

        landmarker.close()
        return

    # Start local webcam recording
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    frame_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    video_fps = camera.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        video_fps = 20.0

    video_writer = cv2.VideoWriter(
        VIDEO_FILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():

        print("Unable to start webcam video recording.")

        camera.release()
        landmarker.close()
        return

    print("Video recording started:", VIDEO_FILE)

    print("\nWebcam started.")

    print("Look at the camera for 3 seconds.")

    start_time = time.monotonic()

    last_timestamp_ms = -1

    calibration_values = []

    center_gaze = None

    off_center_start = None

    alert_reported = False

    try:

        while not stop_event.is_set():

            success, frame = camera.read()

            if not success:

                print("Unable to read webcam frame.")

                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Correct MediaPipe image construction
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = max(
                int(
                    (time.monotonic() - start_time)
                    * 1000
                ),
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            frame_text = "No face detected"

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                gaze_ratio = calculate_gaze_ratio(
                    landmarks
                )

                elapsed = (
                    time.monotonic() - start_time
                )

                # Three-second calibration
                if center_gaze is None:

                    calibration_values.append(
                        gaze_ratio
                    )

                    frame_text = "Calibrating..."

                    if (
                        elapsed >= 3.0
                        and calibration_values
                    ):

                        center_gaze = (
                            sum(calibration_values)
                            / len(calibration_values)
                        )

                        print(
                            "Center gaze calibrated:",
                            round(center_gaze, 3)
                        )

                else:

                    shift_x = (
                        gaze_ratio - center_gaze
                    )

                    off_center = (
                        abs(shift_x) > GAZE_THRESHOLD
                    )

                    if off_center:

                        if off_center_start is None:

                            off_center_start = (
                                time.monotonic()
                            )

                        duration = (
                            time.monotonic()
                            - off_center_start
                        )

                        frame_text = (
                            f"Looking away: {duration:.1f}s"
                        )

                        if duration > GAZE_ALERT_DURATION:

                            frame_text = (
                                "GAZE DEVIATION ALERT!"
                            )

                            gaze_snapshot = {

                                "triggered": True,

                                "duration_seconds": round(
                                    duration, 2
                                ),

                                "vector_shift": {

                                    "x": round(
                                        shift_x, 3
                                    ),

                                    "y": 0.0

                                }

                            }

                            with data_lock:

                                latest_gaze = (
                                    gaze_snapshot.copy()
                                )

                            if not alert_reported:

                                event_queue.put({

                                    "type": "gaze",

                                    "gaze_alert": (
                                        gaze_snapshot.copy()
                                    )

                                })

                                print(
                                    "\nGAZE DEVIATION ALERT!"
                                )

                                alert_reported = True

                    else:

                        # Send a reset event if an alert
                        # was previously active.
                        if alert_reported:

                            reset_gaze = {

                                "triggered": False,

                                "duration_seconds": 0.0,

                                "vector_shift": {
                                    "x": 0.0,
                                    "y": 0.0
                                }

                            }

                            event_queue.put({

                                "type": "gaze",

                                "gaze_alert": reset_gaze

                            })

                        off_center_start = None

                        alert_reported = False

                        with data_lock:

                            latest_gaze = {

                                "triggered": False,

                                "duration_seconds": 0.0,

                                "vector_shift": {
                                    "x": 0.0,
                                    "y": 0.0
                                }

                            }

                        frame_text = "Normal gaze"

                    cv2.putText(
                        frame,
                        f"Gaze ratio: {gaze_ratio:.3f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            cv2.putText(

                frame,

                frame_text,

                (20, 75),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255)
                if "ALERT" in frame_text
                else (0, 255, 0),

                2

            )

            # Save the same webcam frame displayed on screen.
            video_writer.write(frame)

            cv2.imshow(
                "Day 26 - Proctor Monitoring",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):

                stop_event.set()

                break

    except Exception as error:

        print("Webcam error:", error)

        event_queue.put({
            "type": "worker_error",
            "message": str(error)
        })

    finally:

        camera.release()

        video_writer.release()

        landmarker.close()

        cv2.destroyAllWindows()

        print("Webcam video saved:", VIDEO_FILE)

        print("Webcam stopped.")


# ==================================================
# 8. GROQ WHISPER AUDIO PROCESSING
# ==================================================

def process_audio_chunk(audio_data):

    temp_path = None

    start = time.monotonic()

    acoustic_alert = {

        "triggered": False,

        "avg_logprob": None,

        "flag_type": None

    }

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

        with wave.open(
            temp_path,
            "wb"
        ) as wav_file:

            wav_file.setnchannels(
                AUDIO_CHANNELS
            )

            wav_file.setsampwidth(2)

            wav_file.setframerate(
                AUDIO_RATE
            )

            wav_file.writeframes(
                audio_data
            )

        with open(
            temp_path,
            "rb"
        ) as audio_file:

            transcription = (
                client.audio.transcriptions.create(

                    file=audio_file,

                    model="whisper-large-v3",

                    response_format="verbose_json",

                    language="en"

                )
            )

        segments = (
            getattr(
                transcription,
                "segments",
                None
            )
            or []
        )

        logprobs = []

        for segment in segments:

            if isinstance(segment, dict):

                value = segment.get(
                    "avg_logprob"
                )

            else:

                value = getattr(
                    segment,
                    "avg_logprob",
                    None
                )

            if value is not None:

                logprobs.append(
                    float(value)
                )

        if logprobs:

            avg_logprob = (
                sum(logprobs)
                / len(logprobs)
            )

            # KT acoustic confidence threshold
            if avg_logprob < -1.0:

                acoustic_alert = {

                    "triggered": True,

                    "avg_logprob": round(
                        avg_logprob, 4
                    ),

                    "flag_type": "BACKGROUND_WHISPER"

                }

                print(
                    "\nACOUSTIC ANOMALY:",
                    avg_logprob
                )

            else:

                acoustic_alert["avg_logprob"] = (
                    round(avg_logprob, 4)
                )

        else:

            print(
                "No Whisper confidence values returned."
            )

    except Exception as error:

        print(
            "Audio processing error:",
            error
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)

    latency_ms = (
        time.monotonic() - start
    ) * 1000

    event_queue.put({

        "type": "audio",

        "acoustic_alert": acoustic_alert,

        "latency_ms": latency_ms

    })


# ==================================================
# 9. MICROPHONE THREAD
# ==================================================

def microphone_worker():

    audio = pyaudio.PyAudio()

    stream = None
    wav_file = None

    try:

        stream = audio.open(

            format=pyaudio.paInt16,

            channels=AUDIO_CHANNELS,

            rate=AUDIO_RATE,

            input=True,

            frames_per_buffer=AUDIO_CHUNK

        )

        print("\nMicrophone started.")

        # Start local microphone recording
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        wav_file = wave.open(
            AUDIO_FILE,
            "wb"
        )

        wav_file.setnchannels(
            AUDIO_CHANNELS
        )

        wav_file.setsampwidth(
            audio.get_sample_size(pyaudio.paInt16)
        )

        wav_file.setframerate(
            AUDIO_RATE
        )

        print("Audio recording started:", AUDIO_FILE)

        while not stop_event.is_set():

            frames = []

            for _ in range(
                int(
                    AUDIO_RATE
                    / AUDIO_CHUNK
                    * CHUNK_SECONDS
                )
            ):

                if stop_event.is_set():

                    break

                audio_data_part = stream.read(
                    AUDIO_CHUNK,
                    exception_on_overflow=False
                )

                # Save each captured audio sample locally.
                wav_file.writeframes(audio_data_part)

                # Preserve original Whisper processing.
                frames.append(audio_data_part)

            if frames and not stop_event.is_set():

                audio_data = b"".join(
                    frames
                )

                # Run Groq processing separately
                threading.Thread(

                    target=process_audio_chunk,

                    args=(audio_data,),

                    daemon=True

                ).start()

    except Exception as error:

        print(
            "Microphone error:",
            error
        )

        event_queue.put({

            "type": "worker_error",

            "message": str(error)

        })

    finally:

        if stream is not None:

            stream.stop_stream()

            stream.close()

        if wav_file is not None:

            wav_file.close()

            print("Microphone audio saved:", AUDIO_FILE)

        audio.terminate()

        print("Microphone stopped.")


# ==================================================
# 10. CENTRAL EVENT PROCESSOR
# ==================================================

def central_alert_worker():

    print("\nCentral alert engine started.")

    current_gaze = {

        "triggered": False,

        "duration_seconds": 0.0,

        "vector_shift": {
            "x": 0.0,
            "y": 0.0
        }

    }

    current_acoustic = {

        "triggered": False,

        "avg_logprob": None,

        "flag_type": None

    }

    while (

        not stop_event.is_set()

        or not event_queue.empty()

    ):

        try:

            event = event_queue.get(
                timeout=0.5
            )

        except queue.Empty:

            continue

        if event["type"] == "worker_error":

            print(
                "Worker error:",
                event["message"]
            )

            continue

        # Update gaze state
        if event["type"] == "gaze":

            current_gaze = (
                event["gaze_alert"].copy()
            )

        # Update acoustic state
        elif event["type"] == "audio":

            current_acoustic = (
                event["acoustic_alert"].copy()
            )

        else:

            continue

        # Generate combined alert
        payload = generate_proctor_alert(

            current_gaze,

            current_acoustic,

            event.get("latency_ms", 0)

        )

        save_alert(payload)

        print(
            "\nCENTRALIZED ALERT RESULT:"
        )

        print(
            json.dumps(
                payload,
                indent=4
            )
        )

                # Firebase telemetry integration
        if payload["severity_level"] in ["WARNING", "CRITICAL"]:

            firebase_status = send_proctor_alert(
                candidate_id=payload["candidate_id"],
                session_id=payload["session_id"],
                alerts=payload["alerts"],
                severity_level=payload["severity_level"],
                latency_ms=payload["latency_ms"]
            )

            if firebase_status:
                print("Firebase telemetry sync completed.")
            else:
                print("Firebase telemetry sync failed.")


# ==================================================
# 11. MAIN FUNCTION
# ==================================================

def main():

    print("=" * 55)

    print(
        "DAY 26 - CENTRALIZED PROCTOR ALERT ENGINE"
    )

    print("=" * 55)

    print(
        "Candidate ID:",
        CANDIDATE_ID
    )

    print(
        "Session ID:",
        SESSION_ID
    )

    # Run severity tests before monitoring
    test_severity_levels()

    print(
        "\nStarting monitoring threads..."
    )

    central_thread = threading.Thread(

        target=central_alert_worker,

        name="Central-Alert-Thread",

        daemon=True

    )

    webcam_thread = threading.Thread(

        target=webcam_worker,

        name="Webcam-Thread",

        daemon=True

    )

    microphone_thread = threading.Thread(

        target=microphone_worker,

        name="Microphone-Thread",

        daemon=True

    )

    central_thread.start()

    webcam_thread.start()

    microphone_thread.start()

    try:

        while (

            webcam_thread.is_alive()

            and not stop_event.is_set()

        ):

            time.sleep(0.5)

    except KeyboardInterrupt:

        print(
            "\nStopping monitoring..."
        )

    finally:

        stop_event.set()

        webcam_thread.join(timeout=5)

        microphone_thread.join(timeout=5)

        central_thread.join(timeout=5)

        print(
            "\nDay 26 monitoring stopped."
        )

        print(
            "Check proctor_alerts.json for saved alerts."
        )

        print(
            "Video recording:",
            VIDEO_FILE
        )

        print(
            "Audio recording:",
            AUDIO_FILE
        )


if __name__ == "__main__":

    main()
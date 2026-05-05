import os
import sqlite3
import cv2
import numpy as np
from uuid import uuid4
from flask import Flask, render_template, request, redirect
import tensorflow as tf
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DB_PATH = "strabismus_assessments.db"

# Load models
image_model = tf.keras.models.load_model("strabismus_mobilenetv2_model.h5")
questionnaire_model = joblib.load("questionnaire_model.pkl")
scaler = joblib.load("questionnaire_scaler.pkl")
IMAGE_DETECTION_THRESHOLD = 0.80
VERY_HIGH_IMAGE_THRESHOLD = 0.999


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                image_result TEXT NOT NULL,
                detection TEXT NOT NULL,
                risk TEXT NOT NULL,
                final_msg TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                reason TEXT NOT NULL,
                age INTEGER NOT NULL,
                onset INTEGER NOT NULL,
                double_vision INTEGER NOT NULL,
                headache INTEGER NOT NULL,
                diabetes INTEGER NOT NULL,
                thyroid INTEGER NOT NULL,
                crop_method TEXT NOT NULL,
                image_score REAL NOT NULL,
                questionnaire_result INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
            )
            """
        )


def save_assessment(patient_name, assessment_data):
    with get_db_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        patient_row = connection.execute(
            "SELECT id FROM patients WHERE patient_name = ?",
            (patient_name,),
        ).fetchone()

        if patient_row is None:
            cursor = connection.execute(
                "INSERT INTO patients (patient_name) VALUES (?)",
                (patient_name,),
            )
            patient_id = cursor.lastrowid
        else:
            patient_id = patient_row["id"]

        connection.execute(
            """
            INSERT INTO assessments (
                patient_id, image_path, image_result, detection, risk, final_msg,
                recommendation, reason, age, onset, double_vision, headache,
                diabetes, thyroid, crop_method, image_score, questionnaire_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                assessment_data["image_path"],
                assessment_data["image_result"],
                assessment_data["detection"],
                assessment_data["risk"],
                assessment_data["final_msg"],
                assessment_data["recommendation"],
                assessment_data["reason"],
                assessment_data["age"],
                assessment_data["onset"],
                assessment_data["double_vision"],
                assessment_data["headache"],
                assessment_data["diabetes"],
                assessment_data["thyroid"],
                assessment_data["crop_method"],
                assessment_data["image_score"],
                assessment_data["questionnaire_result"],
            ),
        )


def load_history(limit=100):
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                a.id,
                p.patient_name,
                a.image_path,
                a.image_result,
                a.detection,
                a.risk,
                a.final_msg,
                a.recommendation,
                a.reason,
                a.age,
                a.onset,
                a.double_vision,
                a.headache,
                a.diabetes,
                a.thyroid,
                a.crop_method,
                a.image_score,
                a.questionnaire_result,
                a.created_at
            FROM assessments a
            JOIN patients p ON p.id = a.patient_id
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows


init_db()


def preprocess_eye_image(full_path):
    """Load image and return model tensor plus diagnostics about crop selection."""
    img_bgr = cv2.imread(full_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {full_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_bgr.shape[:2]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )

    roi = None
    crop_method = "fallback_upper_center"

    # 1) Preferred path: detect a face, then detect eyes inside that face.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) > 0:
        fx, fy, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_gray = gray[fy:fy + fh, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(eyes) >= 2:
            # Use top two largest eyes and crop a union region around them.
            top_eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            ex1 = min(e[0] for e in top_eyes)
            ey1 = min(e[1] for e in top_eyes)
            ex2 = max(e[0] + e[2] for e in top_eyes)
            ey2 = max(e[1] + e[3] for e in top_eyes)

            pad_x = int(0.35 * (ex2 - ex1))
            pad_y = int(0.65 * (ey2 - ey1))

            x1 = max(0, fx + ex1 - pad_x)
            y1 = max(0, fy + ey1 - pad_y)
            x2 = min(w_img, fx + ex2 + pad_x)
            y2 = min(h_img, fy + ey2 + pad_y)
            roi = img_bgr[y1:y2, x1:x2]
            crop_method = "face_two_eyes_union"

        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            # Keep a broader upper-face ROI so both eyes are likely included.
            band_w = int(0.9 * fw)
            band_h = int(0.62 * fh)
            cx = fx + fw // 2
            cy = fy + int(0.33 * fh)
            x1 = max(0, cx - band_w // 2)
            y1 = max(0, cy - band_h // 2)
            x2 = min(w_img, cx + band_w // 2)
            y2 = min(h_img, cy + band_h // 2)
            roi = img_bgr[y1:y2, x1:x2]
            crop_method = "face_upper_band"

    # 2) Fallback: detect eyes globally (for non-face-centered captures).
    if roi is None:
        eyes_global = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(eyes_global) > 0:
            ex, ey, ew, eh = sorted(eyes_global, key=lambda e: e[2] * e[3], reverse=True)[0]
            # Wider patch to capture both-eye context from a single eye detection.
            side = int(4.2 * max(ew, eh))
            cx = ex + ew // 2
            cy = ey + eh // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w_img, cx + side // 2)
            y2 = min(h_img, cy + side // 2)
            roi = img_bgr[y1:y2, x1:x2]
            crop_method = "global_wide_eye_context"

    # 3) Final fallback: upper-center crop where eyes usually are.
    if roi is None or roi.size == 0:
        side = int(min(h_img, w_img) * 0.7)
        cx = w_img // 2
        cy = int(h_img * 0.38)
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, cx + side // 2)
        y2 = min(h_img, cy + side // 2)
        roi = img_bgr[y1:y2, x1:x2]
        crop_method = "fallback_upper_center"

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (224, 224)).astype(np.float32) / 255.0
    tensor = np.expand_dims(roi_resized, axis=0)
    return tensor, roi_rgb, crop_method

# HOME
@app.route("/")
def home():
    return render_template("home.html")

# UPLOAD
@app.route("/upload")
def upload():
    return render_template("upload.html")

# QUESTIONNAIRE
@app.route("/questionnaire", methods=["POST"])
def questionnaire():

    file = request.files["image"]

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    unique_name = f"input_{uuid4().hex}{ext}"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    img_path = f"uploads/{unique_name}"
    return render_template("questionnaire.html", img_path=img_path)

# PREDICT
@app.route("/predict", methods=["POST"])
def predict():

    img_path = request.form["img_path"]
    full_path = os.path.join("static", img_path)
    patient_name = request.form.get("patient_name", "").strip()
    if not patient_name:
        patient_name = f"Patient-{uuid4().hex[:8]}"

    img, roi_rgb, crop_method = preprocess_eye_image(full_path)

    pred1 = image_model.predict(img)[0][0]
    pred2 = image_model.predict(np.flip(img, axis=2))[0][0]
    pred = float((pred1 + pred2) / 2.0)
    debug_crop_path = os.path.join(app.config["UPLOAD_FOLDER"], "debug_last_crop.jpg")
    cv2.imwrite(debug_crop_path, cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
    print(
        f"DEBUG: Prediction value = {pred}, pred1 = {pred1}, pred2 = {pred2}, "
        f"Image path = {full_path}, crop_method = {crop_method}, crop_shape = {roi_rgb.shape}"
    )
    image_positive = bool(pred1 >= IMAGE_DETECTION_THRESHOLD and pred2 >= IMAGE_DETECTION_THRESHOLD)

    # CLINICAL INPUTS
    age = int(request.form["age"])
    onset = int(request.form["onset"])
    double_vision = int(request.form["double_vision"])
    headache = int(request.form["headache"])
    diabetes = int(request.form["diabetes"])
    thyroid = int(request.form["thyroid"])

    # QUESTIONNAIRE MODEL
    features = np.array([[age, onset, double_vision, headache, diabetes, thyroid]])
    features = scaler.transform(features)

    q_pred = questionnaire_model.predict(features)[0]

    risk_map = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    risk = risk_map[q_pred]

    # False-positive guard:
    # If image is positive but questionnaire risk is LOW, require very high image confidence
    # before confirming strabismus detection.
    is_strabismus = image_positive and (q_pred != 0 or pred >= VERY_HIGH_IMAGE_THRESHOLD)
    image_result = "Strabismus Eye" if image_positive else "Normal Eye"
    detection = "Detected" if is_strabismus else "Not Detected"

    # FINAL MESSAGE
    final_msg = "STRABISMUS PRESENT" if detection == "Detected" else "NO STRABISMUS"

    # ✅ FIX: INDENTATION CORRECTED
    reason = []

    if age > 18:
        reason.append("Age > 18")

    if double_vision == 1:
        reason.append("Double vision present")

    if headache == 1:
        reason.append("Frequent headaches")

    if diabetes == 1:
        reason.append("Diabetes risk factor")

    if thyroid == 1:
        reason.append("Thyroid issue risk")

    if not reason:
        reason.append("No major risk factors detected")

    reason = ", ".join(reason)

    # RECOMMENDATION
    if risk == "HIGH":
        recommendation = "Consult an ophthalmologist immediately."
    elif risk == "MODERATE":
        recommendation = "Eye check-up recommended within a week."
    else:
        recommendation = "Routine monitoring is enough."

    save_assessment(
        patient_name,
        {
            "image_path": img_path,
            "image_result": image_result,
            "detection": detection,
            "risk": risk,
            "final_msg": final_msg,
            "recommendation": recommendation,
            "reason": reason,
            "age": age,
            "onset": onset,
            "double_vision": double_vision,
            "headache": headache,
            "diabetes": diabetes,
            "thyroid": thyroid,
            "crop_method": crop_method,
            "image_score": pred,
            "questionnaire_result": int(q_pred),
        },
    )

    # FINAL OUTPUT
    return render_template("result.html",
                           img_path=img_path,
                           patient_name=patient_name,
                           image_result=image_result,
                           detection=detection,
                           risk=risk,
                           final_msg=final_msg,
                           age=age,
                           reason=reason,
                           recommendation=recommendation)


@app.route("/history")
def history():
    assessments = load_history()
    return render_template("history.html", assessments=assessments)

# RUN
if __name__ == "__main__":
    app.run(debug=True)
import os
import uuid
import html
import re
import traceback
import pandas as pd
from flask import Flask, request, jsonify, render_template

#import models
from models import load_and_build_models, answer_message_for_role, extract_patientid_from_text, extract_doctorid_from_text, get_patient_pin_field, get_doctor_pin_field

app = Flask(__name__, static_folder="static", template_folder="templates")

print("SERVER: starting, loading bundle (initial)...")
try:
    bundle = load_and_build_models(force_rebuild=False)
    print("SERVER: bundle loaded successfully:", type(bundle))
except Exception as e:
    bundle = None
    print("SERVER: ERROR loading bundle (initial):", repr(e))
    traceback.print_exc()

SESSIONS = {}
POLITE_RE = re.compile(r"\b(thanks?|thank you|thx|ty)\b", re.I)


#debug endpoints
@app.route("/api/files", methods=["GET"])
def api_files():
    try:
        files = sorted(os.listdir("."))
        expected = ["doctors_with_pin.csv","doctors.csv","chat_transcripts_seed.csv", "knowledge_base.csv", "ivf_chatbot_flat.csv"]
        file_info = {f: {"exists": os.path.exists(f)} for f in expected}
        previews = {}
        for f in expected:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f, nrows=5)
                    previews[f] = {"columns": list(df.columns), "rows_preview": df.head(3).to_dict(orient="records")}
                    file_info[f]["size_bytes"] = os.path.getsize(f)
                except Exception as e:
                    previews[f] = {"error": repr(e)}
            else:
                previews[f] = None
        return jsonify({"ok": True, "repo_files": files, "file_info": file_info, "previews": previews})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": repr(e)},), 500


@app.route("/api/debug_full", methods=["GET"])
def api_debug_full():
    try:
        b = load_and_build_models(force_rebuild=False)
        summary = {}
        try:
            summary["type"] = str(type(b))
            if isinstance(b, dict):
                summary["keys"] = list(b.keys())
        except Exception as e:
            summary["summary_error"] = repr(e)
        return jsonify({"ok": True, "summary": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": repr(e)}), 500


#session management
def new_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "role": None,
        "patient_id": None,        
        "pending_patient_id": None,
        "doctor_id": None,     
        "pending_doctor_id": None, 
        "stage": "greet",          
        "history": []
    }
    print(f"SERVER: new session {sid}")
    return sid

def get_session(sid):
    return SESSIONS.get(sid)

def reset_session(sid):
    SESSIONS[sid] = {
        "role": None,
        "patient_id": None,
        "pending_patient_id": None,
        "doctor_id": None,
        "pending_doctor_id": None,
        "stage": "greet",
        "history": []
    }
    print(f"SERVER: session {sid} reset")
    return SESSIONS[sid]

def history_to_response(history):
    out = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append({"user": item[0] or "", "bot": item[1] or "", "confidence": float(item[2]) if len(item) > 2 else 0.0})
        elif isinstance(item, dict):
            out.append({"user": item.get("user",""), "bot": item.get("bot",""), "confidence": float(item.get("confidence",0.0))})
        else:
            out.append({"user": "", "bot": str(item), "confidence": 0.0})
    return out

def make_bot_message(text, quick_replies=None):
    safe = html.escape(text).replace("\n", "<br>")
    if not quick_replies:
        return safe
    buttons = " ".join(
        f'<button class="qr" data-payload="{html.escape(payload)}">{html.escape(label)}</button>'
        for label, payload in quick_replies
    )
    return f"{safe}<div class='quick-replies'>{buttons}</div>"

def normalize_model_response(resp):
    try:
        if resp is None:
            return ""
        if isinstance(resp, dict):
            if "answer" in resp:
                return str(resp["answer"] or "")
            for k in ("text","response","output"):
                if k in resp:
                    return str(resp[k] or "")
            return str(resp)
        if isinstance(resp, str):
            return resp
        return str(resp)
    except Exception as e:
        print("SERVER: normalize_model_response error:", e)
        traceback.print_exc()
        return str(resp)

def safe_answer_call(prompt, role, bundle_obj, session_patient_id=None):
    print("SERVER: safe_answer_call START", {"role": role, "patient_id": session_patient_id})
    try:
        try:
            resp = answer_message_for_role(prompt, role, bundle_obj, session_patient_id=session_patient_id)
            print("SERVER: model returned (named arg) type:", type(resp))
        except TypeError as te:
            print("SERVER: TypeError with named arg:", repr(te))
            resp = answer_message_for_role(prompt, role, bundle_obj, session_patient_id)
            print("SERVER: model returned (positional) type:", type(resp))
    except Exception as e:
        print("SERVER: exception while calling answer_message_for_role:", repr(e))
        traceback.print_exc()
        return (f"Error: model call failed: {e}", {"error": str(e)})

    try:
        rrepr = repr(resp)
        if len(rrepr) > 2000:
            rrepr = rrepr[:2000] + "...[truncated]"
        print("SERVER: raw model resp repr (truncated):", rrepr)
    except Exception as e:
        print("SERVER: could not repr(resp):", e)

    try:
        ans = normalize_model_response(resp)
        print("SERVER: normalized answer type:", type(ans), "len:", len(ans) if ans is not None else 0)
    except Exception as e:
        print("SERVER: normalization exception:", repr(e))
        traceback.print_exc()
        return (f"Error: normalization failed: {e}", {"error": str(e)})

    print("SERVER: safe_answer_call END")
    return (ans, resp)

def extract_confidence_from_raw(resp):
    try:
        if resp is None:
            return 0.0
        if isinstance(resp, dict):
            if "confidence" in resp:
                return float(resp.get("confidence", 0.0) or 0.0)
            if "score" in resp:
                return float(resp.get("score", 0.0) or 0.0)
        return 0.0
    except Exception:
        return 0.0

#message processing
def process_message(sid, message):
    sess = get_session(sid)
    if not sess:
        raise ValueError("invalid session")
    u = (message or "").strip()
    if not u:
        return history_to_response(sess["history"])

    sess["history"].append({"user": u, "bot": "", "confidence": 0.0})
    lw = u.lower()

    # Simple role selection via initial messages
    if sess["stage"] in ("greet","role_select") and any(k in lw for k in ("patient","doctor","guest")):
        if "patient" in lw:
            sess["role"] = "Patient"; sess["stage"] = "role_confirmed"
            sess["history"][-1]["bot"] = "Welcome Patient — may I have your Patient ID (e.g., P10001)?"
            return history_to_response(sess["history"])
        if "doctor" in lw:
            sess["role"] = "Doctor"; sess["stage"] = "role_confirmed"
            sess["history"][-1]["bot"] = "Welcome Doctor — may I have your Doctor ID (e.g., D100)?"
            return history_to_response(sess["history"])
        if "guest" in lw:
            sess["role"] = "Guest"; sess["stage"] = "ready"
            sess["history"][-1]["bot"] = "Welcome Guest — how can I help with IVF information?"
            return history_to_response(sess["history"])

    try:
        role = sess.get("role") or "Guest"

        #PATIENT
        if role == "Patient":
            # If role_confirmed -> expect patient ID, then ask PIN
            if sess["stage"] == "role_confirmed":
                pid = extract_patientid_from_text(u)
                if pid:
                    sess["pending_patient_id"] = pid.upper()
                    sess["stage"] = "awaiting_pin"
                    sess["history"][-1]["bot"] = "Please provide your 4-character PIN to view your details."
                    return history_to_response(sess["history"])
                else:
                    sess["history"][-1]["bot"] = "Please provide your Patient ID (e.g., P10001)."
                    return history_to_response(sess["history"])

            if sess["stage"] == "awaiting_pin":
                pending = sess.get("pending_patient_id")
                if not pending:
                    sess["stage"] = "role_confirmed"
                    sess["history"][-1]["bot"] = "Please provide your Patient ID (e.g., P10001)."
                    return history_to_response(sess["history"])

                provided_pin = u.strip()
                patient_rec = bundle.get("patient_lookup", {}).get(pending)
                if not patient_rec:
                    sess["pending_patient_id"] = None
                    sess["stage"] = "role_confirmed"
                    sess["history"][-1]["bot"] = f"Could not find records for {pending}. Please provide a valid Patient ID (e.g., P10001)."
                    return history_to_response(sess["history"])

                expected_pin = get_patient_pin_field(patient_rec)
                if expected_pin is None:
                    sess["pending_patient_id"] = None
                    sess["stage"] = "role_confirmed"
                    sess["history"][-1]["bot"] = "No PIN is set for this account. Please contact clinic support."
                    return history_to_response(sess["history"])

                if provided_pin.strip().upper() == expected_pin.strip().upper():
                    sess["patient_id"] = pending
                    sess["pending_patient_id"] = None
                    sess["stage"] = "ready"
                    ans, raw = safe_answer_call("Show my patient summary", "Patient", bundle, session_patient_id=sess["patient_id"])
                    if ans:
                        ans = ans + "\n\nThank you. Glad I can help you"
                    sess["history"][-1]["bot"] = ans
                    sess["history"][-1]["confidence"] = extract_confidence_from_raw(raw)
                    return history_to_response(sess["history"])
                else:
                    sess["history"][-1]["bot"] = "PIN is wrong. Please try again."
                    sess["history"][-1]["confidence"] = 0.0
                    return history_to_response(sess["history"])

            if sess["stage"] == "ready" and sess.get("patient_id"):
                pid = sess.get("patient_id")
                ans, raw = safe_answer_call(u, "Patient", bundle, session_patient_id=pid)
                if ans:
                    ans = ans + "\n\nThank you. Glad I can help you"
                sess["history"][-1]["bot"] = ans
                sess["history"][-1]["confidence"] = extract_confidence_from_raw(raw)
                return history_to_response(sess["history"])

        #DOCTOR
        if role == "Doctor":
            # If role_confirmed -> expect Doctor ID, then ask DocPIN
            if sess["stage"] == "role_confirmed":
                did = extract_doctorid_from_text(u)
                if did:
                    sess["pending_doctor_id"] = did.upper()
                    sess["stage"] = "awaiting_doc_pin"
                    sess["history"][-1]["bot"] = "Please provide your 4-digit doctor PIN to authenticate."
                    return history_to_response(sess["history"])
                else:
                    sess["history"][-1]["bot"] = "Please provide your Doctor ID (e.g., D100)."
                    return history_to_response(sess["history"])

            if sess["stage"] == "awaiting_doc_pin":
                pending = sess.get("pending_doctor_id")
                if not pending:
                    sess["stage"] = "role_confirmed"
                    sess["history"][-1]["bot"] = "Please provide your Doctor ID (e.g., D100)."
                    return history_to_response(sess["history"])

                provided_pin = u.strip()
                doctor_rec = None
                #check doctors lookup in bundle if present
                if bundle and isinstance(bundle, dict):
                    doctor_rec = bundle.get("doctors_lookup", {}).get(pending) or bundle.get("doctors_lookup", {}).get(pending.lstrip("D"))
                #fallback for backup
                if not doctor_rec and bundle.get("doctors") is not None:
                    try:
                        df = bundle.get("doctors")
                        #try exact match on doctor_id column
                        if 'doctor_id' in df.columns:
                            row = df[df['doctor_id'].astype(str).str.upper() == pending]
                            if len(row) > 0:
                                doctor_rec = row.iloc[0].to_dict()
                    except Exception:
                        pass

                if not doctor_rec:
                    sess["pending_doctor_id"] = None
                    sess["stage"] = "role_confirmed"
                    sess["history"][-1]["bot"] = f"Could not find records for {pending}. Please provide a valid Doctor ID (e.g., D100)."
                    return history_to_response(sess["history"])

                expected_doc_pin = get_doctor_pin_field(doctor_rec)
                if expected_doc_pin is None:
                    sess["pending_doctor_id"] = None
                    sess["stage"] = "role_confirmed"
                    sess["history"][-1]["bot"] = "No PIN is set for this doctor account. Please contact admin."
                    return history_to_response(sess["history"])

                if provided_pin.strip().upper() == expected_doc_pin.strip().upper():
                    sess["doctor_id"] = pending
                    sess["pending_doctor_id"] = None
                    sess["stage"] = "ready"
                    sess["history"][-1]["bot"] = "Authenticated as Doctor. You may now request patient data (e.g., 'Show P10001') or ask clinical questions."
                    sess["history"][-1]["confidence"] = 1.0
                    return history_to_response(sess["history"])
                else:
                    sess["history"][-1]["bot"] = "Doctor PIN is wrong. Please try again."
                    sess["history"][-1]["confidence"] = 0.0
                    return history_to_response(sess["history"])

            
            if sess["stage"] == "ready" and sess.get("doctor_id"):
                # If doctor includes a patient id, show patient info
                pid = extract_patientid_from_text(u)
                if pid:
                    patient = bundle["patient_lookup"].get(str(pid))
                    if not patient:
                        sess["history"][-1]["bot"] = f"No records found for {pid}."
                        sess["history"][-1]["confidence"] = 0.0
                        return history_to_response(sess["history"])
                    if hasattr(patient, "to_dict"):
                        patient = patient.to_dict()
                    clean_patient = {}
                    for k, v in patient.items():
                        clean_patient[k] = "-" if pd.isna(v) else v
                    lines = [
                        f"Patient ID: {clean_patient.get('PatientID', '-')}",
                        f"Record Date: {str(clean_patient.get('record_date', '-'))[:10]}",
                        f"IVF Result: {clean_patient.get('ivf_result', '-')}",
                        f"Sperm Result: {clean_patient.get('sperm_result', '-')}",
                        f"Embryo Grade: {clean_patient.get('embryo_grade', '-')}",
                        f"AMH: {clean_patient.get('AMH', '-')}",
                        f"FSH: {clean_patient.get('FSH', '-')}",
                        f"Age: {clean_patient.get('age', '-')}",
                        f"Doctor ID: {clean_patient.get('doctor_id', '-')}",
                        f"Notes: {clean_patient.get('notes', '-')}",
                    ]
                    formatted = "\n".join(lines)
                    sess["history"][-1]["bot"] = formatted
                    sess["history"][-1]["confidence"] = 1.0
                    return history_to_response(sess["history"])

                
                ans, raw = safe_answer_call(u, "Doctor", bundle)
                if ans:
                    ans = ans + "\n\nThank you. Glad I can help you"
                sess["history"][-1]["bot"] = ans
                sess["history"][-1]["confidence"] = extract_confidence_from_raw(raw)
                return history_to_response(sess["history"])

        #GUEST
        ans, raw = safe_answer_call(u, "Guest", bundle)
        if ans:
            ans = ans + "\n\nThank you. Glad I can help you"
        sess["history"][-1]["bot"] = ans
        sess["history"][-1]["confidence"] = extract_confidence_from_raw(raw)
        return history_to_response(sess["history"])

    except Exception as e:
        print("SERVER: unexpected error in process_message:", repr(e))
        traceback.print_exc()
        sess["history"][-1]["bot"] = f"Server error: {e}"
        sess["history"][-1]["confidence"] = 0.0
        return history_to_response(sess["history"])


#endpoints
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    sid = new_session()
    sess = get_session(sid)
    welcome = make_bot_message("Hello! I'm the IVF Support Assistant. 👋\nHow can I help you today?",
                              quick_replies=[("I'm a Patient","Patient"),("I'm a Doctor","Doctor"),("I'm a Guest","Guest")])
    sess["history"].append({"user": "", "bot": welcome, "confidence": 1.0})
    return jsonify({"session_id": sid, "history": history_to_response(sess["history"]), "role": sess["role"]})


@app.route("/api/role", methods=["POST"])
def api_role():
    data = request.json or {}
    sid = data.get("session_id")
    role = data.get("role")
    if not sid or sid not in SESSIONS:
        return jsonify({"error":"invalid session"}), 400
    sess = get_session(sid)
    sess["role"] = role
    # reset auth/pending on role change
    sess["patient_id"] = None
    sess["pending_patient_id"] = None
    sess["doctor_id"] = None
    sess["pending_doctor_id"] = None
    if role == "Patient":
        sess["stage"] = "role_confirmed"
        bot = "Welcome Patient — may I have your Patient ID (e.g., P10001)?"
    elif role == "Doctor":
        sess["stage"] = "role_confirmed"
        bot = "Welcome Doctor — may I have your Doctor ID (e.g., D100)?"
    else:
        sess["stage"] = "ready"
        bot = "Welcome Guest — how can I help with IVF information?"
    sess["history"].append({"user": "", "bot": bot, "confidence": 1.0})
    return jsonify({"ok": True, "history": history_to_response(sess["history"]), "role": sess["role"]})


@app.route("/api/message", methods=["POST"])
def api_message():
    data = request.json or {}
    sid = data.get("session_id")
    message = data.get("message","")
    if not sid or sid not in SESSIONS:
        return jsonify({"error":"invalid session"}), 400
    history = process_message(sid, message)
    sess = get_session(sid)
    return jsonify({"history": history, "role": sess.get("role"), "patient_id": sess.get("patient_id"), "doctor_id": sess.get("doctor_id")})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.json or {}
    sid = data.get("session_id")
    if not sid or sid not in SESSIONS:
        return jsonify({"error":"invalid session"}), 400
    reset_session(sid)
    sess = get_session(sid)
    welcome = make_bot_message("Conversation reset. Hello! I'm the IVF Support Assistant. 👋\nHow can I help you today?",
                              quick_replies=[("I'm a Patient","Patient"),("I'm a Doctor","Doctor"),("I'm a Guest","Guest")])
    sess["history"].append({"user": "", "bot": welcome, "confidence": 1.0})
    return jsonify({"ok": True, "history": history_to_response(sess["history"]), "role": sess["role"]})


@app.route("/api/debug", methods=["GET"])
def api_debug():
    try:
        model_ok = bundle is not None
        return jsonify({"model_loaded": model_ok, "sessions": list(SESSIONS.keys()), "session_count": len(SESSIONS)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("SERVER: running on port", port)
    app.run(host="0.0.0.0", port=port, debug=False)
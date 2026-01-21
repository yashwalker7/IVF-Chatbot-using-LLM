import os
import pickle
import re
import traceback
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#Chroma and embedding libs
try:
    import chromadb
    from chromadb.config import Settings  # optional
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

#CSV import
ROOT = os.path.abspath(".")
print("models.py: root path:", ROOT)

#Minimum acceptable KB match score (for TF-IDF we treat score ~ (1 - cosine_distance))
#If top score < MIN_KB_SCORE, we treat query as out-of-scope and refuse to answer.
MIN_KB_SCORE = 0.25

def _find_file(candidates):
    for name in candidates:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            return p
    return None

all_csvs = [f for f in os.listdir(ROOT) if f.lower().endswith(".csv")]

FILES = {
    "doctors": _find_file(["doctors_with_pin.csv","doctors.csv", "doctor_list.csv", "doctors_list.csv", "doctors_data.csv"]),
    "chat_transcripts": _find_file(["chat_transcripts_seed.csv", "transcripts.csv", "chat_transcripts.csv", "chats.csv"]),
    "knowledge_base": _find_file(["knowledge_base.csv", "kb.csv", "knowledgebase.csv", "faq.csv"]),
    "ivf_flat": _find_file(["ivf_chatbot_flat.csv", "patients.csv", "patients_flat.csv", "ivf_flat.csv", "ivf.csv"]),
}

#heuristics to fill missing entries
if not FILES["knowledge_base"]:
    for cand in all_csvs:
        if any(k in cand.lower() for k in ("kb", "knowledge", "faq", "qa", "knowledge_base")):
            FILES["knowledge_base"] = os.path.join(ROOT, cand); break

if not FILES["ivf_flat"]:
    for cand in all_csvs:
        if any(k in cand.lower() for k in ("patient", "patients", "ivf", "ivf_chatbot")):
            FILES["ivf_flat"] = os.path.join(ROOT, cand); break
    if not FILES["ivf_flat"] and len(all_csvs) > 0:
        FILES["ivf_flat"] = os.path.join(ROOT, all_csvs[0])

if not FILES["chat_transcripts"]:
    for cand in all_csvs:
        if any(k in cand.lower() for k in ("trans", "chat")):
            FILES["chat_transcripts"] = os.path.join(ROOT, cand); break

if not FILES["doctors"]:
    for cand in all_csvs:
        if "doctor" in cand.lower():
            FILES["doctors"] = os.path.join(ROOT, cand); break

print("models.py: detected FILES mapping:")
for k, v in FILES.items():
    print("  ", k, "->", v)

#artifacts
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CHROMA_DIR = os.path.join(ROOT, "chroma_db")
CHROMA_COLLECTION_NAME = "kb"

#helpers
def read_csv_with_fallback(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    print(f"read_csv_with_fallback: failed to read {path} with standard encodings")
    return None

def save_pickle(obj, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        print("save_pickle error:", e)

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print("load_pickle error:", e)
    return None

def safe_load_or_synth(path: str, synth_fn):
    df = read_csv_with_fallback(path)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return synth_fn()
    return df

#synthesizers
def synthesize_ivf_wide(n=30):
    import random
    from datetime import datetime
    rows = []
    for i in range(n):
        pid = f"P{10000+i}"
        rows.append({
            "PatientID": pid,
            "record_date": datetime.now().strftime("%Y-%m-%d"),
            "ivf_result": "unknown",
            "sperm_result": "not_tested",
            "embryo_grade": "N/A",
            "AMH": 1.0,
            "FSH": 6.0,
            "age": 30,
            "doctor_id": "D100",
            "notes": ""
        })
    return pd.DataFrame(rows)

def synthesize_kb():
    return pd.DataFrame([{"question": "What is IVF?", "answer": "IVF involves lab fertilization."}])

def synthesize_transcripts():
    return pd.DataFrame([{"text": "What is IVF?", "intent": "general_info"}])

def first_text_column(df: pd.DataFrame, candidates: List[str] = None) -> Optional[str]:
    if df is None or df.empty:
        return None
    if candidates is None:
        candidates = ["text", "message", "utterance", "content", "query", "chat"]
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return c
    return df.columns[0]

def kb_qa_columns(df: pd.DataFrame):
    if df is None or df.empty:
        return (None, None)
    lower = {c.lower(): c for c in df.columns}
    q = None; a = None
    for qc in ["question", "q", "prompt", "ask"]:
        if qc in lower:
            q = lower[qc]; break
    for ac in ["answer", "a", "response", "reply"]:
        if ac in lower:
            a = lower[ac]; break
    # fallback pick string columns
    if not q:
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]):
                q = c; break
    if not a:
        strcols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        if len(strcols) >= 2:
            a = strcols[1]
    return (q, a)

def normalize_ivf_wide(ivf: pd.DataFrame) -> pd.DataFrame:
    if ivf is None:
        return ivf
    df = ivf.copy()
    # normalize PatientID
    if "PatientID" not in df.columns:
        for c in df.columns:
            if "patient" in c.lower() and "id" in c.lower():
                df = df.rename(columns={c: "PatientID"})
                break
    # ensure required columns
    required = ["PatientID", "record_date", "ivf_result", "sperm_result", "embryo_grade", "AMH", "FSH", "age"]
    for c in required:
        if c not in df.columns:
            if c in ("AMH", "FSH", "age"):
                df[c] = 0
            else:
                df[c] = ""
    try:
        df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    except Exception:
        pass
    try:
        df = df.sort_values("PatientID").reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)
    return df

#Chroma init
def init_chroma_and_embedder(kb_questions: List[str]):
    if not CHROMA_AVAILABLE:
        print("init_chroma: chromadb or sentence-transformers not importable.")
        return None, None, None

    client = None
    try:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
        print("init_chroma: client via Settings() succeeded")
    except Exception as e1:
        try:
            client = chromadb.Client(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR)
            print("init_chroma: client via fallback signature succeeded")
        except Exception as e2:
            print("init_chroma: failed to construct chroma client:", e1, e2)
            return None, None, None

    try:
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        try:
            collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
        except Exception as e:
            print("init_chroma: failed to create/get collection:", e)
            return None, None, None

    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print("init_chroma: failed to load embedder:", e)
        return None, None, None

    try:
        if kb_questions:
            ids = [str(i) for i in range(len(kb_questions))]
            embs = embedder.encode(kb_questions, convert_to_numpy=True)
            metadatas = [{"source_idx": i} for i in range(len(kb_questions))]
            try:
                collection.upsert(ids=ids, documents=kb_questions, embeddings=embs.tolist(), metadatas=metadatas)
            except Exception:
                # try alternative shape
                try:
                    collection.upsert(ids=ids, documents=kb_questions, embeddings=[e.tolist() for e in embs], metadatas=metadatas)
                except Exception as e:
                    print("init_chroma: upsert failed:", e)
            try:
                client.persist()
            except Exception:
                pass
            print("init_chroma: upserted KB size:", len(kb_questions))
    except Exception as e:
        print("init_chroma: upsert error:", e)
        return None, None, None

    return client, collection, embedder

#retrieval
def retrieve_kb_answer_chroma(query: str, bundle, top_k: int = 1):
    coll = bundle.get("chroma_collection")
    embedder = bundle.get("chroma_embedder")
    if coll is None or embedder is None:
        return []
    try:
        q_emb = embedder.encode([query], convert_to_numpy=True).tolist()
    except Exception as e:
        print("retrieve_kb_answer_chroma: embedding error:", e)
        return []
    try:
        res = coll.query(query_embeddings=q_emb, n_results=top_k)
    except Exception as e:
        print("retrieve_kb_answer_chroma: query exception:", e)
        return []
    out = []
    try:
        docs = res.get("documents", [[]])[0] if isinstance(res, dict) else getattr(res, "documents", [[]])[0]
    except Exception:
        docs = []
    try:
        dists = res.get("distances", [[]])[0] if isinstance(res, dict) else getattr(res, "distances", [[]])[0]
    except Exception:
        dists = []
    try:
        ids = res.get("ids", [[]])[0] if isinstance(res, dict) else getattr(res, "ids", [[]])[0]
    except Exception:
        ids = []
    try:
        metas = res.get("metadatas", [[]])[0] if isinstance(res, dict) and "metadatas" in res else (getattr(res, "metadatas", [[]])[0] if hasattr(res, "metadatas") else [])
    except Exception:
        metas = []

    for i, doc in enumerate(docs):
        idx = None
        try:
            idx = int(ids[i])
        except Exception:
            if metas and i < len(metas) and isinstance(metas[i], dict):
                idx = metas[i].get("source_idx")
        ans = ""
        if idx is None:
            try:
                ans_idx = bundle.get("kb_qs", []).index(doc)
                ans = bundle.get("kb_as", [])[ans_idx]
            except Exception:
                ans = ""
        else:
            if idx < len(bundle.get("kb_as", [])):
                ans = bundle.get("kb_as", [])[idx]
        score = float(dists[i]) if i < len(dists) else 0.0
        out.append({"question": doc, "answer": ans, "score": score, "source": "chroma"})
    return out

def retrieve_kb_answer_tfidf(query: str, bundle, top_k: int = 1):
    from sklearn.neighbors import NearestNeighbors
    kb_qs = bundle.get("kb_qs", [])
    if not kb_qs:
        return []
    try:
        vec = bundle["vectorizer"].transform([query])
        kb_vecs = bundle.get("kb_vecs")
        if kb_vecs is None:
            kb_vecs = bundle["vectorizer"].transform(kb_qs)
        nn = NearestNeighbors(n_neighbors=min(top_k, max(1, kb_vecs.shape[0])), metric="cosine").fit(kb_vecs)
        dists, idxs = nn.kneighbors(vec, n_neighbors=min(top_k, kb_vecs.shape[0]))
    except Exception as e:
        print("retrieve_kb_answer_tfidf: NN failed:", e)
        return []
    out = []
    for sc, idx in zip(dists[0], idxs[0]):
        score = float(1 - sc) if sc is not None else 0.0
        out.append({"question": kb_qs[idx], "answer": bundle["kb_as"][idx], "score": score, "source": "tfidf"})
    return out

#utilities
def extract_patientid_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(r"\bP\d{4,6}\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()
    m2 = re.search(r"\b(\d{4,6})\b", text)
    if m2:
        return "P" + m2.group(1)
    return None

def extract_doctorid_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(r"\bD\d{2,6}\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()
    return None

def get_patient_pin_field(patient_record):
    if not isinstance(patient_record, dict):
        return None
    for candidate in ("PIN","pin","Pin","pin_code","pinCode","passcode","Passcode"):
        if candidate in patient_record and patient_record[candidate] is not None and str(patient_record[candidate]).strip() != "":
            return str(patient_record[candidate]).strip()
    return None

def get_doctor_pin_field(doctor_record):
    if not isinstance(doctor_record, dict):
        return None
    for cand in ("DocPIN","docpin","doc_pin","Pin","PIN","pin","PINCode","DocPin"):
        if cand in doctor_record and doctor_record[cand] is not None and str(doctor_record[cand]).strip() != "":
            return str(doctor_record[cand]).strip()
    return None

def classify_intent(text: str, bundle):
    try:
        v = bundle["vectorizer"].transform([text])
        p = bundle["clf"].predict(v)
        return bundle["le"].inverse_transform(p)[0]
    except Exception:
        return "unknown"

# model loading
def load_and_build_models(force_rebuild: bool = False):
    cached = load_pickle(os.path.join(MODELS_DIR, "model_bundle.pkl"))
    if cached and not force_rebuild:
        print("models.py: Loaded cached model bundle.")
        return cached

    ivf = safe_load_or_synth(FILES.get("ivf_flat"), lambda: synthesize_ivf_wide(30))
    kb = safe_load_or_synth(FILES.get("knowledge_base"), synthesize_kb)
    transcripts = safe_load_or_synth(FILES.get("chat_transcripts"), synthesize_transcripts)
    doctors = safe_load_or_synth(FILES.get("doctors"), lambda: pd.DataFrame())

    try:
        def preview(df, name):
            if df is None:
                print(f"models.py: {name} -> None")
                return
            try:
                print(f"models.py: {name} columns ->", list(df.columns)[:10])
                print(f"models.py: {name} preview rows ->")
                print(df.head(2).to_dict(orient="records"))
            except Exception:
                pass
        preview(ivf, "ivf")
        preview(kb, "kb")
        preview(transcripts, "transcripts")
        preview(doctors, "doctors")
    except Exception:
        pass

    #normalize
    text_col = first_text_column(transcripts)
    if text_col and text_col != "text":
        transcripts = transcripts.rename(columns={text_col: "text"})
    if "intent" not in transcripts.columns:
        transcripts["intent"] = "general_info"

    qcol, acol = kb_qa_columns(kb)
    if qcol and qcol != "question":
        kb = kb.rename(columns={qcol: "question"})
    if acol and acol != "answer":
        kb = kb.rename(columns={acol: "answer"})
    if "question" not in kb.columns:
        kb["question"] = ""
    if "answer" not in kb.columns:
        kb["answer"] = ""

    kb_qs = kb["question"].astype(str).tolist() if (kb is not None and not kb.empty) else []
    kb_as = kb["answer"].astype(str).tolist() if (kb is not None and not kb.empty) else []

    print("load_and_build_models: KB entries loaded:", len(kb_qs))

    combined = list(transcripts["text"].astype(str).tolist()) + kb_qs
    if not combined:
        combined = ["empty"]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    vectorizer.fit(combined)

    if kb_qs:
        kb_vecs = vectorizer.transform(kb_qs)
        print("load_and_build_models: precomputed TF-IDF KB vectors shape:", getattr(kb_vecs, "shape", None))
    else:
        kb_vecs = None
        print("load_and_build_models: no KB questions found to vectorize.")

    try:
        X = vectorizer.transform(transcripts["text"].astype(str).values)
        le = LabelEncoder()
        y = le.fit_transform(transcripts["intent"].astype(str).values)
        if len(np.unique(y)) == 1:
            transcripts = pd.concat([transcripts, pd.DataFrame([{"text": "placeholder", "intent": "other"}])], ignore_index=True)
            X = vectorizer.transform(transcripts["text"].astype(str).values)
            y = le.fit_transform(transcripts["intent"].astype(str).values)
        clf = LogisticRegression(max_iter=1000).fit(X, y)
    except Exception as e:
        print("models.py: classifier build failed:", e)
        clf = None
        le = None

    ivf = normalize_ivf_wide(ivf)
    if "PatientID" not in ivf.columns:
        print("models.py: Warning - PatientID column missing in ivf table")
        ivf["PatientID"] = ivf.index.map(lambda i: f"P{10000+i}")

    ivf["PatientID"] = ivf["PatientID"].astype(str).str.strip().str.upper()

    if "record_date" in ivf.columns:
        try:
            ivf = ivf.sort_values("record_date").drop_duplicates(subset=["PatientID"], keep="last").reset_index(drop=True)
        except Exception:
            ivf = ivf.drop_duplicates(subset=["PatientID"], keep="last").reset_index(drop=True)
    else:
        ivf = ivf.drop_duplicates(subset=["PatientID"], keep="last").reset_index(drop=True)

    patient_lookup = {}
    for _, row in ivf.iterrows():
        pid = str(row.get("PatientID", "")).strip().upper()
        if not pid:
            continue
        patient_lookup[pid] = row.to_dict()
        if pid.startswith("P") and pid[1:].isdigit():
            patient_lookup[pid[1:]] = row.to_dict()
        else:
            if pid.isdigit():
                patient_lookup["P" + pid] = row.to_dict()

    #doctors lookup
    doctors_lookup = {}
    try:
        if isinstance(doctors, pd.DataFrame) and not doctors.empty:
            for _, r in doctors.iterrows():
                did = None
                for col in r.index:
                    if 'doctor' in str(col).lower() and 'id' in str(col).lower():
                        did = r[col]
                        break
                if not did:
                    did = r.get('doctor_id') or r.get('DoctorID') or None
                if did:
                    did = str(did).strip().upper()
                    doctors_lookup[did] = r.to_dict()
                    if did.startswith("D") and did[1:].isdigit():
                        doctors_lookup[did[1:]] = r.to_dict()
    except Exception:
        pass

    print("load_and_build_models: Sample patient IDs loaded:", list(patient_lookup.keys())[:10])
    print("load_and_build_models: Sample doctor IDs loaded:", list(doctors_lookup.keys())[:6])

    chroma_client = chroma_collection = chroma_embedder = None
    if CHROMA_AVAILABLE and kb_qs:
        try:
            chroma_client, chroma_collection, chroma_embedder = init_chroma_and_embedder(kb_qs)
            if chroma_collection is not None:
                print("load_and_build_models: Chroma initialized successfully.")
            else:
                print("load_and_build_models: Chroma init returned no collection.")
        except Exception as e:
            print("load_and_build_models: Chroma init raised exception:", e)
            traceback.print_exc()
            chroma_client = chroma_collection = chroma_embedder = None
    else:
        if not CHROMA_AVAILABLE:
            print("load_and_build_models: Chromadb not installed; skipping Chroma init.")
        else:
            print("load_and_build_models: KB empty or missing, skipping Chroma init.")

    bundle = {
        "ivf": ivf,
        "patient_lookup": patient_lookup,
        "kb": kb,
        "transcripts": transcripts,
        "vectorizer": vectorizer,
        "clf": clf,
        "le": le,
        "kb_qs": kb_qs,
        "kb_as": kb_as,
        "kb_vecs": kb_vecs,
        "doctors": doctors,
        "doctors_lookup": doctors_lookup,
        "chroma_client": chroma_client,
        "chroma_collection": chroma_collection,
        "chroma_embedder": chroma_embedder,
    }

    try:
        save_pickle(bundle, os.path.join(MODELS_DIR, "model_bundle.pkl"))
        print("models.py: Saved model bundle.")
    except Exception as e:
        print("models.py: Failed to save model bundle:", e)

    return bundle

#reply function
def answer_message_for_role(message: str, role: str, bundle, session_patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Restored reply logic:
     - role in {"Patient","Doctor","Guest"}
     - Patient: needs session_patient_id or embedded P####; fetches fields or summary
     - Doctor: if patient id included shows patient table; otherwise falls back to KB retrieval
     - Guest: KB via chroma or tfidf
    Added: explicit greeting & thanks detection + KB score threshold to avoid hallucination.
    """
    msg = (message or "").strip()
    if not msg:
        return {"answer": "", "confidence": 0.0, "source": None, "intent": "empty"}

    lower = msg.lower()

    greeting_pattern = re.compile(r"\b(hi|hello|hey|hey there|good morning|good afternoon|good evening)\b", re.I)
    thanks_pattern = re.compile(r"\b(thank(s)?|thank you|thanks a lot|thx|ty)\b", re.I)

    if greeting_pattern.search(lower):
        return {"answer": "Hello! I'm the IVF Support Assistant. 👋 How can I help you today?", "confidence": 1.0, "source": "meta", "intent": "greeting"}

    if thanks_pattern.search(lower):
        return {"answer": "You're welcome — glad I could help.", "confidence": 1.0, "source": "meta", "intent": "gratitude"}

    intent = classify_intent(msg, bundle) if bundle and bundle.get("clf") else "general_info"
    result = {"answer": "", "confidence": 0.0, "source": None, "intent": intent}

    #PATIENT
    if role == "Patient":
        pid = (session_patient_id or extract_patientid_from_text(message) or "").strip()
        pid = pid.upper() if pid else pid

        def lookup_patient(pid_candidate):
            if not pid_candidate:
                return None
            pnorm = pid_candidate.strip().upper()
            if pnorm in bundle["patient_lookup"]:
                return bundle["patient_lookup"][pnorm]
            if pnorm.startswith("P") and pnorm[1:] in bundle["patient_lookup"]:
                return bundle["patient_lookup"][pnorm[1:]]
            if pnorm.isdigit() and ("P" + pnorm) in bundle["patient_lookup"]:
                return bundle["patient_lookup"]["P" + pnorm]
            for k in bundle["patient_lookup"].keys():
                if k.endswith(pnorm) or pnorm.endswith(k):
                    return bundle["patient_lookup"][k]
            return None

        patient = lookup_patient(pid)

        if not patient:
            result["answer"] = "Please provide a valid Patient ID (e.g., P10001). I couldn't find the ID you provided."
            result["confidence"] = 0.0
            return result

        def present_field(k):
            v = patient.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                return "Not available"
            return v

        top_line = (
            f"Latest IVF result: {present_field('ivf_result')}\n"
            f"Age: {present_field('age')}\n"
            f"AMH: {present_field('AMH')}\n"
            f"FSH: {present_field('FSH')}"
        )

        q = (message or "").lower()
        if "amh" in q:
            result["answer"] = f"AMH for {patient.get('PatientID','')}: {present_field('AMH')}"
            result["confidence"] = 0.95
            result["source"] = "patient_table"
            return result
        if "fsh" in q:
            result["answer"] = f"FSH for {patient.get('PatientID','')}: {present_field('FSH')}"
            result["confidence"] = 0.95
            result["source"] = "patient_table"
            return result
        if "sperm" in q or "semen" in q:
            result["answer"] = f"Sperm result for {patient.get('PatientID','')}: {present_field('sperm_result')}"
            result["confidence"] = 0.95
            result["source"] = "patient_table"
            return result
        if "embryo" in q or "grade" in q:
            result["answer"] = f"Embryo Grade for {patient.get('PatientID','')}: {present_field('embryo_grade')}"
            result["confidence"] = 0.95
            result["source"] = "patient_table"
            return result

        notes = patient.get("notes", "")
        result["answer"] = top_line + ("\n\nNotes: " + str(notes) if notes and str(notes).strip() else "")
        result["confidence"] = 0.9
        result["source"] = "patient_table"
        return result

    #DOCTOR
    if role == "Doctor":
        pid = extract_patientid_from_text(message)
        if pid:
            patient = bundle["patient_lookup"].get(str(pid))
            if not patient:
                return {"answer": f"No records found for {pid}.", "confidence": 0.0}
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
            return {"answer": formatted, "confidence": 1.0, "source": "patient_table"}
        # fallback to KB
        hits = retrieve_kb_answer_chroma(message, bundle, top_k=1) if bundle.get("chroma_collection") else retrieve_kb_answer_tfidf(message, bundle, top_k=1)
        if not hits or len(hits) == 0 or (hits[0].get("score", 0.0) < MIN_KB_SCORE):
            # treat as out-of-scope
            return {"answer": "Please ask IVF related questions only. Thank you", "confidence": 0.0, "source": "meta", "intent": "out_of_scope"}
        return {"answer": hits[0]["answer"], "confidence": hits[0]["score"], "source": hits[0]["source"]}

    #GUEST
    if role == "Guest":
        hits = retrieve_kb_answer_chroma(message, bundle, top_k=1) if bundle.get("chroma_collection") else retrieve_kb_answer_tfidf(message, bundle, top_k=1)
        if not hits or len(hits) == 0:
            return {"answer": "Please ask IVF related questions only. Thank you", "confidence": 0.0, "source": "meta", "intent": "out_of_scope"}
        top = hits[0]
        if top.get("score", 0.0) < MIN_KB_SCORE:
            return {"answer": "Please ask IVF related questions only. Thank you", "confidence": 0.0, "source": "meta", "intent": "out_of_scope"}
        if top.get("answer"):
            return {"answer": top["answer"], "confidence": top["score"], "source": top.get("source")}
        return {"answer": "I don't have that information in the knowledge base. Please contact the clinic.", "confidence": 0.0}

    hits = retrieve_kb_answer_chroma(message, bundle, top_k=1) if bundle.get("chroma_collection") else retrieve_kb_answer_tfidf(message, bundle, top_k=1)
    if not hits or hits[0].get("score", 0.0) < MIN_KB_SCORE:
        return {"answer": "Please ask IVF related questions only. Thank you", "confidence": 0.0, "source": "meta", "intent": "out_of_scope"}
    return {"answer": hits[0]["answer"], "confidence": hits[0]["score"], "source": hits[0]["source"]}
import streamlit as st
import sqlite3
import hashlib
import base64
from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO, StringIO
import csv
import os
import re
import shutil
import tempfile
import streamlit.components.v1 as components
import json
import urllib.request
from urllib.parse import quote as urlquote

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

from severity import assess_severity
from cost_estimation import estimate_repair_cost, estimate_from_detections as get_estimate, LABOUR_RATE


# ============================================================
# AUTOVISION AI - FUTURISTIC MULTI-PORTAL STREAMLIT APPLICATION
# Existing database and trained model are retained.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "vehicle_damage.db"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
HERO_SVG_PATH = BASE_DIR / "static" / "hero_car.svg"
REFERENCE_HOME_PATH = BASE_DIR / "static" / "reference_home.png"
HOME_HERO_PATH = BASE_DIR / "static" / "hero_red_scene.jpg"
CUSTOMER_CARD_PATH = BASE_DIR / "static" / "customer_card_scene.jpg"
MECHANIC_CARD_PATH = BASE_DIR / "static" / "mechanic_card_scene.jpg"
QUOTES_CARD_PATH = BASE_DIR / "static" / "quotes_card_scene.jpg"

st.set_page_config(
    page_title="AI Vehicle Damage Inspection System",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------- DATABASE -------------------------

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=25)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Create missing tables without dropping or rebuilding existing data."""
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS mechanics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            area TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer TEXT,
            phone TEXT,
            area TEXT,
            vehicle TEXT,
            damage TEXT,
            severity TEXT,
            min_cost INTEGER,
            max_cost INTEGER,
            image BLOB,
            created TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER NOT NULL,
            mechanic_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            details TEXT,
            selected INTEGER DEFAULT 0,
            UNIQUE(report_id, mechanic_id)
        )
    """)

    conn.commit()
    conn.close()


init_database()


def migrate_database():
    """Add new columns/tables safely. Existing data is never dropped."""
    conn = db_connect()
    def add(table, col, ddl):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
    add("reports", "repair_status", "TEXT DEFAULT 'Open'")
    for col in ("parts", "labour", "paint"):
        add("quotes", col, "REAL")
    add("quotes", "days", "INTEGER")
    add("quotes", "warranty", "TEXT")
    add("mechanics", "services", "TEXT")
    add("mechanics", "hours", "TEXT")
    conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER, mechanic_id INTEGER,
        rating INTEGER, comment TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER, mechanic_id INTEGER,
        sender TEXT, body TEXT, created TEXT)""")
    add("reports", "email", "TEXT")
    add("mechanics", "available", "INTEGER DEFAULT 1")
    add("mechanics", "price_list", "TEXT")
    add("quotes", "seen", "INTEGER DEFAULT 0")
    add("messages", "seen", "INTEGER DEFAULT 0")
    conn.execute("""CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT, report_id INTEGER, mechanic_id INTEGER,
        kind TEXT, image BLOB, created TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS garage (
        id INTEGER PRIMARY KEY AUTOINCREMENT, phone TEXT, vehicle TEXT, UNIQUE(phone, vehicle))""")
    conn.commit()
    conn.close()


migrate_database()


# ------------------------- HELPERS -------------------------

def hash_password(password):
    # Kept compatible with accounts created by the previous app.py.
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


try:
    import bcrypt
except ImportError:          # falls back to SHA-256 so the app still runs
    bcrypt = None


def hash_password_new(password):
    if bcrypt:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
    return hash_password(password)


def verify_password(password, stored):
    if not stored:
        return False
    if str(stored).startswith("$2"):
        try:
            return bool(bcrypt) and bcrypt.checkpw(password.encode("utf-8"), stored.encode("ascii"))
        except Exception:
            return False
    return hash_password(password) == stored      # older accounts (SHA-256)


def find_mechanic_login(phone, password):
    conn = db_connect()
    row = conn.execute("SELECT * FROM mechanics WHERE phone = ?", (phone.strip(),)).fetchone()
    if row and verify_password(password, row["password"]):
        if bcrypt and not str(row["password"]).startswith("$2"):      # upgrade old hash on login
            conn.execute("UPDATE mechanics SET password = ? WHERE id = ?", (hash_password_new(password), row["id"]))
            conn.commit()
        conn.close()
        return row
    conn.close()
    return None


def money(value):
    return f"₹{int(value or 0):,}"


def load_hero_svg_data():
    if not HERO_SVG_PATH.exists():
        return ""
    raw = HERO_SVG_PATH.read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(raw).decode("utf-8")


def load_reference_home_data():
    if not REFERENCE_HOME_PATH.exists():
        return ""
    raw = REFERENCE_HOME_PATH.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("utf-8")


@st.cache_resource
def load_yolo_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return None


try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:                                    # tz database missing: India has no DST, so a fixed offset is exact
    IST = timezone(timedelta(hours=5, minutes=30))
TS_FORMAT = "%d-%b-%Y, %I:%M %p"                     # e.g. 05-Nov-2026, 08:25 PM
_OLD_TS_FORMATS = ("%d-%m-%Y %I:%M %p", TS_FORMAT)


def now_ist():
    """Current India time (Streamlit Cloud servers run on UTC)."""
    return datetime.now(IST).strftime(TS_FORMAT)


def parse_ts(value):
    """Parse a stored timestamp (old or new format). Returns a datetime or None."""
    for fmt in _OLD_TS_FORMATS:
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except Exception:
            continue
    return None


def pretty_ts(value):
    """Show any stored timestamp in the standard Indian format: 05-Nov-2026, 08:25 PM."""
    dt = parse_ts(value)
    return dt.strftime(TS_FORMAT) if dt else (str(value) if value else "-")


def load_image(image_bytes):
    """Open an image and fix phone-camera rotation (EXIF) before use."""
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def analyse_vehicle(uploaded_bytes, confidence):
    model = load_yolo_model()
    if model is None:
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    image = load_image(uploaded_bytes)
    image.thumbnail((1280, 1280))
    array = np.array(image)

    def run(conf):
        return model.predict(source=array, conf=conf, iou=0.50, verbose=False)[0]

    result = run(confidence)
    low_conf = False
    if (result.boxes is None or len(result.boxes) == 0) and confidence > 0.30:
        retry = run(0.30)          # second chance for phone photos / tricky lighting
        if retry.boxes is not None and len(retry.boxes) > 0:
            result, low_conf = retry, True

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            class_names = model.names
            if isinstance(class_names, dict):
                class_name = class_names.get(class_id, str(class_id))
            else:
                class_name = class_names[class_id]

            xyxy = box.xyxy[0].tolist()
            ah, aw = array.shape[:2]
            detections.append({
                "name": str(class_name),
                "confidence": score,
                "low_conf": low_conf,
                "box": [xyxy[0] / aw, xyxy[1] / ah, xyxy[2] / aw, xyxy[3] / ah]
            })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return image, result.plot(), detections


def insert_customer_report(customer, phone, area, vehicle, damage,
                           severity, min_cost, max_cost, image_bytes, email=None):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reports
        (customer, phone, area, vehicle, damage, severity,
         min_cost, max_cost, image, created, email)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        customer, phone, area, vehicle, damage, severity,
        min_cost, max_cost, image_bytes,
        now_ist(), email
    ))
    report_id = cur.lastrowid
    conn.commit()
    conn.close()
    return report_id


def get_report(report_id):
    conn = db_connect()
    row = conn.execute(
        "SELECT * FROM reports WHERE id = ?", (report_id,)
    ).fetchone()
    conn.close()
    return row


def get_mechanic(mechanic_id):
    conn = db_connect()
    row = conn.execute(
        "SELECT * FROM mechanics WHERE id = ?", (mechanic_id,)
    ).fetchone()
    conn.close()
    return row


# ------------------------- VISUAL DESIGN -------------------------

hero_svg = load_hero_svg_data()
reference_home = load_reference_home_data()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
/* REMOVE BLUE VEHICLE PANEL */
.hero-visual,
.hero-car {
    display: none !important;
}

.hero {
    justify-content: center !important;
}

.hero-copy {
    width: 48%;
    max-width: 620px;
    margin-left: auto;
    margin-right: 2%;
    position: relative;
    z-index: 2;
    text-align: center;
}
:root { --red:#ed101b; --red-soft:#ff5961; --ink:#050506; --panel:#111114; }
.stApp { background:#050506; color:#f5f5f6; font-family:Inter,sans-serif; }
header[data-testid="stHeader"] { background:transparent; }
footer,#MainMenu { visibility:hidden; }
.block-container { max-width:1680px; padding:0.55rem 1.0rem 1.1rem; }
h1,h2,h3,h4,p,label,span { font-family:Inter,sans-serif; }
h1,h2,h3,h4 { color:#f7f7f8; }
.brand-row { display:flex; align-items:center; gap:15px; min-height:72px; padding:4px 10px; }
.brand-icon { color:#f10c19; font-size:38px; line-height:1; text-shadow:0 0 13px rgba(255,0,20,.35); }
.brand-name { color:#fff; font-size:21px; font-weight:850; line-height:1.05; }
.brand-name span { color:#f10c19; }
.brand-sub { color:#aaa; font-size:10px; letter-spacing:1.2px; margin-top:5px; }
div[data-testid="stRadio"] { margin:0; }
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] > div[role="radiogroup"] { display:flex; justify-content:flex-end; gap:10px; flex-wrap:nowrap; }
div[data-testid="stRadio"] div[role="radio"] { background:linear-gradient(180deg,#171719,#09090a); border:1px solid #39393d; border-radius:13px; padding:9px 15px; min-height:48px; color:#eee!important; box-shadow:inset 0 1px 0 rgba(255,255,255,.04); }
div[data-testid="stRadio"] div[role="radio"]:has(input:checked) { border:1px solid #ff303b; background:linear-gradient(180deg,#3b080d,#150507); box-shadow:0 0 8px #f20d1b,0 0 20px rgba(242,13,27,.38),inset 0 0 10px rgba(242,13,27,.18); }
div[data-testid="stRadio"] div[role="radio"] > div:first-child { display:none; }
div[data-testid="stRadio"] div[role="radio"] p { color:#f5f5f5!important; font-size:14px; font-weight:600; margin:0; white-space:nowrap; }
.top-rule { height:2px; background:linear-gradient(90deg,transparent,#f10c19 12%,#f10c19 88%,transparent); margin:0 0 0.15rem; box-shadow:0 0 9px rgba(255,0,25,.35); }
.hero { min-height:350px; position:relative; overflow:hidden; display:flex; align-items:center; justify-content:flex-end; padding:28px 3.1%; margin:0 0 10px; background-color:#080809; background-image:linear-gradient(90deg,rgba(8,8,9,0) 0%,rgba(8,8,9,0) 46%,rgba(8,8,9,.92) 60%,#080809 70%,#080809 100%),url('data:image/jpeg;base64,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'); background-size:100% 100%,auto 100%; background-position:0 0,left center; background-repeat:no-repeat; border-bottom:1px solid #351014; }
.hero:before { content:''; position:absolute; inset:0; pointer-events:none; background:linear-gradient(135deg,transparent 0 72%,rgba(220,0,20,.12) 72.2%,transparent 74%); }
.hero-copy { width:49%; max-width:650px; position:relative; z-index:1; text-align:center; }
.hero-visual { position:absolute; left:1.5%; top:50%; transform:translateY(-50%); width:48%; max-height:92%; display:flex; align-items:center; justify-content:center; z-index:1; pointer-events:none; }
.hero-car { display:block; width:100%; max-height:330px; object-fit:contain; filter:drop-shadow(0 12px 28px rgba(237,16,27,.25)); }
.hero-emblem { color:#f10c19; font-size:62px; line-height:1; margin-bottom:8px; text-shadow:0 0 17px rgba(255,0,25,.45); }
.hero h1 { font-size:clamp(29px,3.3vw,54px); line-height:1.12; font-weight:900; font-style:italic; margin:0 0 18px; letter-spacing:-1.2px; }
.hero h1 .red { color:#f10c19; }
.hero-tagline { color:#f1f1f1; font-size:19px; font-weight:800; font-style:italic; letter-spacing:1.1px; margin:8px 0 14px; }
.hero-tagline b { color:#f10c19; padding:0 8px; }
.hero-desc { color:#eee; font-size:15px; line-height:1.55; max-width:500px; margin:0 auto; }
.portal-heading { text-align:center; font-size:28px; font-weight:900; font-style:italic; letter-spacing:.7px; margin:3px 0 9px; color:#fff; }
.portal-heading span { color:#f10c19; }
.portal-card-wrap { position:relative; overflow:hidden; min-height:175px; padding:12px 9px 8px; border-radius:13px; background-size:cover; background-position:center; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.portal-card-wrap) { background:#0b0b0d!important; border:2px solid #f02a3b!important; border-radius:21px!important; box-shadow:0 0 12px rgba(255,0,24,.12),inset 0 0 18px rgba(255,255,255,.03)!important; padding:5px!important; }
.portal-card-top { display:flex; align-items:center; gap:18px; min-height:72px; margin-bottom:8px; }
.portal-icon { color:#fff; font-size:49px; line-height:1; width:85px; text-align:center; text-shadow:0 0 10px rgba(255,255,255,.14); }
.portal-card-title { font-size:23px; line-height:1.2; font-weight:850; color:#fff; }
.portal-card-title span { color:#f10c19; }
.portal-card-desc { color:#f0eff0; font-size:15px; line-height:1.45; min-height:48px; max-width:390px; margin:6px 0 12px; }
.portal-card-wrap .stButton > button { max-width:300px; margin:0 auto; display:block; }
.feature-cell { display:flex; align-items:center; justify-content:center; gap:17px; min-height:78px; border:1px solid rgba(130,75,83,.55); background:linear-gradient(90deg,#09090a,#141416,#09090a); }
.feature-icon { color:#f10c19; font-size:38px; font-weight:900; line-height:1; }
.feature-text { color:#f2f2f2; font-size:17px; line-height:1.25; }
.home-footer { text-align:center; color:#e7e3e4; font-size:17px; font-style:italic; letter-spacing:1.5px; padding:13px 0 3px; border-top:1px solid rgba(220,0,20,.55); margin-top:10px; }
.home-footer span { color:#f10c19; padding:0 10px; }
.panel { background:linear-gradient(145deg,#171719,#0d0d0f); border:1px solid rgba(167,25,45,.42); border-radius:14px; padding:20px; box-shadow:inset 0 0 18px rgba(130,20,40,.05); margin-bottom:14px; }
.metric-card { background:#171619; border:1px solid #64202c; border-radius:12px; padding:16px; min-height:95px; }
.metric-label { color:#aaa; font-size:11px; letter-spacing:1px; }
.metric-value { color:#f04452; font-size:23px; font-weight:800; margin-top:7px; }
.stTextInput input,.stNumberInput input,.stTextArea textarea { background:#171719!important; border:1px solid #493038!important; color:#f5f2f3!important; border-radius:8px!important; }
[data-baseweb="select"] > div { background:#171719!important; border-color:#493038!important; }
[data-testid="stFileUploader"] { background:rgba(25,23,25,.8); border:1px dashed #843044; border-radius:12px; padding:12px; }
.stButton > button,.stFormSubmitButton > button { width:100%; background:linear-gradient(100deg,#8e101d,#e10e20)!important; border:1px solid #ff3545!important; color:#fff!important; border-radius:8px!important; font-weight:800!important; min-height:43px; letter-spacing:.4px; box-shadow:0 0 12px rgba(230,0,25,.18); }
.stButton > button:hover,.stFormSubmitButton > button:hover { background:linear-gradient(100deg,#ad1424,#ff172b)!important; border-color:#ff7a84!important; box-shadow:0 0 18px rgba(255,0,25,.35)!important; }
div[data-testid="stTabs"] button { color:#eee; }
hr { border-color:rgba(167,25,45,.35); }
.small-note { color:#999196; font-size:12px; }
@media(max-width:900px) { .hero { min-height:300px; background-position:35% center; } .hero-copy { width:60%; } .portal-card-wrap { min-height:230px; } div[data-testid="stRadio"] > div[role="radiogroup"] { flex-wrap:wrap; justify-content:center; } }
/* HERO LAYOUT: full car on the left, text on the right */
.hero { min-height:420px !important; padding:28px 4% !important; justify-content:flex-end !important; border:1px solid #2a0c10; border-radius:16px; box-shadow:0 0 30px rgba(237,16,27,.12), inset 0 0 40px rgba(0,0,0,.5); }
.hero-copy { width:40% !important; max-width:600px !important; margin:0 !important; text-align:center !important; z-index:2 !important; }
.hero-emblem { animation:glowPulse 3s ease-in-out infinite; }
.hero h1 { text-shadow:0 4px 22px rgba(0,0,0,.8); }
.hero h1 .red { text-shadow:0 0 22px rgba(241,12,25,.55); }
.hero-tagline { text-shadow:0 2px 10px rgba(0,0,0,.9); }
@keyframes glowPulse { 0%,100% { text-shadow:0 0 12px rgba(255,0,25,.35); } 50% { text-shadow:0 0 28px rgba(255,0,25,.85); } }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.portal-card-wrap) { transition:transform .25s ease, box-shadow .25s ease; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.portal-card-wrap):hover { transform:translateY(-4px); box-shadow:0 8px 28px rgba(255,0,24,.28) !important; }
.feature-cell { border-radius:10px; transition:border-color .25s ease; }
.feature-cell:hover { border-color:#f10c19; }
@media(max-width:900px) {
  .hero { min-height:520px !important; align-items:flex-end !important; background-size:100% 100%,100% auto !important; background-position:0 0,top center !important; padding-top:230px !important; }
  .hero-copy { width:100% !important; max-width:100% !important; }
}
/* LOGO */
.av-logo { display:flex; align-items:center; gap:14px; padding:4px 6px; }
.av-logo svg { width:62px; height:62px; flex:none; filter:drop-shadow(0 0 10px rgba(241,12,25,.55)); }
.av-logo .scan { animation:avScan 2.6s ease-in-out infinite; }
@keyframes avScan { 0%,100% { transform:translateY(-9px); opacity:.2; } 50% { transform:translateY(9px); opacity:1; } }
.av-name { color:#fff; font-size:27px; font-weight:900; font-style:italic; letter-spacing:.5px; line-height:1; text-shadow:0 2px 12px rgba(0,0,0,.7); }
.av-name span { color:#f10c19; text-shadow:0 0 16px rgba(241,12,25,.6); }
.av-sub { color:#b9b4b6; font-size:10px; letter-spacing:2.6px; margin-top:7px; font-weight:600; }
.av-bar { height:2px; width:100%; margin-top:7px; background:linear-gradient(90deg,#f10c19,transparent); }
/* PORTAL UPGRADE STYLES */
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.3px; margin-right:6px; vertical-align:middle; }
.b-green { background:#0f2a18; color:#6fe39a; border:1px solid #1f6b3a; }
.b-orange { background:#2e1a07; color:#ffb454; border:1px solid #8a5212; }
.b-red { background:#2e0a0d; color:#ff6b75; border:1px solid #8a1a22; }
.b-blue { background:#0a1d33; color:#6fb7ff; border:1px solid #1f4f85; }
.b-gray { background:#1d1d21; color:#bdb8ba; border:1px solid #3a3a40; }
.stepbar { display:flex; gap:8px; margin:8px 0 14px; flex-wrap:wrap; }
.stp { flex:1; min-width:90px; padding:8px 10px; border-radius:10px; background:#141416; border:1px solid #2e2e33; color:#8d8689; font-size:13px; font-weight:600; display:flex; align-items:center; gap:8px; }
.stp span { width:22px; height:22px; border-radius:50%; background:#26262b; display:inline-flex; align-items:center; justify-content:center; font-size:12px; flex:none; }
.stp.now { border-color:#ff303b; color:#fff; box-shadow:0 0 10px rgba(242,13,27,.3); }
.stp.now span { background:#e10e20; }
.stp.done { border-color:#1f6b3a; color:#6fe39a; }
.stp.done span { background:#1f6b3a; color:#fff; }
.res-card { display:flex; justify-content:space-between; align-items:center; background:#141416; border:1px solid #64202c; border-radius:12px; padding:14px 16px; margin-bottom:10px; }
.res-big { font-size:22px; font-weight:800; color:#fff; margin-top:4px; }
.res-cost { font-size:28px; font-weight:900; color:#ff4b57; margin-top:4px; }
.id-card { text-align:center; background:linear-gradient(145deg,#1a0a0d,#0d0d0f); border:1px solid #ff303b; border-radius:14px; padding:16px; margin:12px 0 6px; }
.id-num { font-size:42px; font-weight:900; color:#ff4b57; line-height:1.1; }
.band { position:relative; height:8px; background:#26262b; border-radius:4px; margin:6px 0 12px; }
.band-ai { position:absolute; top:0; bottom:0; background:rgba(111,227,154,.45); border-radius:4px; }
.band-pin { position:absolute; top:-4px; width:4px; height:16px; background:#ff4b57; border-radius:2px; transform:translateX(-2px); }
.link-btn { display:inline-block; margin:8px 8px 0 0; padding:8px 14px; border-radius:8px; background:#18181b; border:1px solid #ff303b; color:#fff !important; text-decoration:none; font-weight:700; font-size:13px; }
.link-btn:hover { background:#3b080d; }
/* THREE-DOTS MENU */
[data-testid="stPopover"] button, details[data-testid="stExpander"] > summary.menu-dots { min-width:46px; height:46px; padding:0 !important; font-size:24px !important; font-weight:900 !important; border-radius:10px; line-height:1; }
[data-testid="stPopoverBody"] { background:#101012 !important; border:1px solid #64202c !important; border-radius:12px; min-width:300px; }
/* NAV FIX: Streamlit header overlay was blocking clicks on the top buttons */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
.block-container { padding-top:1rem !important; }
.hero:before, .hero-visual, .hero-car { pointer-events:none !important; }
div[data-testid="stHorizontalBlock"]:has(button[kind]) { position:relative; z-index:50; }
</style>
""", unsafe_allow_html=True)

# Existing SVG is used as a decorative car visual if available.
hero_visual_html = ""
if hero_svg:
    hero_visual_html = (
        f'<img class="hero-car" src="{hero_svg}" '
        f'alt="Vehicle illustration">'
    )
# ------------------------- TOP NAV -------------------------

nav_items = ["Home", "Customer", "Mechanic", "Quotations", "About"]

nav_labels = {
    "Home": "⌂  HOME",
    "Customer": "♟  CUSTOMER",
    "Mechanic": "🔧  MECHANIC",
    "Quotations": "▤  QUOTATIONS",
    "About": "ⓘ  ABOUT"
}

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

def go_to_page(page_name):
    st.session_state["page"] = page_name


LOGO_HTML = """<div class="av-logo"><svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="avg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff4b57"/><stop offset="1" stop-color="#b00a14"/></linearGradient></defs><polygon points="32,3 58,18 58,46 32,61 6,46 6,18" fill="#0d0d0f" stroke="url(#avg)" stroke-width="3" stroke-linejoin="round"/><path d="M14 39 L18 31 Q20 28 24 28 L38 28 Q42 28 45 32 L50 36 Q52 37 52 39 L52 42 L14 42 Z" fill="url(#avg)"/><path d="M22 31 L26 31 L26 35 L19 35 Z M29 31 L38 31 L42 35 L29 35 Z" fill="#0d0d0f" opacity=".75"/><circle cx="22" cy="43" r="4.5" fill="#0d0d0f" stroke="#fff" stroke-width="2"/><circle cx="44" cy="43" r="4.5" fill="#0d0d0f" stroke="#fff" stroke-width="2"/><g class="scan"><line x1="11" y1="36" x2="53" y2="36" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/></g><path d="M12 22 V17 H17 M52 22 V17 H47" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg><div><div class="av-name">AUTOVISION <span>AI</span></div><div class="av-sub">VEHICLE DAMAGE INSPECTION</div><div class="av-bar"></div></div></div>"""

LANGS = ["English", "ಕನ್ನಡ", "हिन्दी"]
TR = {'⋮ Menu': ('⋮ ಮೆನು', '⋮ मेनू'), 'Signed in as mechanic': ('ಮೆಕ್ಯಾನಿಕ್ ಆಗಿ ಲಾಗಿನ್ ಆಗಿದ್ದೀರಿ', 'मैकेनिक के रूप में लॉग इन'), '**👋 Welcome, guest**': ('**👋 ಸ್ವಾಗತ, ಅತಿಥಿ**', '**👋 स्वागत है, अतिथि**'), 'Not signed in. Customers do not need to log in.': ('ಲಾಗಿನ್ ಆಗಿಲ್ಲ. ಗ್ರಾಹಕರು ಲಾಗಿನ್ ಆಗುವ ಅಗತ್ಯವಿಲ್ಲ.', 'लॉग इन नहीं है। ग्राहकों को लॉग इन करने की ज़रूरत नहीं।'), '🔧 Mechanic dashboard': ('🔧 ಮೆಕ್ಯಾನಿಕ್ ಡ್ಯಾಶ್\u200cಬೋರ್ಡ್', '🔧 मैकेनिक डैशबोर्ड'), '🚪 Log out': ('🚪 ಲಾಗ್ ಔಟ್', '🚪 लॉग आउट'), 'Mechanic phone number': ('ಮೆಕ್ಯಾನಿಕ್ ಫೋನ್ ಸಂಖ್ಯೆ', 'मैकेनिक फ़ोन नंबर'), 'Password': ('ಪಾಸ್\u200cವರ್ಡ್', 'पासवर्ड'), '🔑 LOG IN': ('🔑 ಲಾಗಿನ್', '🔑 लॉग इन'), 'Incorrect phone number or password.': ('ಫೋನ್ ಸಂಖ್ಯೆ ಅಥವಾ ಪಾಸ್\u200cವರ್ಡ್ ತಪ್ಪಾಗಿದೆ.', 'फ़ोन नंबर या पासवर्ड गलत है।'), '📝 Register a workshop': ('📝 ವರ್ಕ್\u200cಶಾಪ್ ನೋಂದಣಿ', '📝 वर्कशॉप पंजीकरण'), 'QUICK LINKS': ('ತ್ವರಿತ ಲಿಂಕ್\u200cಗಳು', 'त्वरित लिंक'), '📷 New damage inspection': ('📷 ಹೊಸ ಹಾನಿ ಪರಿಶೀಲನೆ', '📷 नई क्षति जाँच'), '📂 My reports and history': ('📂 ನನ್ನ ವರದಿಗಳು ಮತ್ತು ಇತಿಹಾಸ', '📂 मेरी रिपोर्ट और इतिहास'), '🧾 Track repair / quotations': ('🧾 ರಿಪೇರಿ / ಕೊಟೇಶನ್ ಟ್ರ್ಯಾಕ್', '🧾 मरम्मत / कोटेशन ट्रैक करें'), '🔐 Admin dashboard': ('🔐 ಅಡ್ಮಿನ್ ಡ್ಯಾಶ್\u200cಬೋರ್ಡ್', '🔐 एडमिन डैशबोर्ड'), 'Log out of admin': ('ಅಡ್ಮಿನ್\u200cನಿಂದ ಲಾಗ್ ಔಟ್', 'एडमिन से लॉग आउट'), '❓ How it works': ('❓ ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ', '❓ यह कैसे काम करता है'), '🛟 Help and tips': ('🛟 ಸಹಾಯ ಮತ್ತು ಸಲಹೆಗಳು', '🛟 सहायता और सुझाव'), '🔄 Reset session': ('🔄 ಸೆಶನ್ ಮರುಹೊಂದಿಸಿ', '🔄 सेशन रीसेट करें'), 'Ask about your repair…': ('ನಿಮ್ಮ ರಿಪೇರಿ ಬಗ್ಗೆ ಕೇಳಿ…', 'अपनी मरम्मत के बारे में पूछें…'), 'SEND': ('ಕಳುಹಿಸಿ', 'भेजें'), 'Clear chat': ('ಚಾಟ್ ತೆರವುಗೊಳಿಸಿ', 'चैट साफ़ करें'), '💬 Ask the AutoVision assistant': ('💬 ಆಟೋವಿಷನ್ ಸಹಾಯಕರನ್ನು ಕೇಳಿ', '💬 ऑटोविज़न सहायक से पूछें'), 'Admin password': ('ಅಡ್ಮಿನ್ ಪಾಸ್\u200cವರ್ಡ್', 'एडमिन पासवर्ड'), 'OPEN DASHBOARD': ('ಡ್ಯಾಶ್\u200cಬೋರ್ಡ್ ತೆರೆಯಿರಿ', 'डैशबोर्ड खोलें'), 'Incorrect password.': ('ಪಾಸ್\u200cವರ್ಡ್ ತಪ್ಪಾಗಿದೆ.', 'पासवर्ड गलत है।'), '**Most common damage**': ('**ಅತಿ ಸಾಮಾನ್ಯ ಹಾನಿ**', '**सबसे आम क्षति**'), '**Severity split**': ('**ತೀವ್ರತೆ ವಿಂಗಡಣೆ**', '**गंभीरता विभाजन**'), '**Reports by area**': ('**ಪ್ರದೇಶವಾರು ವರದಿಗಳು**', '**क्षेत्रवार रिपोर्ट**'), '**Repair status**': ('**ರಿಪೇರಿ ಸ್ಥಿತಿ**', '**मरम्मत की स्थिति**'), '**Reports per day**': ('**ದಿನಕ್ಕೆ ವರದಿಗಳು**', '**प्रति दिन रिपोर्ट**'), '**Recent reports**': ('**ಇತ್ತೀಚಿನ ವರದಿಗಳು**', '**हाल की रिपोर्ट**'), '**Workshop leaderboard**': ('**ವರ್ಕ್\u200cಶಾಪ್ ಲೀಡರ್\u200cಬೋರ್ಡ್**', '**वर्कशॉप लीडरबोर्ड**'), '⬇ EXPORT ALL REPORTS (CSV)': ('⬇ ಎಲ್ಲಾ ವರದಿಗಳನ್ನು ರಫ್ತು ಮಾಡಿ (CSV)', '⬇ सभी रिपोर्ट एक्सपोर्ट करें (CSV)'), 'Vehicle Damage Inspection': ('ವಾಹನ ಹಾನಿ ಪರಿಶೀಲನೆ', 'वाहन क्षति निरीक्षण'), '📂 My previous reports': ('📂 ನನ್ನ ಹಿಂದಿನ ವರದಿಗಳು', '📂 मेरी पिछली रिपोर्ट'), 'Enter your 10-digit phone number': ('ನಿಮ್ಮ 10 ಅಂಕಿಯ ಫೋನ್ ಸಂಖ್ಯೆಯನ್ನು ನಮೂದಿಸಿ', 'अपना 10 अंकों का फ़ोन नंबर दर्ज करें'), 'No reports found for this number.': ('ಈ ಸಂಖ್ಯೆಗೆ ವರದಿಗಳು ಸಿಗಲಿಲ್ಲ.', 'इस नंबर के लिए कोई रिपोर्ट नहीं मिली।'), '🧾 Insurance claim checklist': ('🧾 ವಿಮಾ ಕ್ಲೈಮ್ ಪರಿಶೀಲನಾ ಪಟ್ಟಿ', '🧾 बीमा क्लेम चेकलिस्ट'), 'Upload vehicle image': ('ವಾಹನದ ಚಿತ್ರ ಅಪ್\u200cಲೋಡ್ ಮಾಡಿ', 'वाहन की तस्वीर अपलोड करें'), 'Detection confidence': ('ಪತ್ತೆ ವಿಶ್ವಾಸ ಮಟ್ಟ', 'पहचान विश्वास स्तर'), '⚡ ANALYSE VEHICLE DAMAGE': ('⚡ ವಾಹನ ಹಾನಿ ವಿಶ್ಲೇಷಿಸಿ', '⚡ वाहन क्षति का विश्लेषण करें'), 'Upload a vehicle image first.': ('ಮೊದಲು ವಾಹನದ ಚಿತ್ರ ಅಪ್\u200cಲೋಡ್ ಮಾಡಿ.', 'पहले वाहन की तस्वीर अपलोड करें।'), 'START NEW INSPECTION': ('ಹೊಸ ಪರಿಶೀಲನೆ ಪ್ರಾರಂಭಿಸಿ', 'नई जाँच शुरू करें'), '### AI Inspection Result': ('### AI ಪರಿಶೀಲನೆ ಫಲಿತಾಂಶ', '### AI निरीक्षण परिणाम'), 'AI cost estimates are indicative and must be confirmed by a mechanic.': ('AI ವೆಚ್ಚದ ಅಂದಾಜುಗಳು ಸೂಚಕ ಮಾತ್ರ; ಮೆಕ್ಯಾನಿಕ್ ದೃಢೀಕರಿಸಬೇಕು.', 'AI लागत अनुमान केवल संकेतक हैं; मैकेनिक को पुष्टि करनी होगी।'), '### Share report with mechanics': ('### ಮೆಕ್ಯಾನಿಕ್\u200cಗಳೊಂದಿಗೆ ವರದಿ ಹಂಚಿಕೊಳ್ಳಿ', '### मैकेनिकों के साथ रिपोर्ट साझा करें'), 'Customer name': ('ಗ್ರಾಹಕರ ಹೆಸರು', 'ग्राहक का नाम'), '10-digit phone number': ('10 ಅಂಕಿಯ ಫೋನ್ ಸಂಖ್ಯೆ', '10 अंकों का फ़ोन नंबर'), 'Vehicle make': ('ವಾಹನ ತಯಾರಕ', 'वाहन निर्माता'), 'Area in Bengaluru': ('ಬೆಂಗಳೂರಿನ ಪ್ರದೇಶ', 'बेंगलुरु में क्षेत्र'), 'If other, type your area': ('ಇತರೆ ಆದರೆ, ನಿಮ್ಮ ಪ್ರದೇಶ ಬರೆಯಿರಿ', 'अन्य हो तो अपना क्षेत्र लिखें'), 'Vehicle model': ('ವಾಹನ ಮಾದರಿ', 'वाहन मॉडल'), 'SHARE DAMAGE REPORT': ('ಹಾನಿ ವರದಿ ಹಂಚಿಕೊಳ್ಳಿ', 'क्षति रिपोर्ट साझा करें'), 'Please complete all fields.': ('ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಕ್ಷೇತ್ರಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿ.', 'कृपया सभी फ़ील्ड भरें।'), 'Enter a valid 10-digit phone number.': ('ಮಾನ್ಯ 10 ಅಂಕಿಯ ಫೋನ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ.', 'मान्य 10 अंकों का फ़ोन नंबर दर्ज करें।'), '⬇ DOWNLOAD PDF REPORT': ('⬇ PDF ವರದಿ ಡೌನ್\u200cಲೋಡ್ ಮಾಡಿ', '⬇ PDF रिपोर्ट डाउनलोड करें'), 'Mechanic Portal': ('ಮೆಕ್ಯಾನಿಕ್ ಪೋರ್ಟಲ್', 'मैकेनिक पोर्टल'), 'Mechanic / workshop name': ('ಮೆಕ್ಯಾನಿಕ್ / ವರ್ಕ್\u200cಶಾಪ್ ಹೆಸರು', 'मैकेनिक / वर्कशॉप का नाम'), 'Mobile number': ('ಮೊಬೈಲ್ ಸಂಖ್ಯೆ', 'मोबाइल नंबर'), 'Workshop area': ('ವರ್ಕ್\u200cಶಾಪ್ ಪ್ರದೇಶ', 'वर्कशॉप क्षेत्र'), 'Create password': ('ಪಾಸ್\u200cವರ್ಡ್ ರಚಿಸಿ', 'पासवर्ड बनाएँ'), 'Confirm password': ('ಪಾಸ್\u200cವರ್ಡ್ ದೃಢೀಕರಿಸಿ', 'पासवर्ड की पुष्टि करें'), 'REGISTER WORKSHOP': ('ವರ್ಕ್\u200cಶಾಪ್ ನೋಂದಣಿ ಮಾಡಿ', 'वर्कशॉप पंजीकृत करें'), 'Registered phone number': ('ನೋಂದಾಯಿತ ಫೋನ್ ಸಂಖ್ಯೆ', 'पंजीकृत फ़ोन नंबर'), 'LOGIN': ('ಲಾಗಿನ್', 'लॉग इन'), 'Register Workshop': ('ವರ್ಕ್\u200cಶಾಪ್ ನೋಂದಣಿ', 'वर्कशॉप पंजीकरण'), 'Mechanic Login': ('ಮೆಕ್ಯಾನಿಕ್ ಲಾಗಿನ್', 'मैकेनिक लॉगिन'), '⭐ Customer reviews': ('⭐ ಗ್ರಾಹಕರ ವಿಮರ್ಶೆಗಳು', '⭐ ग्राहक समीक्षाएँ'), 'Edit workshop profile': ('ವರ್ಕ್\u200cಶಾಪ್ ಪ್ರೊಫೈಲ್ ಸಂಪಾದಿಸಿ', 'वर्कशॉप प्रोफ़ाइल संपादित करें'), 'Services offered': ('ನೀಡುವ ಸೇವೆಗಳು', 'दी जाने वाली सेवाएँ'), 'Working hours': ('ಕೆಲಸದ ಸಮಯ', 'कार्य समय'), 'SAVE PROFILE': ('ಪ್ರೊಫೈಲ್ ಉಳಿಸಿ', 'प्रोफ़ाइल सहेजें'), '🔒 Change password': ('🔒 ಪಾಸ್\u200cವರ್ಡ್ ಬದಲಿಸಿ', '🔒 पासवर्ड बदलें'), 'Current password': ('ಪ್ರಸ್ತುತ ಪಾಸ್\u200cವರ್ಡ್', 'वर्तमान पासवर्ड'), 'New password': ('ಹೊಸ ಪಾಸ್\u200cವರ್ಡ್', 'नया पासवर्ड'), 'Confirm new password': ('ಹೊಸ ಪಾಸ್\u200cವರ್ಡ್ ದೃಢೀಕರಿಸಿ', 'नए पासवर्ड की पुष्टि करें'), 'UPDATE PASSWORD': ('ಪಾಸ್\u200cವರ್ಡ್ ನವೀಕರಿಸಿ', 'पासवर्ड अपडेट करें'), 'LOG OUT': ('ಲಾಗ್ ಔಟ್', 'लॉग आउट'), '### Customer requests matching your area': ('### ನಿಮ್ಮ ಪ್ರದೇಶಕ್ಕೆ ಹೊಂದುವ ಗ್ರಾಹಕರ ವಿನಂತಿಗಳು', '### आपके क्षेत्र के ग्राहक अनुरोध'), 'Show requests': ('ವಿನಂತಿಗಳನ್ನು ತೋರಿಸಿ', 'अनुरोध दिखाएँ'), 'Severity': ('ತೀವ್ರತೆ', 'गंभीरता'), 'UPDATE STATUS': ('ಸ್ಥಿತಿ ನವೀಕರಿಸಿ', 'स्थिति अपडेट करें'), 'Parts (₹)': ('ಬಿಡಿಭಾಗಗಳು (₹)', 'पार्ट्स (₹)'), 'Labour (₹)': ('ಕೂಲಿ (₹)', 'मज़दूरी (₹)'), 'Paint (₹)': ('ಬಣ್ಣ (₹)', 'पेंट (₹)'), 'Days needed': ('ಬೇಕಾದ ದಿನಗಳು', 'आवश्यक दिन'), 'Warranty': ('ವಾರಂಟಿ', 'वारंटी'), 'Repair details': ('ರಿಪೇರಿ ವಿವರಗಳು', 'मरम्मत विवरण'), 'SUBMIT / UPDATE QUOTATION': ('ಕೊಟೇಶನ್ ಸಲ್ಲಿಸಿ / ನವೀಕರಿಸಿ', 'कोटेशन जमा / अपडेट करें'), 'WITHDRAW MY QUOTE': ('ನನ್ನ ಕೊಟೇಶನ್ ಹಿಂಪಡೆಯಿರಿ', 'मेरा कोटेशन वापस लें'), 'Compare Mechanic Quotations': ('ಮೆಕ್ಯಾನಿಕ್ ಕೊಟೇಶನ್\u200cಗಳನ್ನು ಹೋಲಿಸಿ', 'मैकेनिक कोटेशन की तुलना करें'), 'Damage Report ID': ('ಹಾನಿ ವರದಿ ಐಡಿ', 'क्षति रिपोर्ट आईडी'), 'Customer phone number': ('ಗ್ರಾಹಕರ ಫೋನ್ ಸಂಖ್ಯೆ', 'ग्राहक का फ़ोन नंबर'), 'FIND QUOTATIONS': ('ಕೊಟೇಶನ್\u200cಗಳನ್ನು ಹುಡುಕಿ', 'कोटेशन खोजें'), 'Report ID not found.': ('ವರದಿ ಐಡಿ ಸಿಗಲಿಲ್ಲ.', 'रिपोर्ट आईडी नहीं मिली।'), 'Phone number does not match this report.': ('ಫೋನ್ ಸಂಖ್ಯೆ ಈ ವರದಿಗೆ ಹೊಂದುವುದಿಲ್ಲ.', 'फ़ोन नंबर इस रिपोर्ट से मेल नहीं खाता।'), '⬇ DOWNLOAD COMPLETE REPAIR JOURNEY (PDF)': ('⬇ ಸಂಪೂರ್ಣ ರಿಪೇರಿ ಪ್ರಯಾಣ ಡೌನ್\u200cಲೋಡ್ ಮಾಡಿ (PDF)', '⬇ पूरी मरम्मत यात्रा डाउनलोड करें (PDF)'), 'No quotations received yet. Please check again later.': ('ಇನ್ನೂ ಕೊಟೇಶನ್\u200cಗಳು ಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ನಂತರ ಪರಿಶೀಲಿಸಿ.', 'अभी तक कोई कोटेशन नहीं मिला। कृपया बाद में जाँचें।'), '**Rate this repair**': ('**ಈ ರಿಪೇರಿಗೆ ರೇಟಿಂಗ್ ನೀಡಿ**', '**इस मरम्मत को रेटिंग दें**'), 'Comment (optional)': ('ಕಾಮೆಂಟ್ (ಐಚ್ಛಿಕ)', 'टिप्पणी (वैकल्पिक)'), 'SUBMIT REVIEW': ('ವಿಮರ್ಶೆ ಸಲ್ಲಿಸಿ', 'समीक्षा जमा करें'), 'Sort quotes by': ('ಕೊಟೇಶನ್\u200cಗಳನ್ನು ವಿಂಗಡಿಸಿ', 'कोटेशन क्रमबद्ध करें'), 'Maximum budget (₹)': ('ಗರಿಷ್ಠ ಬಜೆಟ್ (₹)', 'अधिकतम बजट (₹)'), '📊 Compare side by side': ('📊 ಅಕ್ಕಪಕ್ಕ ಹೋಲಿಸಿ', '📊 साथ-साथ तुलना करें'), '⬇ DOWNLOAD COMPARISON (CSV)': ('⬇ ಹೋಲಿಕೆ ಡೌನ್\u200cಲೋಡ್ ಮಾಡಿ (CSV)', '⬇ तुलना डाउनलोड करें (CSV)'), 'CHANGE MECHANIC': ('ಮೆಕ್ಯಾನಿಕ್ ಬದಲಿಸಿ', 'मैकेनिक बदलें'), 'SELECT THIS MECHANIC': ('ಈ ಮೆಕ್ಯಾನಿಕ್ ಆಯ್ಕೆಮಾಡಿ', 'इस मैकेनिक को चुनें'), 'AI Vehicle Damage Inspection System': ('AI ವಾಹನ ಹಾನಿ ಪರಿಶೀಲನಾ ವ್ಯವಸ್ಥೆ', 'AI वाहन क्षति निरीक्षण प्रणाली'), 'Upload': ('ಅಪ್\u200cಲೋಡ್', 'अपलोड'), 'Analyse': ('ವಿಶ್ಲೇಷಣೆ', 'विश्लेषण'), 'Share report': ('ವರದಿ ಹಂಚಿಕೆ', 'रिपोर्ट साझा करें'), '🗺 Workshops map': ('🗺 ವರ್ಕ್\u200cಶಾಪ್ ನಕ್ಷೆ', '🗺 वर्कशॉप का नक्शा'), 'Show workshops in': ('ಇಲ್ಲಿನ ವರ್ಕ್\u200cಶಾಪ್\u200cಗಳನ್ನು ತೋರಿಸಿ', 'यहाँ के वर्कशॉप दिखाएँ'), '🗺 Where are these workshops?': ('🗺 ಈ ವರ್ಕ್\u200cಶಾಪ್\u200cಗಳು ಎಲ್ಲಿವೆ?', '🗺 ये वर्कशॉप कहाँ हैं?'), 'No workshops with a known area to show yet.': ('ತೋರಿಸಲು ತಿಳಿದ ಪ್ರದೇಶದ ವರ್ಕ್\u200cಶಾಪ್\u200cಗಳು ಇನ್ನೂ ಇಲ್ಲ.', 'दिखाने के लिए ज्ञात क्षेत्र वाले वर्कशॉप अभी नहीं हैं।'), '💬 Message this workshop': ('💬 ಈ ವರ್ಕ್\u200cಶಾಪ್\u200cಗೆ ಸಂದೇಶ ಕಳುಹಿಸಿ', '💬 इस वर्कशॉप को संदेश भेजें'), '**💬 Messages with customer**': ('**💬 ಗ್ರಾಹಕರೊಂದಿಗೆ ಸಂದೇಶಗಳು**', '**💬 ग्राहक के साथ संदेश**'), 'No messages yet. Ask a question or share details about the damage.': ('ಇನ್ನೂ ಸಂದೇಶಗಳಿಲ್ಲ. ಪ್ರಶ್ನೆ ಕೇಳಿ ಅಥವಾ ಹಾನಿಯ ವಿವರ ಹಂಚಿಕೊಳ್ಳಿ.', 'अभी कोई संदेश नहीं। प्रश्न पूछें या क्षति का विवरण साझा करें।'), 'Your message': ('ನಿಮ್ಮ ಸಂದೇಶ', 'आपका संदेश'), 'SEND MESSAGE': ('ಸಂದೇಶ ಕಳುಹಿಸಿ', 'संदेश भेजें'), 'Type a message first.': ('ಮೊದಲು ಸಂದೇಶ ಬರೆಯಿರಿ.', 'पहले संदेश लिखें।'), 'For privacy, phone numbers are shared automatically once a mechanic is selected.': ('ಗೌಪ್ಯತೆಗಾಗಿ, ಮೆಕ್ಯಾನಿಕ್ ಆಯ್ಕೆಯಾದ ನಂತರ ಫೋನ್ ಸಂಖ್ಯೆ ತಾನಾಗಿ ಹಂಚಿಕೆಯಾಗುತ್ತದೆ.', 'गोपनीयता के लिए, मैकेनिक चुने जाने के बाद फ़ोन नंबर अपने आप साझा हो जाता है।'), '🗑 Delete this report': ('🗑 ಈ ವರದಿಯನ್ನು ಅಳಿಸಿ', '🗑 यह रिपोर्ट हटाएँ'), 'This permanently deletes the report, all quotations and messages.': ('ಇದು ವರದಿ, ಎಲ್ಲಾ ಕೊಟೇಶನ್\u200cಗಳು ಮತ್ತು ಸಂದೇಶಗಳನ್ನು ಶಾಶ್ವತವಾಗಿ ಅಳಿಸುತ್ತದೆ.', 'यह रिपोर्ट, सभी कोटेशन और संदेश हमेशा के लिए हटा देगा।'), 'I understand, delete this report': ('ಅರ್ಥವಾಯಿತು, ಈ ವರದಿಯನ್ನು ಅಳಿಸಿ', 'मैं समझता हूँ, यह रिपोर्ट हटाएँ'), 'DELETE REPORT': ('ವರದಿ ಅಳಿಸಿ', 'रिपोर्ट हटाएँ'), 'Tick the box to confirm.': ('ದೃಢೀಕರಿಸಲು ಬಾಕ್ಸ್ ಆಯ್ಕೆಮಾಡಿ.', 'पुष्टि के लिए बॉक्स चुनें।'), 'Cost breakdown (parts, labour, paint)': ('ವೆಚ್ಚ ವಿಂಗಡಣೆ (ಬಿಡಿಭಾಗ, ಕೂಲಿ, ಬಣ್ಣ)', 'लागत विवरण (पार्ट्स, मज़दूरी, पेंट)')}
TR.update({'➕ Add more angles (optional, up to 3)': ('➕ ಇನ್ನಷ್ಟು ಕೋನಗಳನ್ನು ಸೇರಿಸಿ (ಐಚ್ಛಿಕ, ಗರಿಷ್ಠ 3)', '➕ और एंगल जोड़ें (वैकल्पिक, अधिकतम 3)'), '📸 Other angles': ('📸 ಇತರ ಕೋನಗಳು', '📸 अन्य एंगल'), '🚗 My Garage': ('🚗 ನನ್ನ ಗ್ಯಾರೇಜ್', '🚗 मेरा गैराज'), 'Phone number for My Garage': ('ಗ್ಯಾರೇಜ್\u200cಗಾಗಿ ಫೋನ್ ಸಂಖ್ಯೆ', 'गैराज के लिए फ़ोन नंबर'), 'Remove': ('ತೆಗೆಯಿರಿ', 'हटाएँ'), 'ADD VEHICLE': ('ವಾಹನ ಸೇರಿಸಿ', 'वाहन जोड़ें'), 'Use a saved vehicle': ('ಉಳಿಸಿದ ವಾಹನ ಬಳಸಿ', 'सहेजा हुआ वाहन उपयोग करें'), 'Email (optional, for quote alerts)': ('ಇಮೇಲ್ (ಐಚ್ಛಿಕ, ಕೊಟೇಶನ್ ಸೂಚನೆಗಳಿಗಾಗಿ)', 'ईमेल (वैकल्पिक, कोटेशन अलर्ट के लिए)'), 'Save this vehicle to My Garage': ('ಈ ವಾಹನವನ್ನು ನನ್ನ ಗ್ಯಾರೇಜ್\u200cಗೆ ಉಳಿಸಿ', 'इस वाहन को मेरे गैराज में सहेजें'), 'Open for new jobs': ('ಹೊಸ ಕೆಲಸಗಳಿಗೆ ತೆರೆದಿದೆ', 'नए कामों के लिए खुला'), 'Price list (optional)': ('ಬೆಲೆ ಪಟ್ಟಿ (ಐಚ್ಛಿಕ)', 'मूल्य सूची (वैकल्पिक)'), '**📷 Repair photos**': ('**📷 ರಿಪೇರಿ ಫೋಟೋಗಳು**', '**📷 मरम्मत की तस्वीरें**'), '🎨 Theme (experimental)': ('🎨 ಥೀಮ್ (ಪ್ರಾಯೋಗಿಕ)', '🎨 थीम (प्रायोगिक)')})
NAV_TR = {"Home": ("ಮುಖಪುಟ", "होम"), "Customer": ("ಗ್ರಾಹಕ", "ग्राहक"), "Mechanic": ("ಮೆಕ್ಯಾನಿಕ್", "मैकेनिक"),
          "Quotations": ("ಕೊಟೇಶನ್‌ಗಳು", "कोटेशन"), "About": ("ಬಗ್ಗೆ", "परिचय")}


def tr(text):
    """Translate a UI label into the language chosen in the three-dots menu (falls back to English)."""
    lang = st.session_state.get("lang", "English")
    pair = TR.get(text)
    if lang == "English" or not pair:
        return text
    return pair[0] if lang == "ಕನ್ನಡ" else pair[1]


def nav_label(page):
    lang = st.session_state.get("lang", "English")
    if lang == "English":
        return nav_labels[page]
    icon = nav_labels[page].split()[0]
    return f"{icon}  " + (NAV_TR[page][0] if lang == "ಕನ್ನಡ" else NAV_TR[page][1])


def do_logout():
    st.session_state.mechanic_id = None
    st.session_state.admin_ok = False
    st.session_state.page = "Home"


def reset_session():
    for k in list(st.session_state.keys()):
        if k != "page":
            try:
                del st.session_state[k]
            except Exception:
                pass


def render_account_menu():
    """Three-dots menu (top right): login / logout, quick links, help and session tools."""
    mech_id = st.session_state.get("mechanic_id")
    mech = get_mechanic(mech_id) if mech_id else None
    admin = st.session_state.get("admin_ok", False)

    if hasattr(st, "popover"):
        box = st.popover("⋮", use_container_width=True)
    else:
        box = st.expander(tr("⋮ Menu"))

    with box:
        if mech:
            st.markdown(f"**👤 {mech['name']}**  \n📍 {mech['area']}  •  📞 {mech['phone']}")
            st.caption(tr("Signed in as mechanic"))
        else:
            st.markdown(tr("**👋 Welcome, guest**"))
            st.caption(tr("Not signed in. Customers do not need to log in."))
        st.divider()

        if mech:
            st.button(tr("🔧 Mechanic dashboard"), key="menu_dash", on_click=go_to_page, args=("Mechanic",),
                      use_container_width=True)
            st.button(tr("🚪 Log out"), key="menu_logout", on_click=do_logout, use_container_width=True)
        else:
            with st.form("menu_login_form"):
                m_phone = st.text_input(tr("Mechanic phone number"), key="menu_phone")
                m_pw = st.text_input(tr("Password"), type="password", key="menu_pw")
                go = st.form_submit_button(tr("🔑 LOG IN"), use_container_width=True)
            if go:
                found = find_mechanic_login(m_phone, m_pw)
                if found:
                    st.session_state.mechanic_id = found["id"]
                    st.session_state.page = "Mechanic"
                    st.rerun()
                else:
                    st.error(tr("Incorrect phone number or password."))
            st.button(tr("📝 Register a workshop"), key="menu_register", on_click=go_to_page, args=("Mechanic",),
                      use_container_width=True)
        st.divider()

        st.caption(tr("QUICK LINKS"))
        st.button(tr("📷 New damage inspection"), key="menu_new", on_click=go_to_page, args=("Customer",),
                  use_container_width=True)
        st.button(tr("📂 My reports and history"), key="menu_hist", on_click=go_to_page, args=("Customer",),
                  use_container_width=True)
        st.button(tr("🧾 Track repair / quotations"), key="menu_track", on_click=go_to_page, args=("Quotations",),
                  use_container_width=True)
        st.button(tr("🔐 Admin dashboard"), key="menu_admin", on_click=go_to_page, args=("About",),
                  use_container_width=True)
        if admin:
            st.button(tr("Log out of admin"), key="menu_admin_out",
                      on_click=lambda: st.session_state.update(admin_ok=False), use_container_width=True)
        st.divider()

        with st.expander(tr("❓ How it works")):
            st.markdown("""1. **Customer** uploads a car photo and the AI finds the damage
2. The app shows severity, repair cost and repair time
3. The report is shared with mechanics in your area
4. Mechanics send itemised quotations
5. You compare, select a mechanic and track the repair""")
        with st.expander(tr("🛟 Help and tips")):
            st.markdown("""- Use a clear, well-lit photo with the damage in the centre
- Keep your **Report ID** and phone number to check quotations
- Customer phone numbers are shown to a mechanic only after they are selected
- AI estimates are indicative. A mechanic confirms the final price""")
        st.selectbox("🌐 Language / ಭಾಷೆ / भाषा", LANGS, key="lang")
        st.selectbox(tr("🎨 Theme (experimental)"), ["Dark", "Light"], key="theme")
        st.button(tr("🔄 Reset session"), key="menu_reset", on_click=reset_session, use_container_width=True)
        st.caption("AutoVision AI v2.0 • Prototype")


brand_col, nav_col, menu_col = st.columns([2.4, 8, 0.7], vertical_alignment="center")

with brand_col:
    brand_path = BASE_DIR / "static" / "brand_header.png"

    if brand_path.exists():
        brand_b64 = base64.b64encode(
            brand_path.read_bytes()
        ).decode("ascii")

        st.markdown(
            f'<div class="brand-row"><img src="data:image/png;base64,{brand_b64}" style="width:100%;max-width:350px;height:auto;"></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(LOGO_HTML, unsafe_allow_html=True)

with nav_col:
    nav_columns = st.columns(5, gap="small")

    for nav_column, target_page in zip(nav_columns, nav_items):
        with nav_column:
            st.button(
                nav_label(target_page),
                key=f"working_nav_{target_page.lower()}",
                type="primary" if st.session_state["page"] == target_page else "secondary",
                use_container_width=True,
                on_click=go_to_page,
                args=(target_page,)
            )

with menu_col:
    render_account_menu()

if st.session_state.get("theme") == "Light":
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
# ------------------------- HOME PAGE -------------------------
def render_home():
    st.markdown(f"""
    <section class="hero">
      <div class="hero-copy">
        <div class="hero-emblem">⬡</div>
        <h1>AI Vehicle Damage<br><span class="red">Inspection System</span></h1>
        <div class="hero-tagline">DETECT <b>•</b> ANALYZE <b>•</b> REPAIR SMARTER</div>
        <p class="hero-desc">Smart AI-powered vehicle damage detection,<br>severity analysis and repair cost estimation.<br>Connect with mechanics and compare quotations.</p>
      </div>
    </section>
    """, unsafe_allow_html=True)

    st.markdown('<div class="portal-heading">CHOOSE YOUR <span>PORTAL</span></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    portals = [
        (c1, "customer", "♟", "Customer <span>Portal</span>",
         "Upload vehicle images, detect damage<br>and get repair cost estimates.",
         "OPEN CUSTOMER  →", "Customer", "customer_card_scene.jpg"),
        (c2, "mechanic", "🔧", "Mechanic <span>Portal</span>",
         "Register or log in, view customer<br>reports and submit repair quotations.",
         "OPEN MECHANIC  →", "Mechanic", "mechanic_card_scene.jpg"),
        (c3, "quotes", "▤", "Quotations", 
         "Compare mechanic quotations and<br>select the best repair offer.",
         "VIEW QUOTATIONS  →", "Quotations", "quotes_card_scene.jpg"),
    ]
    for col, key, icon, title, description, button_text, target, image_name in portals:
        with col:
            with st.container(border=True):
                image_path = BASE_DIR / "static" / image_name
                image_data = ""
                if image_path.exists():
                    image_data = __import__('base64').b64encode(image_path.read_bytes()).decode('ascii')
                card_style = f"background-image:linear-gradient(90deg,rgba(5,5,7,.98) 0%,rgba(5,5,7,.86) 58%,rgba(5,5,7,.28) 100%),url(data:image/jpeg;base64,{image_data});" if image_data else ""
                st.markdown(
                    f'<div class="portal-card-wrap" style="{card_style}">'
                    f'<div class="portal-card-top"><div class="portal-icon">{icon}</div>'
                    f'<div class="portal-card-title">{title}</div></div>'
                    f'<div class="portal-card-desc">{description}</div></div>',
                    unsafe_allow_html=True
                )
                st.button(button_text, key=f"home_{key}_button", on_click=go_to_page,
                          args=(target,), use_container_width=True)

    features = st.columns(4, gap="small")
    feature_items = [("◎", "Accurate<br>AI Analysis"), ("ϟ", "Fast &amp;<br>Reliable"),
                     ("⌕", "Find<br>Mechanics"), ("⬟", "Transparent<br>Pricing")]
    for col, (icon, label) in zip(features, feature_items):
        with col:
            st.markdown(f'<div class="feature-cell"><div class="feature-icon">{icon}</div><div class="feature-text">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="home-footer">Drive Safe <span>•</span> Repair Smart</div>', unsafe_allow_html=True)


# ------------------------- SHARED UI HELPERS -------------------------

BENGALURU_AREAS = ["Whitefield", "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar",
                   "Electronic City", "Marathahalli", "BTM Layout", "Yelahanka", "Hebbal",
                   "Malleshwaram", "JP Nagar", "Banashankari", "Rajajinagar", "Other"]
CAR_MAKES = ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota", "Kia",
             "Ford", "Renault", "Volkswagen", "Skoda", "BMW", "Mercedes-Benz", "Audi", "Other"]
STATUS_FLOW = ["Selected", "Car dropped", "In repair", "Ready"]


def badge(text, kind="gray"):
    return f"<span class='badge b-{kind}'>{text}</span>"


def sev_kind(severity):
    s = str(severity).lower()
    if any(k in s for k in ("low", "minor", "light")):
        return "green"
    if any(k in s for k in ("high", "severe", "major", "critical")):
        return "red"
    return "orange"


def repair_time(severity):
    return {"green": (1, 2), "orange": (2, 4), "red": (5, 10)}[sev_kind(severity)]


def drive_tip(severity):
    return {"green": "Usually safe to drive. Get it repaired soon to prevent rust.",
            "orange": "Drive carefully for short distances only. Check lights, tyres and brakes first.",
            "red": "Avoid driving. Arrange towing and a professional inspection."}[sev_kind(severity)]


def step_bar(names, active):
    html = "".join(
        f"<div class='stp {'done' if i < active else 'now' if i == active else ''}'>"
        f"<span>{'✓' if i < active else i + 1}</span>{n}</div>"
        for i, n in enumerate(names)
    )
    st.markdown(f"<div class='stepbar'>{html}</div>", unsafe_allow_html=True)


def check_quality(image_bytes):
    """Return a list of photo-quality problems (warnings only)."""
    try:
        img = load_image(image_bytes).convert("L")
        img.thumbnail((512, 512))
        a = np.asarray(img, dtype=float)
        lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
        notes = []
        if a.mean() < 50:
            notes.append("too dark")
        if a.mean() > 225:
            notes.append("overexposed")
        if lap.var() < 25:
            notes.append("blurry")
        return notes
    except Exception:
        return []


def build_pdf(rep):
    """Build a PDF report. Returns bytes, or None if reportlab is missing."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    c.setFillColorRGB(0.1, 0.1, 0.12)
    c.rect(0, h - 90, w, 90, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, h - 45, "AUTOVISION AI")
    c.setFont("Helvetica", 11)
    c.drawString(40, h - 66, "Vehicle damage inspection report")
    qr = make_qr_png(report_qr_text(rep))
    if qr:
        c.drawImage(ImageReader(BytesIO(qr)), w - 85, h - 85, 70, 70)
    c.setFillColorRGB(0, 0, 0)
    y = h - 130
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, f"Report ID: {rep['id']}")
    c.setFont("Helvetica", 12)
    rows = [("Customer", rep["customer"]), ("Phone", rep["phone"]), ("Area", rep["area"]),
            ("Vehicle", rep["vehicle"]), ("Damage", str(rep["damage"]).replace("_", " ").title()),
            ("Severity", rep["severity"]),
            ("AI estimate", f"Rs {int(rep['min_cost'] or 0):,} - Rs {int(rep['max_cost'] or 0):,}"),
            ("Created", pretty_ts(rep["created"]))]
    for k, v in rows:
        y -= 22
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, f"{k}:")
        c.setFont("Helvetica", 12)
        c.drawString(140, y, str(v))
    try:
        img = ImageReader(BytesIO(rep["image_bytes"]))
        iw, ih = img.getSize()
        scale = min(500 / iw, 300 / ih)
        c.drawImage(img, 40, y - 40 - ih * scale, iw * scale, ih * scale)
    except Exception:
        pass
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, 40, "AI estimates are indicative and must be confirmed by a mechanic.")
    c.save()
    return buf.getvalue()


def build_journey_pdf(report_id):
    """Complete repair-journey PDF for the customer: report, quotes, mechanic, tracker, review."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                        Image as RLImage, KeepTogether)
    except ImportError:
        return None
    from xml.sax.saxutils import escape as esc

    report = get_report(report_id)
    if report is None:
        return None
    conn = db_connect()
    quotes = conn.execute("""SELECT q.*, m.name, m.phone AS mphone, m.area AS marea,
        (SELECT AVG(rating) FROM reviews r WHERE r.mechanic_id = m.id) AS rating
        FROM quotes q JOIN mechanics m ON m.id = q.mechanic_id WHERE q.report_id = ? ORDER BY q.amount""",
        (report_id,)).fetchall()
    review = conn.execute("SELECT rating, comment FROM reviews WHERE report_id = ?", (report_id,)).fetchone()
    conn.close()
    chosen = next((q for q in quotes if q["selected"]), None)
    status = report["repair_status"] or "Open"

    red, dark = colors.HexColor("#C40F1D"), colors.HexColor("#141416")
    base = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14)
    h = ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=13, textColor=red, spaceBefore=12, spaceAfter=5)
    cell = ParagraphStyle("c", parent=base, fontSize=9, leading=12)
    cellh = ParagraphStyle("ch", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    small = ParagraphStyle("s", parent=base, fontSize=8.5, textColor=colors.HexColor("#666666"))

    def tbl(rows, widths, head=True):
        data = [[Paragraph(esc(str(c)), cellh if head and i == 0 else cell) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if head:
            style.append(("BACKGROUND", (0, 0), (-1, 0), dark))
        t.setStyle(TableStyle(style))
        return t

    W = A4[0] - 40 * mm
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=f"AutoVision Report {report_id}")
    title_style = ParagraphStyle("t", parent=base, fontName="Helvetica-Bold", fontSize=22, leading=28, textColor=red)
    S = [Paragraph("AUTOVISION AI", title_style),
         Paragraph("Complete repair journey report", base),
         Paragraph(f"Report ID #{report_id}  |  Generated {now_ist()}", small), Spacer(1, 6)]

    S.append(Paragraph("1. Vehicle and customer details", h))
    S.append(tbl([["Customer", report["customer"]], ["Phone", report["phone"]], ["Area", report["area"]],
                  ["Vehicle", report["vehicle"]], ["Report created", pretty_ts(report["created"])]],
                 [40 * mm, W - 40 * mm], head=False))

    S.append(Paragraph("2. AI inspection result", h))
    S.append(tbl([["Damage type", str(report["damage"]).replace("_", " ").title()],
                  ["Severity", report["severity"]],
                  ["AI repair estimate", f"Rs {int(report['min_cost'] or 0):,} - Rs {int(report['max_cost'] or 0):,}"],
                  ["Estimated repair time", "{} to {} days".format(*repair_time(report["severity"]))],
                  ["Driving advice", drive_tip(report["severity"])]], [40 * mm, W - 40 * mm], head=False))
    try:
        pil = load_image(report["image"])
        pil.thumbnail((900, 900))
        ib = BytesIO()
        pil.save(ib, "JPEG", quality=85)
        ib.seek(0)
        scale = min(W / pil.width, 75 * mm / pil.height)
        S += [Spacer(1, 6), RLImage(ib, pil.width * scale, pil.height * scale)]
    except Exception:
        pass

    S.append(Paragraph("3. Quotations received", h))
    if quotes:
        rows = [["Mechanic", "Area", "Total", "Days", "Warranty", "Rating", "Status"]]
        for q in quotes:
            rows.append([q["name"], q["marea"], f"Rs {int(q['amount']):,}", q["days"] or "-", q["warranty"] or "-",
                         f"{q['rating']:.1f}" if q["rating"] else "-", "SELECTED" if q["selected"] else "-"])
        S.append(tbl(rows, [36 * mm, 27 * mm, 25 * mm, 13 * mm, 25 * mm, 15 * mm, W - 141 * mm]))
        for q in quotes:
            if q["parts"] or q["labour"] or q["paint"]:
                S.append(Paragraph(esc(f"{q['name']}: parts Rs {int(q['parts'] or 0):,}, labour Rs {int(q['labour'] or 0):,}, "
                                       f"paint Rs {int(q['paint'] or 0):,}. {q['details'] or ''}"), small))
    else:
        S.append(Paragraph("No quotations received yet.", base))

    S.append(Paragraph("4. Selected mechanic", h))
    if chosen:
        S.append(tbl([["Workshop", chosen["name"]], ["Phone", chosen["mphone"]], ["Area", chosen["marea"]],
                      ["Agreed amount", f"Rs {int(chosen['amount']):,}"], ["Warranty", chosen["warranty"] or "-"]],
                     [40 * mm, W - 40 * mm], head=False))
    else:
        S.append(Paragraph("No mechanic selected yet.", base))

    S.append(Paragraph("5. Repair progress", h))
    idx = STATUS_FLOW.index(status) if status in STATUS_FLOW else -1
    steps = [("Report created", True), ("Quotations received", bool(quotes)), ("Mechanic selected", bool(chosen)),
             ("Car dropped at workshop", idx >= 1), ("Repair in progress", idx >= 2),
             ("Ready for pickup", idx >= 3), ("Workshop rated", bool(review))]
    S.append(tbl([["Step", "Status"]] + [[n, "Done" if d else "Pending"] for n, d in steps], [W - 40 * mm, 40 * mm]))
    if review:
        S.append(Spacer(1, 4))
        S.append(Paragraph(esc(f"Your rating: {review['rating']} / 5. {review['comment'] or ''}"), base))

    S.append(Spacer(1, 10))
    S.append(Paragraph("AI estimates are indicative. Final diagnosis and price must be confirmed by the mechanic "
                       "after a physical inspection.", small))
    doc.build(S)
    return buf.getvalue()


# ------------------------- HEATMAP / ASSISTANT / ADMIN -------------------------

def make_heatmap(image_bytes, detections):
    """Red/yellow heat overlay built from the YOLO boxes (brighter = more confident damage)."""
    base = load_image(image_bytes)
    base.thumbnail((800, 800))
    w, h = base.size
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((h, w), dtype=np.float32)
    for d in detections:
        b = d.get("box")
        if not b:
            continue
        x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx, sy = max((x2 - x1) / 2.5, 12), max((y2 - y1) / 2.5, 12)
        heat += d["confidence"] * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
    if heat.max() <= 0:
        return base
    heat /= heat.max()
    r = np.clip(heat * 2, 0, 1)
    g = np.clip(heat * 2, 0, 1) * np.clip(2 - heat * 2, 0, 1)
    color = np.stack([r, g, np.zeros_like(r)], axis=-1) * 255
    alpha = (0.75 * np.power(heat, 0.9))[..., None]
    out = np.asarray(base, dtype=np.float32) * (1 - alpha) + color * alpha
    return Image.fromarray(out.astype("uint8"))


def _secret(name):
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val or os.environ.get(name)


def _kw_score(t, kws):
    """Score = total length of matched keywords. Phrases match as text; short words match whole words;
    longer words match as prefixes (repair -> repairing)."""
    score = 0
    for k in kws:
        if " " in k:
            hit = k in t
        elif len(k) <= 3:
            hit = re.search(r"\b" + re.escape(k) + r"\b", t) is not None
        else:
            hit = re.search(r"\b" + re.escape(k), t) is not None
        if hit:
            score += len(k)
    return score


_INTENTS = [
    ("greet", ["hi", "hello", "hey", "namaste", "good morning", "good evening", "good afternoon"]),
    ("thanks", ["thank", "thanks", "thx", "appreciate"]),
    ("bye", ["bye", "goodbye", "see you"]),
    ("start", ["want repair", "want to repair", "need repair", "need to repair", "repair my", "repair the car",
               "repair car", "fix my", "fix the car", "car repair", "my car is damaged", "car damaged",
               "car got damaged", "accident", "hit my car", "scratched my car", "dented my car", "get my car",
               "how to start", "how do i start", "where do i start", "what should i do", "what do i do",
               "need help", "help me", "get started"]),
    ("dashboard", ["dashboard", "dashbord", "dash board", "warning light", "check engine", "engine light",
                   "abs light", "oil light"]),
    ("howapp", ["how does this work", "how it works", "how does it work", "how to use", "what is this",
                "what can you do", "about this app", "this app", "the app", "app work", "autovision", "features", "tutorial", "explain the app",
                "five pages", "portals"]),
    ("upload", ["upload", "photo", "picture", "image", "camera", "gallery", "heic", "not detecting",
                "no damage", "nothing detected", "blurry", "detect"]),
    ("reportid", ["report id", "forgot id", "lost id", "where is my id", "my id", "report number", "case id"]),
    ("account", ["register", "login", "log in", "logout", "log out", "password", "sign up", "sign in",
                 "my account", "registration"]),
    ("mech_quote", ["send quote", "submit quote", "give quote", "my quote", "withdraw", "as a mechanic",
                    "i am a mechanic", "i'm a mechanic", "workshop owner"]),
    ("choose_quote", ["which quote", "which mechanic", "best quote", "best mechanic", "choose", "compare",
                      "recommended", "select mechanic", "which one"]),
    ("quotes_flow", ["quote", "quotation", "offer", "bids"]),
    ("cost", ["cost", "price", "estimate", "how much", "expensive", "charge", "budget", "rupee", "rs "]),
    ("worth", ["worth", "replace", "repair or", "should i repair", "is it necessary"]),
    ("drive", ["drive", "driving", "safe", "tow", "towing", "road"]),
    ("time", ["how long", "time", "days", "when will", "ready", "duration", "delay"]),
    ("insurance", ["insurance", "claim", "policy", "cashless"]),
    ("warranty", ["warranty", "guarantee", "guaranty"]),
    ("paint", ["paint", "colour", "color", "polish", "touch-up", "touch up"]),
    ("scratch", ["scratch", "scuff", "scrape"]),
    ("dent", ["dent", "deformation", "deformed", "bent", "pushed in"]),
    ("crack", ["crack", "hole", "windshield", "windscreen", "glass", "shatter"]),
    ("lamp", ["lamp", "headlight", "taillight", "tail light", "indicator", "bumper"]),
    ("severity", ["severity", "slight", "severe", "medium damage", "what does severity", "how bad"]),
    ("accuracy", ["accurate", "accuracy", "reliable", "trust", "correct", "wrong", "mistake", "confidence"]),
    ("review", ["review", "rating", "rate the", "feedback", "stars"]),
    ("language", ["language", "kannada", "hindi", "translate", "english"]),
    ("delete", ["delete", "remove report", "cancel report", "erase"]),
    ("message", ["message", "chat with", "talk to", "contact the mechanic", "call the mechanic", "whatsapp"]),
    ("map", ["map", "nearest", "near me", "nearby", "distance", "location", "where are"]),
    ("pdf", ["pdf", "download", "share report", "print", "qr"]),
    ("trouble", ["problem", "issue", "error", "not working", "bug", "stuck", "failed", "cannot", "can't", "doesn't work"]),
]


def _assistant_answer(intent, ctx, q):
    dmg, sev = ctx.get("damage"), ctx.get("severity")
    lo, hi = ctx.get("min_cost"), ctx.get("max_cost")
    quotes = ctx.get("quotes") or []
    have = lo is not None
    est_txt = (f"Rs {int(lo):,} to Rs {int(hi):,}" if hi else f"from Rs {int(lo):,}") if have else ""

    if intent == "greet":
        return "Hi! I'm the AutoVision assistant. Ask me about repair cost, time, safety, insurance, or how to use this app."
    if intent == "thanks":
        return "You're welcome! Ask me anything else about your repair."
    if intent == "bye":
        return "Bye! Come back anytime to check your quotations."
    if intent == "start":
        if have:
            return (f"You already analysed your photo: {dmg} ({sev}), estimate {est_txt}. Next: fill the share form below "
                    "(name, phone, area, vehicle) and tap SHARE DAMAGE REPORT. Then open Quotations with your Report ID.")
        return ("Here is how to get your car repaired with AutoVision:\n"
                "1. Customer page: upload a clear photo and tap ANALYSE VEHICLE DAMAGE.\n"
                "2. Check the AI result and estimate, then fill the form and SHARE DAMAGE REPORT. Note your Report ID.\n"
                "3. Mechanics near you send quotations.\n"
                "4. Quotations page: enter Report ID and phone, compare quotes and select a mechanic.")
    if intent == "dashboard":
        return ("Which dashboard do you mean?\n"
                "• Car dashboard warning lights (engine, oil, ABS, battery): AutoVision can't read these. A mechanic with a "
                "scanner must check them, and a red light means stop driving soon.\n"
                "• App dashboards: the Mechanic dashboard is on the Mechanic page after login (requests, quotes, earnings). "
                "The Admin dashboard is at the bottom of the About page and needs the admin password.")
    if intent == "howapp":
        return ("AutoVision has 5 pages: Home, Customer (upload a photo and get an AI damage estimate), Mechanic "
                "(workshops register, see requests and send quotes), Quotations (compare offers and track the repair) and About. "
                "Use the ⋮ menu at the top right for login, language and quick links.")
    if intent == "upload":
        return ("For a good result: take the photo in daylight, 1 to 2 metres away, with the whole damaged area in the frame and "
                "no blur. JPG, PNG and HEIC are fine, and you can use the camera button. If it says no damage was found, try a "
                "closer or clearer photo from another angle.")
    if intent == "reportid":
        return ("Your Report ID is shown after you share a report, and it is also in your PDF. If you lost it, open the Customer page "
                "→ 'My previous reports' and enter your phone number to see all your report IDs.")
    if intent == "account":
        return ("Only workshops need an account. On the Mechanic page, register once with your name, mobile, area and password, "
                "then log in (or use the ⋮ menu). Customers do not need to register. Forgot your password? Log in once and use "
                "'Change password', or register a new account with a different number.")
    if intent == "mech_quote":
        return ("Mechanics: log in on the Mechanic page, open a request, enter parts, labour, paint, days and warranty, and press "
                "SUBMIT / UPDATE QUOTATION. You can withdraw or update it any time before the customer selects you.")
    if intent == "choose_quote":
        if quotes:
            cheap = min(quotes, key=lambda x: x["amount"])
            fast = min(quotes, key=lambda x: x["days"] or 999)
            return (f"The lowest quote is {cheap['name']} at Rs {int(cheap['amount']):,}. The fastest is {fast['name']} "
                    f"({fast['days'] or '?'} days). Check the 'Recommended' badge, which balances price, speed and rating, "
                    "and use the comparison table.")
        return ("Quotations appear on the Quotations page once mechanics respond. Compare price, days, warranty and rating, and "
                "look for the 'Recommended' badge.")
    if intent == "quotes_flow":
        if quotes:
            return f"You have {len(quotes)} quotation(s). Use 'Compare side by side' to see parts, labour and paint, then select a mechanic."
        return ("After you share your report, mechanics in your area can send quotations. Open the Quotations page, enter your "
                "Report ID and phone number, and the offers will appear there with a 🔔 notice for new ones.")
    if intent == "cost":
        if have:
            return (f"For {dmg or 'this damage'}, the AI estimate is {est_txt}. It depends on the damage type, its size, whether "
                    "the part is repaired or replaced, labour and paint. Final price comes from the mechanic, so compare 2 or 3 quotes.")
        return ("Upload a photo on the Customer page to get an estimate. Cost depends on the damage type, size, repair or "
                "replacement, labour (about Rs 500 per hour) and paint. Slight scratches cost the least, severe dents the most.")
    if intent == "worth":
        if sev:
            return (f"For {sev} damage, repair is usually cheaper than replacing the panel. Ask the mechanic about paintless dent "
                    "repair or touch-up paint first. Replacement makes sense for deep cracks, holes or badly bent panels.")
        return "Repair is usually cheaper than replacement for dents and scratches; replace badly cracked or bent parts."
    if intent == "drive":
        return drive_tip(sev) if sev else ("Run the analysis first so I can advise on driving safety. In general, avoid driving with "
                                           "cracked glass, loose parts, fluid leaks or broken lights.")
    if intent == "time":
        if sev:
            a, b = repair_time(sev)
            return f"For {sev} damage, repairs usually take about {a} to {b} days. Each quotation also shows the mechanic's own estimate."
        return "Slight damage takes 1 to 2 days, medium 2 to 4 days and severe damage 5 to 10 days."
    if intent == "insurance":
        return ("For an insurance claim keep clear photos, the policy copy, RC, driving licence, an FIR for major accidents, your "
                "report ID and a mechanic quotation. The checklist on the Customer page lists everything.")
    if intent == "warranty":
        return "Ask for a written warranty on paint and parts. Each quotation shows the warranty the mechanic offers."
    if intent == "paint":
        return "Paint is a big share of the bill. Quotations show paint separately so you can compare it between mechanics."
    if intent == "scratch":
        return ("Light scratches can often be polished or touched up cheaply. Deep scratches that reach the metal need sanding and "
                "repainting to prevent rust.")
    if intent == "dent":
        return ("Many dents can be pulled out without repainting (paintless dent repair). Large creases or stretched metal need "
                "panel beating and paint, and severe ones may need a new panel.")
    if intent == "crack":
        return ("Cracked windshields should be checked quickly: small chips can be resin-repaired, larger cracks need replacement. "
                "Cracks or holes in body panels or bumpers can be welded or patched, or the part is replaced.")
    if intent == "lamp":
        return "Broken lamps and bumpers affect safety and may fail a check, so replace them soon. Ask the mechanic for part prices."
    if intent == "severity":
        return ("Slight = cosmetic (scratches, small dents). Medium = needs panel work and paint. Severe = major deformation, "
                "likely panel replacement and a full inspection. It is based on the damage type, its size and how many damages there are.")
    if intent == "accuracy":
        return ("The AI only recognises the 7 damage types it was trained on, and a confidence score is not a guarantee. Treat the "
                "estimate as a guide and confirm it with a mechanic after a physical inspection.")
    if intent == "review":
        return "When your repair status reaches Ready, the Quotations page lets you rate the workshop from 1 to 5 stars."
    if intent == "language":
        return "Open the ⋮ menu at the top right and choose English, ಕನ್ನಡ or हिन्दी under Language."
    if intent == "delete":
        return "On the Quotations page open your report and use '🗑 Delete this report'. It removes the report, quotes and messages for good."
    if intent == "message":
        return ("Use '💬 Message this workshop' on a quotation to chat with the mechanic. Phone numbers are shared automatically "
                "once you select a mechanic, then you can call or WhatsApp them.")
    if intent == "map":
        return "Open '🗺 Workshops map' on the Customer page, or '🗺 Where are these workshops?' on Quotations. You can also sort quotes by Nearest."
    if intent == "pdf":
        return ("After sharing your report you can download the PDF, show the QR code or share on WhatsApp. On the Quotations page "
                "you can download the complete repair journey PDF.")
    if intent == "trouble":
        return ("Sorry about that. Tell me where it happens: photo upload, analysis, sharing the report, login, quotations or the "
                "dashboard. Try refreshing the page once. If a red error message appears, tell me what it says.")
    return None


def _assistant_fallback(q, ctx):
    short = q.strip()[:60]
    if ctx.get("min_cost") is not None:
        hint = "You can ask: “Is this repair worth it?”, “How long will it take?”, “Is it safe to drive?” or “Which quote should I choose?”."
    else:
        hint = "You can ask: “How do I start a repair?”, “How does this app work?”, “What does severity mean?” or “Why is my photo not detected?”."
    return (f"I'm not sure I understood “{short}”. I can help with repair cost, repair time, driving safety, insurance, quotes, "
            f"photo tips and how to use the app. {hint}")


def _best_intent(q):
    """Topic that best matches a typed or spoken question (None if nothing matches)."""
    t = " " + re.sub(r"\s+", " ", q.lower().strip()) + " "
    best, best_score = None, 0
    for name, kws in _INTENTS:
        sc = _kw_score(t, kws)
        if sc > best_score:
            best, best_score = name, sc
    return best


def assistant_rules(q, ctx):
    """Offline assistant: scores every topic against the question and answers with the best match.
    Falls back to guidance (never silently repeats one canned answer)."""
    best = _best_intent(q)
    if best:
        ans = _assistant_answer(best, ctx, q)
        if ans:
            return ans
    return _assistant_fallback(q, ctx)


def assistant_llm(q, ctx, history):
    """Optional: uses Claude if ANTHROPIC_API_KEY is set in Streamlit secrets. Falls back to rules on any error."""
    key = _secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    system = ("You are the AutoVision AI assistant for vehicle damage repair in India (Bengaluru). "
              "Answer in 2 to 4 short sentences, in simple English, with prices in Rs. Estimates are indicative and must be confirmed "
              "by a mechanic. Context about the user's report and quotations (JSON): " + json.dumps(ctx, default=str))
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    if not msgs or msgs[-1]["content"] != q:
        msgs.append({"role": "user", "content": q})
    body = json.dumps({"model": _secret("ANTHROPIC_MODEL") or "claude-haiku-5-5", "max_tokens": 400,
                       "system": system, "messages": msgs}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    return "".join(b.get("text", "") for b in data.get("content", [])).strip() or None


def assistant_reply(q, ctx, history):
    try:
        answer = assistant_llm(q, ctx, history)
        if answer:
            return answer
    except Exception:
        pass
    return assistant_rules(q, ctx)


# Tap-only assistant: the user picks from fixed options, so every choice has a clear answer.
ASSIST_LABELS = {
    "start": "🚗 I want to repair my car",
    "worth": "✅ Is this repair worth it?",
    "cost": "💰 How much will it cost?",
    "time": "🕒 How long will the repair take?",
    "drive": "🚦 Is it safe to drive?",
    "choose_quote": "🧾 Which quote should I choose?",
    "quotes_flow": "📨 Where do I see mechanic quotes?",
    "upload": "📸 Photo tips / damage not detected",
    "insurance": "🛡 Insurance claim help",
    "howapp": "ℹ️ How does this app work?",
    "dashboard": "📊 Dashboard help",
    "severity": "⚠️ What does severity mean?",
    "accuracy": "🎯 How accurate is the AI?",
    "warranty": "🔧 Warranty on repairs",
    "reportid": "🔑 I lost my Report ID",
    "account": "👤 Mechanic login / register",
    "mech_quote": "📝 How do mechanics send quotes?",
    "message": "💬 How do I contact a mechanic?",
    "map": "🗺 Find workshops near me",
    "language": "🌐 Change language",
    "delete": "🗑 Delete my report",
    "pdf": "📄 Download / share my report",
    "review": "⭐ Rate the workshop",
    "paint": "🎨 About paint work",
    "scratch": "〰️ About scratches",
    "dent": "🔨 About dents",
    "crack": "🪟 Cracks, holes and windshield",
    "lamp": "💡 Lamps and bumpers",
}
ASSIST_MAIN = ["start", "cost", "time", "drive", "choose_quote", "upload", "insurance", "howapp", "dashboard"]
ASSIST_MORE = ["severity", "accuracy", "warranty", "reportid", "account", "mech_quote", "message", "map",
               "pdf", "review", "language", "delete", "paint", "scratch", "dent", "crack", "lamp"]
ASSIST_NEXT = {   # follow-up options shown after each answer
    "start": ["upload", "cost", "howapp"], "worth": ["cost", "time", "choose_quote"],
    "cost": ["time", "worth", "choose_quote"], "time": ["cost", "drive", "insurance"],
    "drive": ["time", "cost", "insurance"], "choose_quote": ["message", "warranty", "map"],
    "quotes_flow": ["choose_quote", "message", "reportid"], "upload": ["start", "accuracy", "howapp"],
    "insurance": ["pdf", "cost", "warranty"], "howapp": ["start", "dashboard", "account"],
    "dashboard": ["howapp", "account", "start"], "severity": ["cost", "time", "drive"],
    "accuracy": ["upload", "cost", "severity"], "warranty": ["choose_quote", "paint", "insurance"],
    "reportid": ["quotes_flow", "pdf", "start"], "account": ["mech_quote", "howapp", "dashboard"],
    "mech_quote": ["account", "message", "dashboard"], "message": ["choose_quote", "map", "quotes_flow"],
    "map": ["choose_quote", "message", "quotes_flow"], "language": ["howapp", "start", "dashboard"],
    "delete": ["reportid", "start", "pdf"], "pdf": ["insurance", "quotes_flow", "reportid"],
    "review": ["warranty", "choose_quote", "howapp"], "paint": ["cost", "scratch", "warranty"],
    "scratch": ["cost", "paint", "dent"], "dent": ["cost", "worth", "time"],
    "crack": ["drive", "cost", "time"], "lamp": ["drive", "cost", "insurance"],
}


def _assistant_pick(key, intent, ctx):
    hist = st.session_state.setdefault(f"chat_{key}", [])
    hist.append({"role": "user", "content": ASSIST_LABELS[intent]})
    hist.append({"role": "assistant", "content": _assistant_answer(intent, ctx, "") or "Please ask a mechanic about this."})
    st.session_state[f"assist_last_{key}"] = intent
    st.session_state[f"assist_more_{key}"] = False


def render_assistant(key, ctx):
    """Tap-only assistant: no typing, the user chooses from a short list of questions."""
    hist_key, last_key, more_key = f"chat_{key}", f"assist_last_{key}", f"assist_more_{key}"
    st.session_state.setdefault(hist_key, [])
    hist = st.session_state[hist_key]
    last = st.session_state.get(last_key)
    show_more = st.session_state.get(more_key, False)

    for m in hist[-6:]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    if last:
        st.markdown("**Related questions**")
        options = ASSIST_NEXT.get(last, ASSIST_MAIN[:3])
    elif show_more:
        st.markdown("**More topics**")
        options = ASSIST_MORE
    else:
        st.markdown("**Choose a question**")
        options = list(ASSIST_MAIN)
        if ctx.get("min_cost") is not None:               # photo already analysed: put the personal questions first
            options = ["worth"] + [o for o in options if o != "start"][:8]
        if ctx.get("quotes"):
            options = ["choose_quote"] + [o for o in options if o != "choose_quote"][:8]

    cols = st.columns(2)
    for i, intent in enumerate(options):
        if cols[i % 2].button(ASSIST_LABELS[intent], key=f"opt_{key}_{intent}_{last}_{show_more}",
                              use_container_width=True):
            _assistant_pick(key, intent, ctx)
            st.rerun()

    nav = st.columns(2)
    if not last and not show_more:
        if nav[0].button("➕ More topics", key=f"more_{key}", use_container_width=True):
            st.session_state[more_key] = True
            st.rerun()
    else:
        if nav[0].button("🏠 Main menu", key=f"home_{key}", use_container_width=True):
            st.session_state[last_key] = None
            st.session_state[more_key] = False
            st.rerun()
    if hist and nav[1].button("🧹 Clear chat", key=f"clear_{key}", use_container_width=True):
        st.session_state[hist_key] = []
        st.session_state[last_key] = None
        st.session_state[more_key] = False
        st.rerun()

    st.markdown("**Or type / speak your own question**")
    voice_lang = {"English": "en-IN", "ಕನ್ನಡ": "kn-IN", "हिन्दी": "hi-IN"}.get(st.session_state.get("lang", "English"), "en-IN")
    components.html(VOICE_HTML.replace("__LANG__", voice_lang), height=48)
    with st.form(f"assistant_form_{key}", clear_on_submit=True):
        question = st.text_input(tr("Ask about your repair…"), key=f"assistant_q_{key}")
        sent = st.form_submit_button(tr("SEND"), use_container_width=True)
    if sent and question.strip():
        q = question.strip()
        hist.append({"role": "user", "content": q})
        with st.spinner("Thinking…"):
            answer = assistant_reply(q, ctx, hist)
        hist.append({"role": "assistant", "content": answer})
        st.session_state[last_key] = _best_intent(q)      # show related options if the topic was understood
        st.session_state[more_key] = False
        st.rerun()


def render_admin_dashboard():
    import pandas as pd
    admin_pw = _secret("ADMIN_PASSWORD") or "admin123"
    if not st.session_state.get("admin_ok"):
        with st.form("admin_login_form"):
            pw = st.text_input(tr("Admin password"), type="password")
            if st.form_submit_button(tr("OPEN DASHBOARD"), use_container_width=True):
                if pw == admin_pw:
                    st.session_state.admin_ok = True
                    st.rerun()
                else:
                    st.error(tr("Incorrect password."))
        st.caption("Default password is admin123. Set ADMIN_PASSWORD in Streamlit secrets to change it.")
        return

    conn = db_connect()
    reports = pd.read_sql_query("""SELECT id, created, area, vehicle, damage, severity, min_cost, max_cost,
                                   COALESCE(repair_status, 'Open') AS repair_status FROM reports""", conn)
    n_mech = conn.execute("SELECT COUNT(*) FROM mechanics").fetchone()[0]
    n_quotes = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
    avg_quote = conn.execute("SELECT COALESCE(AVG(amount), 0) FROM quotes").fetchone()[0]
    mechs = pd.read_sql_query("""SELECT m.name AS Workshop, m.area AS Area, COUNT(q.id) AS Quotes,
                                 COALESCE(SUM(q.selected), 0) AS Won, ROUND(COALESCE(AVG(q.amount), 0)) AS "Avg quote",
                                 (SELECT ROUND(AVG(rating), 1) FROM reviews r WHERE r.mechanic_id = m.id) AS Rating
                                 FROM mechanics m LEFT JOIN quotes q ON q.mechanic_id = m.id
                                 GROUP BY m.id ORDER BY Won DESC, Quotes DESC""", conn)
    conn.close()

    avg_est = ((reports["min_cost"].fillna(0) + reports["max_cost"].fillna(0)) / 2).mean() if len(reports) else 0
    cols = st.columns(3)
    for col, label, val in [(cols[0], "TOTAL REPORTS", len(reports)), (cols[1], "WORKSHOPS", n_mech),
                            (cols[2], "QUOTES SENT", n_quotes)]:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                     f"<div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    done = int((reports["repair_status"] == "Ready").sum()) if len(reports) else 0
    for col, label, val in [(cols[0], "REPAIRS READY", done), (cols[1], "AVG AI ESTIMATE", money(avg_est)),
                            (cols[2], "AVG QUOTE", money(avg_quote))]:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                     f"<div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

    if reports.empty:
        st.info("No reports yet. Charts appear after customers submit reports.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(tr("**Most common damage**"))
            st.bar_chart(reports["damage"].str.replace("_", " ").str.title().value_counts())
        with c2:
            st.markdown(tr("**Severity split**"))
            st.bar_chart(reports["severity"].value_counts())
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(tr("**Reports by area**"))
            st.bar_chart(reports["area"].value_counts())
        with c4:
            st.markdown(tr("**Repair status**"))
            order = ["Open", "Selected", "Car dropped", "In repair", "Ready"]
            st.bar_chart(reports["repair_status"].value_counts().reindex(order, fill_value=0))
        days = pd.to_datetime(pd.Series([parse_ts(v) for v in reports["created"]]), errors="coerce").dropna().dt.date
        if len(days):
            st.markdown(tr("**Reports per day**"))
            st.line_chart(days.value_counts().sort_index())
        st.markdown(tr("**Recent reports**"))
        st.dataframe(reports.sort_values("id", ascending=False).head(10), use_container_width=True, hide_index=True)
        st.download_button(tr("⬇ EXPORT ALL REPORTS (CSV)"), reports.to_csv(index=False), "autovision_reports.csv", "text/csv")

    conn = db_connect()
    acc = pd.read_sql_query("""SELECT r.min_cost, r.max_cost, q.amount FROM quotes q
                               JOIN reports r ON r.id = q.report_id WHERE q.selected = 1""", conn)
    conn.close()
    st.markdown("**🎯 AI estimate vs accepted price**")
    if acc.empty:
        st.caption("This appears after customers select a mechanic.")
    else:
        lo_s, hi_s = acc["min_cost"].fillna(0), acc["max_cost"].fillna(0)
        mid_s = (lo_s + hi_s) / 2
        within = ((acc["amount"] >= lo_s) & (acc["amount"] <= hi_s)).mean() * 100
        err = ((acc["amount"] - mid_s).abs() / mid_s.replace(0, np.nan)).mean() * 100
        err = 0 if pd.isna(err) else err
        ac = st.columns(3)
        for col, label, val in [(ac[0], "ACCEPTED QUOTES", len(acc)), (ac[1], "INSIDE AI RANGE", f"{within:.0f}%"),
                                (ac[2], "AVG DIFFERENCE", f"{err:.0f}%")]:
            col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                         f"<div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown(tr("**Workshop leaderboard**"))
    st.dataframe(mechs, use_container_width=True, hide_index=True)
    st.markdown("**🗄 Database backup and restore**")
    try:
        with open(DB_PATH, "rb") as fh:
            st.download_button("⬇ DOWNLOAD DATABASE BACKUP", fh.read(), "vehicle_damage_backup.db",
                               "application/octet-stream", key="db_backup")
    except Exception as exc:
        st.caption(f"Backup unavailable: {exc}")
    restore = st.file_uploader("Restore from a backup (.db)", type=["db", "sqlite", "sqlite3"], key="restore_db")
    if restore and st.button("RESTORE DATABASE", key="restore_btn"):
        data = restore.getvalue()
        ok = data[:16] == b"SQLite format 3\x00"
        if ok:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            tmp.write(data)
            tmp.close()
            try:
                chk = sqlite3.connect(tmp.name)
                names = {r[0] for r in chk.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                chk.close()
                ok = {"reports", "mechanics", "quotes"} <= names
            except Exception:
                ok = False
            if ok:
                shutil.copy(tmp.name, DB_PATH)
                migrate_database()
                st.success("Database restored. Refresh the page.")
        if not ok:
            st.error("This is not a valid AutoVision backup file.")
    if st.button(tr("Log out of admin"), key="admin_logout"):
        st.session_state.admin_ok = False
        st.rerun()


# ------------------------- MAP / DISTANCE / MESSAGES -------------------------

AREA_COORDS = {
    "Whitefield": (12.9698, 77.7500), "Indiranagar": (12.9784, 77.6408), "Koramangala": (12.9352, 77.6245),
    "HSR Layout": (12.9116, 77.6474), "Jayanagar": (12.9250, 77.5938), "Electronic City": (12.8452, 77.6602),
    "Marathahalli": (12.9591, 77.6974), "BTM Layout": (12.9166, 77.6101), "Yelahanka": (13.1007, 77.5963),
    "Hebbal": (13.0358, 77.5970), "Malleshwaram": (13.0035, 77.5643), "JP Nagar": (12.9063, 77.5857),
    "Banashankari": (12.9255, 77.5468), "Rajajinagar": (12.9916, 77.5521),
}


def area_coords(area):
    a = (area or "").strip().lower()
    if not a:
        return None
    for name, coords in AREA_COORDS.items():
        n = name.lower()
        if n == a or n in a or a in n:
            return coords
    return None


def area_distance(a, b):
    """Approximate straight-line distance in km between two Bengaluru areas."""
    ca, cb = area_coords(a), area_coords(b)
    if not ca or not cb:
        return None
    from math import radians, sin, cos, asin, sqrt
    la1, lo1, la2, lo2 = map(radians, [ca[0], ca[1], cb[0], cb[1]])
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return round(2 * 6371 * asin(sqrt(h)), 1)


def render_workshop_map(mech_ids=None, customer_area=None, area_filter=None):
    import pandas as pd
    conn = db_connect()
    rows = conn.execute("SELECT id, name, area, available FROM mechanics").fetchall()
    conn.close()
    pts = []
    for r in rows:
        if mech_ids is not None and r["id"] not in mech_ids:
            continue
        if area_filter and area_filter.lower() not in (r["area"] or "").lower():
            continue
        c = area_coords(r["area"])
        if not c:
            continue
        off = ((r["id"] * 37) % 20 - 10) * 0.0006      # small offset so workshops do not overlap
        pts.append({"lat": c[0] + off, "lon": c[1] - off, "name": r["name"], "area": r["area"],
                    "color": "#ff303b" if r["available"] != 0 else "#888888", "size": 140})
    me = area_coords(customer_area) if customer_area else None
    if me:
        pts.append({"lat": me[0], "lon": me[1], "name": "You", "area": customer_area,
                    "color": "#2f80ff", "size": 260})
    if not pts:
        st.info(tr("No workshops with a known area to show yet."))
        return
    df = pd.DataFrame(pts)
    try:
        st.map(df, latitude="lat", longitude="lon", color="color", size="size")
    except TypeError:
        st.map(df[["lat", "lon"]])
    st.caption("🔴 Workshops   🔵 You")
    shops = df[df["name"] != "You"][["name", "area"]]
    if len(shops):
        st.dataframe(shops.rename(columns={"name": "Workshop", "area": "Area"}),
                     use_container_width=True, hide_index=True)


def render_thread(report_id, mechanic_id, role, key, block_phone=True):
    """Customer <-> mechanic messages for one report. role is 'customer' or 'mechanic'."""
    conn = db_connect()
    rows = conn.execute("""SELECT sender, body, created FROM messages
                           WHERE report_id = ? AND mechanic_id = ? ORDER BY id""",
                        (report_id, mechanic_id)).fetchall()
    conn.close()
    if not rows:
        st.caption(tr("No messages yet. Ask a question or share details about the damage."))
    other = "Mechanic" if role == "customer" else "Customer"
    for r in rows[-20:]:
        mine = r["sender"] == role
        with st.chat_message("user" if mine else "assistant"):
            st.caption(f"{'You' if mine else other} • {pretty_ts(r['created'])}")
            st.text(r["body"])
    with st.form(f"msg_form_{key}", clear_on_submit=True):
        body = st.text_area(tr("Your message"), max_chars=500, height=80, key=f"msg_body_{key}")
        send = st.form_submit_button(tr("SEND MESSAGE"), use_container_width=True)
    if send:
        text = body.strip()
        if not text:
            st.warning(tr("Type a message first."))
        elif block_phone and re.search(r"\d{10}", re.sub(r"[\s\-\+\.]", "", text)):
            st.error(tr("For privacy, phone numbers are shared automatically once a mechanic is selected."))
        else:
            conn = db_connect()
            conn.execute("INSERT INTO messages (report_id, mechanic_id, sender, body, created) VALUES (?, ?, ?, ?, ?)",
                         (report_id, mechanic_id, role, text, now_ist()))
            conn.commit()
            conn.close()
            st.rerun()


# ------------------------- EXTRAS: QR / EMAIL / PHOTOS / VOICE / THEME -------------------------

LIGHT_CSS = """<style>
.stApp, [data-testid="stAppViewContainer"], .main { background:#f5f2f2 !important; color:#18181b !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp li { color:#18181b !important; }
.panel, .portal-card, .metric-card, .res-card, .id-card, .st, [data-testid="stExpander"], [data-testid="stForm"], [data-testid="stPopoverBody"] { background:#ffffff !important; border-color:#e3c9cd !important; }
.metric-value, .res-cost, .id-num { color:#c40f1d !important; }
input, textarea, [data-baseweb="select"] > div { background:#ffffff !important; color:#18181b !important; }
.stButton button, .stDownloadButton button, .stFormSubmitButton button { color:#ffffff !important; }
.hero, .hero * { color:#ffffff !important; }
</style>"""

VOICE_HTML = """<div style="font-family:sans-serif;display:flex;gap:10px;align-items:center">
<button id="mic" style="background:#c40f1d;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer">🎤 Speak</button>
<span id="st" style="color:#aaa;font-size:13px"></span></div>
<script>
const b=document.getElementById('mic'), s=document.getElementById('st');
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!SR){s.textContent='Voice input works in Chrome or Edge';b.disabled=true;}
else{const r=new SR();r.lang='__LANG__';r.interimResults=false;
 b.onclick=()=>{s.textContent='Listening…';try{r.start();}catch(e){}};
 r.onresult=e=>{const t=e.results[0][0].transcript;s.textContent='“'+t+'” - now press SEND';
  let el=null;window.parent.document.querySelectorAll('input[type=text]').forEach(i=>{
   const l=(i.getAttribute('aria-label')||'');
   if(l.startsWith('Ask about your repair')||l.startsWith('ನಿಮ್ಮ ರಿಪೇರಿ')||l.startsWith('अपनी मरम्मत'))el=i;});
  if(el){const set=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
   set.call(el,t);el.dispatchEvent(new Event('input',{bubbles:true}));}};
 r.onerror=e=>{s.textContent='Mic error: '+e.error;};}
</script>"""


def make_qr_png(text):
    try:
        import qrcode
        buf = BytesIO()
        qrcode.make(text).save(buf, "PNG")
        return buf.getvalue()
    except Exception:
        return None


def report_qr_text(rep):
    return (f"AutoVision Report #{rep['id']} | {rep['vehicle']} | "
            f"{str(rep['damage']).replace('_', ' ')} | {rep['severity']} | "
            f"AI est Rs {int(rep['min_cost'] or 0)}-{int(rep['max_cost'] or 0)}")


def send_email(to, subject, body):
    """Optional: works only when SMTP_HOST, SMTP_USER and SMTP_PASSWORD are set in Streamlit secrets."""
    host, user, pw = _secret("SMTP_HOST"), _secret("SMTP_USER"), _secret("SMTP_PASSWORD")
    if not (host and user and pw and to):
        return False
    try:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = _secret("SMTP_FROM") or user, to, subject
        msg.set_content(body)
        with smtplib.SMTP(host, int(_secret("SMTP_PORT") or 587), timeout=15) as server:
            server.starttls()
            server.login(user, pw)
            server.send_message(msg)
        return True
    except Exception:
        return False


def compress_image(data):
    img = load_image(data)
    img.thumbnail((1000, 1000))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def save_photo(report_id, mechanic_id, kind, data):
    conn = db_connect()
    conn.execute("INSERT INTO photos (report_id, mechanic_id, kind, image, created) VALUES (?, ?, ?, ?, ?)",
                 (report_id, mechanic_id, kind, compress_image(data), now_ist()))
    conn.commit()
    conn.close()


def get_photos(report_id, kind=None):
    conn = db_connect()
    if kind:
        rows = conn.execute("SELECT id, kind, image, created FROM photos WHERE report_id = ? AND kind = ? ORDER BY id",
                            (report_id, kind)).fetchall()
    else:
        rows = conn.execute("SELECT id, kind, image, created FROM photos WHERE report_id = ? ORDER BY id",
                            (report_id,)).fetchall()
    conn.close()
    return rows


@st.cache_resource(show_spinner=False)
def _load_general_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")          # general model, downloaded automatically on first use


def looks_like_vehicle(image_bytes):
    """True / False if a general object model sees a vehicle, None if the check is unavailable."""
    try:
        model = _load_general_model()
        img = load_image(image_bytes)
        img.thumbnail((640, 640))
        res = model.predict(source=np.array(img), conf=0.25, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return False
        return any(int(c) in (2, 3, 5, 7) for c in res.boxes.cls.tolist())      # car, motorcycle, bus, truck
    except Exception:
        return None


# ------------------------- PHONE IMAGE HANDLING -------------------------

try:                                                 # HEIC / HEIF support (iPhone and some Android cameras)
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

PHONE_TYPES = ["jpg", "jpeg", "jfif", "png", "webp", "bmp", "heic", "heif"]


@st.cache_data(show_spinner=False, max_entries=24)
def prepare_image(raw):
    """Return (jpeg_bytes, width, height): upright, RGB, at most 1600 px. Raises ValueError with a clear message."""
    if not raw:
        raise ValueError("The file is empty. Please select the photo again.")
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception:
        hint = "" if HEIF_OK else " HEIC/HEIF photos need the pillow-heif package; set the phone camera to JPEG or add pillow-heif."
        raise ValueError("This file could not be read as an image." + hint)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), img.width, img.height


def go_to_quotes():
    """Button callback: open Quotations for the report the customer just shared (no retyping)."""
    last = st.session_state.get("last_report")
    if last:
        st.session_state.quotation_report_id = last["id"]
    st.session_state.page = "Quotations"


# ------------------------- CUSTOMER PAGE -------------------------

def render_customer():
    st.markdown('<div class="section-heading">CUSTOMER WORKSPACE</div>', unsafe_allow_html=True)
    st.title(tr("Vehicle Damage Inspection"))
    st.write("Upload a clear vehicle image and let the trained AI model analyse visible damage.")
    st.info("👤 **For vehicle owners.** Upload a photo → check the AI result → share your report. "
            "Then open **Quotations** to compare mechanic offers. You do not need to register.")

    with st.expander(tr("📂 My previous reports")):
        hist_phone = st.text_input(tr("Enter your 10-digit phone number"), key="hist_phone")
        if hist_phone.strip().isdigit() and len(hist_phone.strip()) == 10:
            conn = db_connect()
            rows = conn.execute("""SELECT r.id, r.created, r.vehicle, r.damage, r.severity, r.repair_status,
                (SELECT COUNT(*) FROM quotes q WHERE q.report_id = r.id) AS qcount
                FROM reports r WHERE r.phone = ? ORDER BY r.id DESC""", (hist_phone.strip(),)).fetchall()
            conn.close()
            if rows:
                st.dataframe([{"Report ID": r["id"], "Date": pretty_ts(r["created"]), "Vehicle": r["vehicle"],
                               "Damage": str(r["damage"]).replace("_", " ").title(), "Severity": r["severity"],
                               "Quotes": r["qcount"], "Repair status": r["repair_status"] or "Open"}
                              for r in rows], use_container_width=True, hide_index=True)
            else:
                st.info(tr("No reports found for this number."))

    with st.expander(tr("🧾 Insurance claim checklist")):
        st.markdown("""- Clear photos of the damage from 3 angles and a wide shot of the whole car
- Copy of your insurance policy and the vehicle RC (registration certificate)
- Driving licence of the driver at the time of the incident
- FIR copy if there was a major accident, theft or third-party damage
- Your AutoVision report ID and PDF report
- Repair estimate or quotation from a workshop""")

    st.session_state.setdefault("up_n", 0)
    analysis = st.session_state.get("customer_analysis")
    last = st.session_state.get("last_report")

    with st.expander(tr("🚗 My Garage")):
        g_phone = st.text_input(tr("Phone number for My Garage"), key="garage_phone")
        if g_phone.strip().isdigit() and len(g_phone.strip()) == 10:
            conn = db_connect()
            g_rows = conn.execute("SELECT id, vehicle FROM garage WHERE phone = ? ORDER BY id", (g_phone.strip(),)).fetchall()
            conn.close()
            st.session_state.garage_list = [r["vehicle"] for r in g_rows]
            for r in g_rows:
                gc1, gc2 = st.columns([4, 1])
                gc1.write("🚗 " + r["vehicle"])
                if gc2.button(tr("Remove"), key=f"gdel_{r['id']}"):
                    conn = db_connect()
                    conn.execute("DELETE FROM garage WHERE id = ?", (r["id"],))
                    conn.commit()
                    conn.close()
                    st.rerun()
            with st.form("garage_add_form", clear_on_submit=True):
                gm1, gm2 = st.columns(2)
                g_make = gm1.selectbox("Vehicle make", CAR_MAKES, key="garage_make")
                g_model = gm2.text_input("Vehicle model", key="garage_model")
                add_v = st.form_submit_button(tr("ADD VEHICLE"), use_container_width=True)
            if add_v and g_model.strip():
                conn = db_connect()
                conn.execute("INSERT OR IGNORE INTO garage (phone, vehicle) VALUES (?, ?)",
                             (g_phone.strip(), f"{g_make} {g_model.strip()}"))
                conn.commit()
                conn.close()
                st.rerun()
        else:
            st.caption("Enter your 10-digit phone number to see and save your vehicles.")

    chat_ctx = {}
    if analysis and analysis.get("detections"):
        _e = get_estimate(analysis["detections"])
        chat_ctx = {"damage": _e["primary"].replace("_", " "), "severity": _e["severity"],
                    "min_cost": _e["min"], "max_cost": _e["max"]}
    with st.expander(tr("💬 Ask the AutoVision assistant")):
        render_assistant("customer", chat_ctx)
    with st.expander(tr("🗺 Workshops map")):
        pick = st.selectbox(tr("Show workshops in"), ["All areas"] + BENGALURU_AREAS[:-1], key="map_area_pick")
        render_workshop_map(area_filter=None if pick == "All areas" else pick,
                            customer_area=None if pick == "All areas" else pick)

    left, right = st.columns([1, 1.05], gap="large")

    with left:
        upload = st.file_uploader(tr("Upload vehicle image"), type=PHONE_TYPES,
                                  key=f"customer_vehicle_upload_{st.session_state.up_n}")
        st.caption("Tap the box, then choose Gallery / Photos or Camera. JPG, PNG, WEBP and HEIC photos are supported.")
        with st.expander("📷 Use the camera instead"):
            cam_file = st.camera_input("Take a photo of the damage", key=f"customer_camera_{st.session_state.up_n}")
        source = upload if upload is not None else cam_file
        confidence = st.slider(tr("Detection confidence"), 0.25, 0.90, 0.50, 0.05,
                               help="A higher threshold may reduce weak detections but can miss real damage.")
        extra_files = st.file_uploader(tr("➕ Add more angles (optional, up to 3)"), type=PHONE_TYPES,
                                       accept_multiple_files=True,
                                       key=f"customer_extra_upload_{st.session_state.up_n}") or []
        extra_files = extra_files[:3]
        image_bytes, upload_error, file_id, img_w, img_h = None, None, None, 0, 0
        src_name = getattr(source, "name", "photo.jpg") if source else None
        if source is not None:
            try:
                image_bytes, img_w, img_h = prepare_image(source.getvalue())
                file_id = hashlib.md5(image_bytes).hexdigest()
            except ValueError as exc:
                upload_error = str(exc)
        extra_prepared = []
        for f in extra_files:
            try:
                extra_prepared.append((f.name, prepare_image(f.getvalue())[0]))
            except ValueError as exc:
                st.warning(f"{f.name}: {exc}")
        analysed_now = bool(analysis and image_bytes and analysis.get("file_id") == file_id)

        active = 3 if last else 2 if analysed_now else 1 if image_bytes else 0
        step_bar([tr("Upload"), tr("Analyse"), tr("Share report")], active)

        if upload_error:
            st.error(f"❌ {upload_error}")
        elif image_bytes:
            st.success(f"✅ Photo ready: {img_w}×{img_h}px, {len(image_bytes) // 1024} KB")
            for note in check_quality(image_bytes):
                st.warning(f"Photo looks {note}. A clearer photo gives a better result.")
            if not analysed_now:
                st.image(image_bytes, caption=f"Uploaded vehicle image ({src_name})", use_container_width=True)

        if st.button(tr("⚡ ANALYSE VEHICLE DAMAGE"), key="analyse_vehicle"):
            if upload_error:
                st.error(f"❌ {upload_error}")
            elif not image_bytes:
                st.warning(tr("Upload a vehicle image first."))
            else:
                try:
                    with st.spinner("AI is scanning your vehicle..."):
                        shots = []
                        for nm, data in [(src_name, image_bytes)] + extra_prepared:
                            _, ann, dets = analyse_vehicle(data, confidence)
                            cost = get_estimate(dets)["max"] if dets else 0
                            shots.append({"name": nm, "bytes": data, "annotated": ann, "detections": dets, "cost": cost})
                        main_idx = max(range(len(shots)), key=lambda i: shots[i]["cost"])
                        main = shots[main_idx]
                    st.session_state.customer_analysis = {
                        "file_name": src_name, "file_id": file_id, "image_bytes": main["bytes"],
                        "annotated": main["annotated"], "detections": main["detections"],
                        "shots": shots, "main_idx": main_idx}
                    st.session_state.pop("last_report", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

        if analysed_now and st.button(tr("START NEW INSPECTION"), key="new_inspection"):
            st.session_state.pop("customer_analysis", None)
            st.session_state.pop("last_report", None)
            st.session_state.up_n += 1
            st.rerun()

    with right:
        if not analysed_now:
            st.markdown("""<div class="panel" style="min-height:300px;text-align:center;
            display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:50px;">🛡️</div><h3>AI Inspection Console</h3>
            <p class="small-note">Your annotated image, detected damage, severity and
            repair estimate will appear here.</p></div>""", unsafe_allow_html=True)
            return

        st.markdown(tr("### AI Inspection Result"))
        if not analysis["detections"]:
            if looks_like_vehicle(analysis["image_bytes"]) is False:
                st.error("This photo may not show a vehicle. Please upload a clear photo of the damaged car.")
            st.warning("No supported damage was detected. This does not prove the vehicle is "
                       "undamaged. Try a clearer image or arrange a manual inspection.")
            return

        tab_boxes, tab_heat = st.tabs(["Detection boxes", "Damage heatmap"])
        with tab_boxes:
            st.image(analysis["annotated"], channels="BGR", caption="Detected damage areas",
                     use_container_width=True)
        with tab_heat:
            if "heatmap" not in analysis:
                analysis["heatmap"] = make_heatmap(analysis["image_bytes"], analysis["detections"])
            st.image(analysis["heatmap"], caption="Damage heatmap", use_container_width=True)
            st.caption("Red = highest damage concentration. Brighter colour = more confident detection.")

        shots = analysis.get("shots") or []
        if len(shots) > 1:
            m_idx = analysis.get("main_idx", 0)
            st.caption(f"Estimate is based on the most severe angle: {shots[m_idx]['name']}")
            with st.expander(tr("📸 Other angles")):
                for i, sh in enumerate(shots):
                    if i == m_idx:
                        continue
                    st.image(sh["annotated"], channels="BGR", caption=sh["name"], use_container_width=True)
                    st.write(", ".join(f"{d['name'].replace('_', ' ')} ({d['confidence'] * 100:.0f}%)"
                                       for d in sh["detections"]) or "No damage detected in this angle.")
        if analysis["detections"][0].get("low_conf"):
            st.warning("Damage found only at a lower confidence (0.30). Please verify it manually.")
        est = get_estimate(analysis["detections"])
        primary = est["primary"]
        severity, description = est["severity"], est["description"]
        min_cost, max_cost = est["min"], est["max"]
        kind = sev_kind(severity)

        st.markdown(
            f"<div class='res-card'><div><div class='metric-label'>DAMAGE TYPE</div>"
            f"<div class='res-big'>{primary.replace('_', ' ').title()}</div></div>"
            f"<div>{badge(severity, kind)}</div></div>"
            f"<div class='res-card'><div class='metric-label'>REPAIR ESTIMATE</div>"
            f"<div class='res-cost'>{money(min_cost)} – {money(max_cost) if max_cost else 'Manual check'}</div></div>",
            unsafe_allow_html=True)
        for item in analysis["detections"]:
            st.progress(min(item["confidence"], 1.0),
                        text=f"{item['name'].replace('_', ' ').title()} — {item['confidence'] * 100:.0f}% confidence")
        st.info(description)
        if est.get("safety"):
            st.warning("⚠️ " + est["safety"])
        if est["items"]:
            with st.expander("🧮 How this estimate was calculated"):
                st.dataframe([{"Damage": it["label"], "Detected as": it["class"].replace("_", " "),
                               "Action": it["action"], "Severity": f"{it['s'] * 100:.0f}%",
                               "Estimate": f"{money(it['lo'])} – {money(it['hi'])}"} for it in est["items"]],
                             use_container_width=True, hide_index=True)
                st.caption(f"Based on the damaged part, the size of each damage, repair vs replacement, labour "
                           f"(Rs {LABOUR_RATE}/hour), paint, and the number of damages repaired together.")
        d_lo, d_hi = repair_time(severity)
        st.markdown(f"🕒 **Estimated repair time:** {d_lo} to {d_hi} days")
        st.markdown(f"🚦 **Driving advice:** {drive_tip(severity)}")
        st.caption(tr("AI cost estimates are indicative and must be confirmed by a mechanic."))

        st.markdown("---")
        st.markdown(tr("### Share report with mechanics"))
        with st.form("customer_report_submission"):
            c1, c2 = st.columns(2)
            with c1:
                customer = st.text_input(tr("Customer name"), placeholder="Your full name")
                phone = st.text_input(tr("10-digit phone number"), placeholder="9876543210",
                                      help="Your mobile number without +91. You need it again on the Quotations page.")
                make = st.selectbox(tr("Vehicle make"), CAR_MAKES)
            with c2:
                area_pick = st.selectbox(tr("Area in Bengaluru"), BENGALURU_AREAS)
                area_other = st.text_input(tr("If other, type your area"))
                model = st.text_input(tr("Vehicle model"), placeholder="Example: i20")
            saved_cars = st.session_state.get("garage_list", [])
            vsel = st.selectbox(tr("Use a saved vehicle"), ["New vehicle"] + saved_cars) if saved_cars else "New vehicle"
            email = st.text_input(tr("Email (optional, for quote alerts)"))
            save_garage = st.checkbox(tr("Save this vehicle to My Garage"), value=True)
            submitted = st.form_submit_button(tr("SHARE DAMAGE REPORT"), use_container_width=True)

        if submitted:
            area = area_other.strip() if area_pick == "Other" else area_pick
            missing = [n for n, ok in [("your name", customer.strip()), ("phone number", phone.strip()),
                                       ("area", area), ("vehicle model", model.strip() or vsel != "New vehicle")] if not ok]
            if missing:
                st.error("Please fill in: " + ", ".join(missing) + ".")
            elif not phone.strip().isdigit() or len(phone.strip()) != 10:
                st.error(tr("Enter a valid 10-digit phone number."))
            else:
                vehicle = vsel if vsel != "New vehicle" else f"{make} {model.strip()}"
                rid = insert_customer_report(customer.strip(), phone.strip(), area, vehicle,
                                             primary, severity, min_cost, max_cost,
                                             analysis["image_bytes"],
                                             email=email.strip() if "@" in email else None)
                for i, sh in enumerate(analysis.get("shots") or []):
                    if i != analysis.get("main_idx", 0):
                        save_photo(rid, None, "angle", sh["bytes"])
                if save_garage:
                    conn = db_connect()
                    conn.execute("INSERT OR IGNORE INTO garage (phone, vehicle) VALUES (?, ?)", (phone.strip(), vehicle))
                    conn.commit()
                    conn.close()
                st.session_state.last_report = {
                    "id": rid, "customer": customer.strip(), "phone": phone.strip(),
                    "area": area, "vehicle": vehicle, "damage": primary, "severity": severity,
                    "min_cost": min_cost, "max_cost": max_cost,
                    "created": now_ist(),
                    "image_bytes": analysis["image_bytes"]}
                st.rerun()

        last = st.session_state.get("last_report")
        if last:
            st.markdown(
                f"<div class='id-card'><div class='metric-label'>YOUR REPORT ID</div>"
                f"<div class='id-num'>#{last['id']}</div>"
                f"<div class='small-note'>Keep this ID and your phone number to check quotations.</div></div>",
                unsafe_allow_html=True)
            st.code(str(last["id"]), language=None)
            st.success("✅ Your report is shared with mechanics in your area.")
            st.markdown("**What happens next**  \n1. Mechanics near you send quotations  \n"
                        "2. Open **Quotations** (button below) to compare them  \n3. Select a mechanic and track the repair")
            st.button("➡ Go to Quotations", key="go_quotes_btn", on_click=go_to_quotes, use_container_width=True)
            share_msg = (f"AutoVision AI report #{last['id']}: {last['vehicle']}, "
                         f"{str(last['damage']).replace('_', ' ')}, severity {last['severity']}. "
                         f"AI estimate Rs {int(last['min_cost'] or 0):,} - Rs {int(last['max_cost'] or 0):,}.")
            st.markdown(f"<a class='link-btn' target='_blank' href='https://wa.me/?text={urlquote(share_msg)}'>💬 Share on WhatsApp</a>",
                        unsafe_allow_html=True)
            qr_png = make_qr_png(report_qr_text(last))
            if qr_png:
                st.image(qr_png, width=150, caption="Scan for report summary")
            pdf = build_pdf(last)
            if pdf:
                st.download_button(tr("⬇ DOWNLOAD PDF REPORT"), pdf, f"autovision_report_{last['id']}.pdf",
                                   "application/pdf", use_container_width=True)
            else:
                st.caption("Install reportlab (pip install reportlab) to enable PDF download.")


# ------------------------- MECHANIC PAGE -------------------------

def mechanic_report_state(report_id, mechanic_id):
    conn = db_connect()
    rows = conn.execute("SELECT mechanic_id, selected FROM quotes WHERE report_id = ?",
                        (report_id,)).fetchall()
    conn.close()
    mine = [r for r in rows if r["mechanic_id"] == mechanic_id]
    if any(r["selected"] for r in mine):
        return "Won"
    if any(r["selected"] for r in rows):
        return "Lost"
    return "Quoted" if mine else "New"


def render_mechanic():
    st.markdown('<div class="section-heading">SERVICE PARTNER WORKSPACE</div>', unsafe_allow_html=True)
    st.title(tr("Mechanic Portal"))
    st.write("Register your workshop or log in to receive customer reports and submit quotations.")
    st.info("🔧 **For workshop owners only.** Register once, then log in to see customer requests and send quotations. "
            "Customers do not need to register here.")
    step_bar(["Register", "Log in", "Send quotes", "Update repair"],
             2 if st.session_state.get("mechanic_id") is not None else 0)
    st.session_state.setdefault("mechanic_id", None)

    if st.session_state.mechanic_id is None:
        st.caption("New workshop? Use **Register Workshop** once. Already registered? Open **Mechanic Login**.")
        reg_tab, login_tab = st.tabs([tr("Register Workshop"), tr("Mechanic Login")])
        with reg_tab:
            with st.form("mechanic_registration_form"):
                name = st.text_input(tr("Mechanic / workshop name"))
                phone = st.text_input(tr("Mobile number"), placeholder="9876543210", help="10-digit mobile number without +91. You will use it to log in.")
                area = st.selectbox(tr("Workshop area"), BENGALURU_AREAS)
                area_other = st.text_input(tr("If other, type your area"))
                password = st.text_input(tr("Create password"), type="password")
                confirm = st.text_input(tr("Confirm password"), type="password")
                register = st.form_submit_button(tr("REGISTER WORKSHOP"), use_container_width=True)
            if register:
                area_final = area_other.strip() if area == "Other" else area
                if not all([name.strip(), phone.strip(), area_final, password, confirm]):
                    st.error("Please fill all fields.")
                elif not phone.strip().isdigit() or len(phone.strip()) != 10:
                    st.error("Enter a valid 10-digit mobile number.")
                elif len(password) < 6:
                    st.error("Password must contain at least 6 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    conn = db_connect()
                    try:
                        conn.execute("INSERT INTO mechanics (name, phone, area, password) VALUES (?, ?, ?, ?)",
                                     (name.strip(), phone.strip(), area_final, hash_password_new(password)))
                        conn.commit()
                        st.success("Registration successful. Please log in.")
                    except sqlite3.IntegrityError:
                        st.error("This phone number is already registered. Please log in.")
                    finally:
                        conn.close()
        with login_tab:
            with st.form("mechanic_login_form"):
                phone_login = st.text_input(tr("Registered phone number"), placeholder="9876543210", help="The mobile number you registered with.")
                password_login = st.text_input(tr("Password"), type="password")
                login = st.form_submit_button(tr("LOGIN"), use_container_width=True)
            if login:
                mech = find_mechanic_login(phone_login, password_login)
                if mech:
                    st.session_state.mechanic_id = mech["id"]
                    st.rerun()
                else:
                    st.error(tr("Incorrect phone number or password."))
        return

    mechanic = get_mechanic(st.session_state.mechanic_id)
    if mechanic is None:
        st.session_state.mechanic_id = None
        st.rerun()

    if "flash" in st.session_state:
        kind, msg = st.session_state.pop("flash")
        getattr(st, kind)(msg)

    conn = db_connect()
    rating = conn.execute("SELECT AVG(rating) a, COUNT(*) n FROM reviews WHERE mechanic_id = ?",
                          (mechanic["id"],)).fetchone()
    sent = conn.execute("SELECT COUNT(*) FROM quotes WHERE mechanic_id = ?", (mechanic["id"],)).fetchone()[0]
    won = conn.execute("SELECT COUNT(*) FROM quotes WHERE mechanic_id = ? AND selected = 1",
                       (mechanic["id"],)).fetchone()[0]
    reports = conn.execute("SELECT * FROM reports WHERE LOWER(area) LIKE ? ORDER BY id DESC",
                           (f"%{mechanic['area'].lower()}%",)).fetchall()
    conn.close()

    verified = bool((mechanic["services"] or "").strip() and (mechanic["hours"] or "").strip())
    star = f"⭐ {rating['a']:.1f} ({rating['n']})" if rating["n"] else "No ratings yet"
    st.markdown(
        f"<div class='panel'><div class='section-heading'>SIGNED-IN WORKSHOP</div>"
        f"<h2>{mechanic['name']} {badge('Verified', 'green') if verified else badge('Complete profile to get verified', 'gray')}</h2>"
        f"<p>📍 {mechanic['area']} &nbsp;•&nbsp; 📞 {mechanic['phone']} &nbsp;•&nbsp; {star}</p></div>",
        unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [(m1, "NEW REQUESTS", len(reports)), (m2, "QUOTES SENT", sent),
                            (m3, "JOBS WON", won), (m4, "WIN RATE", f"{(won / sent * 100):.0f}%" if sent else "–")]:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                     f"<div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

    conn = db_connect()
    earn = conn.execute("SELECT COALESCE(SUM(amount),0) FROM quotes WHERE mechanic_id = ? AND selected = 1",
                        (mechanic["id"],)).fetchone()[0]
    avg_q = conn.execute("SELECT COALESCE(AVG(amount),0) FROM quotes WHERE mechanic_id = ?",
                         (mechanic["id"],)).fetchone()[0]
    active_jobs = conn.execute("""SELECT COUNT(*) FROM quotes q JOIN reports r ON r.id = q.report_id
                                  WHERE q.mechanic_id = ? AND q.selected = 1
                                  AND COALESCE(r.repair_status,'Open') != 'Ready'""", (mechanic["id"],)).fetchone()[0]
    reviews = conn.execute("SELECT rating, comment FROM reviews WHERE mechanic_id = ? ORDER BY id DESC LIMIT 10",
                           (mechanic["id"],)).fetchall()
    conn.close()
    e1, e2, e3 = st.columns(3)
    for col, label, val in [(e1, "TOTAL EARNINGS", money(earn)), (e2, "AVG QUOTE", money(avg_q)),
                            (e3, "ACTIVE JOBS", active_jobs)]:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                     f"<div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

    with st.expander(tr("⭐ Customer reviews")):
        if not reviews:
            st.info("No reviews yet. They appear after customers rate your finished repairs.")
        for r in reviews:
            st.write("⭐" * int(r["rating"]) + f"  {r['comment'] or ''}")

    with st.expander(tr("Edit workshop profile")):
        with st.form("mechanic_profile_form"):
            services = st.text_input(tr("Services offered"), value=mechanic["services"] or "",
                                     placeholder="Denting, painting, bumper repair")
            hours = st.text_input(tr("Working hours"), value=mechanic["hours"] or "",
                                  placeholder="Mon–Sat, 9 AM – 7 PM")
            is_open = st.checkbox(tr("Open for new jobs"), value=(mechanic["available"] is None or bool(mechanic["available"])))
            price_list = st.text_area(tr("Price list (optional)"), value=mechanic["price_list"] or "",
                                      placeholder="Dent removal from Rs 1500, full paint Rs 6000 per panel...")
            if st.form_submit_button(tr("SAVE PROFILE"), use_container_width=True):
                conn = db_connect()
                conn.execute("UPDATE mechanics SET services = ?, hours = ?, available = ?, price_list = ? WHERE id = ?",
                             (services.strip(), hours.strip(), 1 if is_open else 0, price_list.strip(), mechanic["id"]))
                conn.commit()
                conn.close()
                st.session_state.flash = ("success", "Profile saved.")
                st.rerun()

    with st.expander(tr("🔒 Change password")):
        with st.form("mechanic_pw_form"):
            old_pw = st.text_input(tr("Current password"), type="password")
            new_pw = st.text_input(tr("New password"), type="password")
            new_pw2 = st.text_input(tr("Confirm new password"), type="password")
            if st.form_submit_button(tr("UPDATE PASSWORD"), use_container_width=True):
                if not verify_password(old_pw, mechanic["password"]):
                    st.error("Current password is incorrect.")
                elif len(new_pw) < 6:
                    st.error("New password must contain at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("New passwords do not match.")
                else:
                    conn = db_connect()
                    conn.execute("UPDATE mechanics SET password = ? WHERE id = ?",
                                 (hash_password_new(new_pw), mechanic["id"]))
                    conn.commit()
                    conn.close()
                    st.success("Password updated.")

    if st.button(tr("LOG OUT"), key="mechanic_logout"):
        st.session_state.mechanic_id = None
        st.rerun()

    st.markdown(tr("### Customer requests matching your area"))
    st.caption("Matched by locality name. Customer phone numbers unlock when you win the job.")
    if not reports:
        st.info("No matching customer reports yet.")

    kinds = {"New": "orange", "Quoted": "blue", "Won": "green", "Lost": "gray"}
    f1, f2 = st.columns(2)
    status_filter = f1.selectbox(tr("Show requests"), ["All", "New", "Quoted", "Won", "Lost"])
    sev_filter = f2.selectbox(tr("Severity"), ["All", "Minor", "Moderate", "Severe"])
    sev_map = {"Minor": "green", "Moderate": "orange", "Severe": "red"}
    conn = db_connect()
    msg_counts = {r[0]: r[1] for r in conn.execute(
        "SELECT report_id, COUNT(*) FROM messages WHERE mechanic_id = ? AND sender = 'customer' GROUP BY report_id",
        (mechanic["id"],)).fetchall()}
    conn.close()
    for rep in reports:
        state = mechanic_report_state(rep["id"], mechanic["id"])
        if status_filter != "All" and state != status_filter:
            continue
        if sev_filter != "All" and sev_kind(rep["severity"]) != sev_map[sev_filter]:
            continue
        with st.expander(f"Report #{rep['id']}  •  {rep['vehicle']}  •  {rep['area']}  •  {state}"
                         + (f"  •  💬 {msg_counts[rep['id']]}" if msg_counts.get(rep["id"]) else "")):
            st.markdown(badge(state, kinds[state]) + " " + badge(rep["severity"], sev_kind(rep["severity"])),
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                if rep["image"]:
                    st.image(load_image(rep["image"]), use_container_width=True)
                for ph in get_photos(rep["id"], "angle"):
                    st.image(ph["image"], caption="Another angle", use_container_width=True)
            with b:
                st.write(f"**Vehicle:** {rep['vehicle']}")
                st.write(f"**Damage:** {str(rep['damage']).replace('_', ' ').title()}")
                st.write(f"**AI estimate:** {money(rep['min_cost'])} – {money(rep['max_cost'])}")
                st.write(f"**Submitted:** {pretty_ts(rep['created'])}")
                if state == "Won":
                    st.success(f"Customer: {rep['customer']}  •  📞 {rep['phone']}")
                else:
                    st.write("**Customer:** hidden until you win this job")

            st.markdown(tr("**💬 Messages with customer**"))
            render_thread(rep["id"], mechanic["id"], "mechanic", f"m_{rep['id']}", block_phone=(state != "Won"))

            if state == "Won":
                cur = rep["repair_status"] if rep["repair_status"] in STATUS_FLOW else "Selected"
                new = st.selectbox("Repair status", STATUS_FLOW, index=STATUS_FLOW.index(cur),
                                   key=f"status_{rep['id']}")
                if st.button(tr("UPDATE STATUS"), key=f"status_btn_{rep['id']}"):
                    conn = db_connect()
                    conn.execute("UPDATE reports SET repair_status = ? WHERE id = ?", (new, rep["id"]))
                    conn.commit()
                    conn.close()
                    st.session_state.flash = ("success", f"Report #{rep['id']} marked {new}.")
                    st.rerun()
                st.markdown(tr("**📷 Repair photos**"))
                pc1, pc2 = st.columns(2)
                for col, kind, label in [(pc1, "before", "Before repair photo"), (pc2, "after", "After repair photo")]:
                    up = col.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"ph_{kind}_{rep['id']}")
                    if up and col.button(f"SAVE {kind.upper()} PHOTO", key=f"ph_save_{kind}_{rep['id']}"):
                        save_photo(rep["id"], mechanic["id"], kind, up.getvalue())
                        st.session_state.flash = ("success", f"{kind.title()} photo saved for report #{rep['id']}.")
                        st.rerun()
                for ph in get_photos(rep["id"]):
                    if ph["kind"] in ("before", "after"):
                        st.image(ph["image"], caption=ph["kind"].title(), width=220)
                continue
            if state == "Lost":
                st.info("The customer selected another workshop.")
                continue

            conn = db_connect()
            mine = conn.execute("SELECT * FROM quotes WHERE report_id = ? AND mechanic_id = ?",
                                (rep["id"], mechanic["id"])).fetchone()
            conn.close()

            def pv(col, default):
                return mine[col] if mine and mine[col] is not None else default

            st.caption(f"AI estimate for this report: {money(rep['min_cost'])} – {money(rep['max_cost'])}. "
                       "Your total = parts + labour + paint.")
            with st.form(f"mechanic_quote_{rep['id']}"):
                q1, q2, q3 = st.columns(3)
                parts = q1.number_input(tr("Parts (₹)"), 0, step=100, value=int(pv("parts", 0)))
                labour = q2.number_input(tr("Labour (₹)"), 0, step=100, value=int(pv("labour", 0)))
                paint = q3.number_input(tr("Paint (₹)"), 0, step=100, value=int(pv("paint", 0)))
                d1, d2 = st.columns(2)
                days = d1.number_input(tr("Days needed"), 1, 60, value=int(pv("days", 2)))
                wopts = ["No warranty", "30 days", "3 months", "6 months", "1 year"]
                warranty = d2.selectbox(tr("Warranty"), wopts, index=wopts.index(pv("warranty", "30 days"))
                                        if pv("warranty", "30 days") in wopts else 1)
                details = st.text_area(tr("Repair details"), value=mine["details"] if mine and mine["details"] else "",
                                       placeholder="Mention the repair work, brands and what is included...")
                submit_quote = st.form_submit_button(tr("SUBMIT / UPDATE QUOTATION"))

            if submit_quote:
                total = parts + labour + paint
                if total <= 0:
                    st.error("Enter at least one cost above ₹0.")
                else:
                    conn = db_connect()
                    conn.execute("""INSERT INTO quotes (report_id, mechanic_id, amount, details, selected,
                                    parts, labour, paint, days, warranty)
                                    VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                                    ON CONFLICT(report_id, mechanic_id) DO UPDATE SET
                                    amount = excluded.amount, details = excluded.details,
                                    parts = excluded.parts, labour = excluded.labour, paint = excluded.paint,
                                    days = excluded.days, warranty = excluded.warranty""",
                                 (rep["id"], mechanic["id"], total, details, parts, labour, paint, days, warranty))
                    conn.commit()
                    conn.close()
                    if rep["email"]:
                        send_email(rep["email"], f"New quotation for your AutoVision report #{rep['id']}",
                                   f"{mechanic['name']} ({mechanic['area']}) sent a quotation of {money(total)}.\n"
                                   "Open AutoVision AI, go to Quotations and enter your report ID and phone number to compare.")
                    hi, lo = rep["max_cost"] or 0, rep["min_cost"] or 0
                    if hi and total > hi * 1.4:
                        st.session_state.flash = ("warning", f"Saved ({money(total)}). This is well above the AI estimate, so customers may skip it.")
                    elif lo and total < lo * 0.6:
                        st.session_state.flash = ("warning", f"Saved ({money(total)}). This is far below the AI estimate. Double-check your costs.")
                    else:
                        st.session_state.flash = ("success", f"Quotation of {money(total)} saved.")
                    st.rerun()

            if mine and st.button(tr("WITHDRAW MY QUOTE"), key=f"withdraw_{rep['id']}"):
                conn = db_connect()
                conn.execute("DELETE FROM quotes WHERE report_id = ? AND mechanic_id = ?", (rep["id"], mechanic["id"]))
                conn.commit()
                conn.close()
                st.session_state.flash = ("info", f"Your quote for report #{rep['id']} was withdrawn.")
                st.rerun()


# ------------------------- QUOTATIONS PAGE -------------------------

def ai_band_html(amount, lo, hi):
    if not hi:
        return ""
    s_lo, s_hi = lo * 0.5, hi * 1.5
    span = max(s_hi - s_lo, 1)
    clamp = lambda v: max(0, min(100, v))
    left, width = clamp((lo - s_lo) / span * 100), clamp((hi - lo) / span * 100)
    pos = clamp((amount - s_lo) / span * 100)
    return (f"<div class='small-note'>Compared with AI estimate ({money(lo)} – {money(hi)})</div>"
            f"<div class='band'><div class='band-ai' style='left:{left}%;width:{width}%'></div>"
            f"<div class='band-pin' style='left:{pos}%'></div></div>")


def render_quotations():
    st.markdown('<div class="section-heading">REPAIR PRICE COMPARISON</div>', unsafe_allow_html=True)
    st.title(tr("Compare Mechanic Quotations"))
    st.write("Use your report ID and customer phone number to view submitted quotations.")
    st.info("🧾 **For vehicle owners.** Enter the Report ID you received on the Customer page and the same phone number "
            "you used there.")
    _qid = st.session_state.get("quotation_report_id")
    _active = 0
    if _qid:
        _c = db_connect()
        _won = _c.execute("SELECT COUNT(*) FROM quotes WHERE report_id = ? AND selected = 1", (_qid,)).fetchone()[0]
        _st = _c.execute("SELECT repair_status FROM reports WHERE id = ?", (_qid,)).fetchone()
        _c.close()
        _active = 4 if (_st and _st[0] == "Ready") else 3 if _won else 1
    step_bar(["Enter Report ID", "Compare quotes", "Select mechanic", "Track repair"], _active)
    _last = st.session_state.get("last_report") or {}
    if "flash_q" in st.session_state:
        st.success(st.session_state.pop("flash_q"))

    with st.form("quotation_lookup_form"):
        c1, c2 = st.columns(2)
        report_id_input = c1.number_input(tr("Damage Report ID"), min_value=1, step=1, value=int(_last.get("id", 1)),
                                          help="The number shown after you shared your report (also in your PDF).")
        phone_input = c2.text_input(tr("Customer phone number"), value=_last.get("phone", ""), placeholder="9876543210",
                                    help="The same phone number you entered when sharing the report.")
        lookup = st.form_submit_button(tr("FIND QUOTATIONS"), use_container_width=True)

    if lookup:
        report = get_report(int(report_id_input))
        if report is None:
            st.error(tr("Report ID not found."))
            st.session_state.pop("quotation_report_id", None)
        elif report["phone"] != phone_input.strip():
            st.error(tr("Phone number does not match this report."))
            st.session_state.pop("quotation_report_id", None)
        else:
            st.session_state.quotation_report_id = int(report_id_input)

    report_id = st.session_state.get("quotation_report_id")
    report = get_report(report_id) if report_id else None
    if not report:
        return

    st.markdown(
        f"<div class='panel'><div class='section-heading'>YOUR INSPECTION REPORT</div>"
        f"<h3>Report #{report['id']} — {report['vehicle']}</h3>"
        f"<p>📍 {report['area']} &nbsp;•&nbsp; Damage: {str(report['damage']).replace('_', ' ').title()} "
        f"&nbsp;•&nbsp; {badge(report['severity'], sev_kind(report['severity']))} "
        f"&nbsp;•&nbsp; AI estimate {money(report['min_cost'])} – {money(report['max_cost'])}</p></div>",
        unsafe_allow_html=True)

    journey = build_journey_pdf(report_id)
    if journey:
        st.download_button(tr("⬇ DOWNLOAD COMPLETE REPAIR JOURNEY (PDF)"), journey,
                           f"autovision_journey_{report_id}.pdf", "application/pdf",
                           use_container_width=True, key="journey_pdf")
        st.caption("Includes your report, AI result, all quotations, the selected mechanic and the repair progress.")
    else:
        st.caption("Install reportlab (pip install reportlab) to enable the journey PDF.")

    with st.expander(tr("🗑 Delete this report")):
        st.warning(tr("This permanently deletes the report, all quotations and messages."))
        sure = st.checkbox(tr("I understand, delete this report"), key="del_confirm")
        if st.button(tr("DELETE REPORT"), key="del_report"):
            if not sure:
                st.error(tr("Tick the box to confirm."))
            else:
                conn = db_connect()
                for tbl in ("messages", "reviews", "quotes"):
                    conn.execute(f"DELETE FROM {tbl} WHERE report_id = ?", (report_id,))
                conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
                conn.commit()
                conn.close()
                st.session_state.pop("quotation_report_id", None)
                st.session_state.flash_q = "Report deleted."
                st.rerun()

    conn = db_connect()
    quotes = conn.execute("""
        SELECT q.*, m.name, m.phone AS mphone, m.area AS marea, m.services, m.hours, m.available, m.price_list,
               (SELECT AVG(rating) FROM reviews r WHERE r.mechanic_id = m.id) AS rating,
               (SELECT COUNT(*) FROM reviews r WHERE r.mechanic_id = m.id) AS rcount
        FROM quotes q JOIN mechanics m ON m.id = q.mechanic_id WHERE q.report_id = ?""",
        (report_id,)).fetchall()
    conn.close()

    if not quotes:
        st.info(tr("No quotations received yet. Please check again later."))
        return

    # In-app notifications: new quotations / mechanic messages since the customer last looked
    notif_key = f"notif_{report_id}"
    unseen_q = [q for q in quotes if not q["seen"]]
    conn = db_connect()
    unseen_m = conn.execute("""SELECT COUNT(*) FROM messages WHERE report_id = ? AND sender = 'mechanic'
                               AND COALESCE(seen, 0) = 0""", (report_id,)).fetchone()[0]
    if unseen_q or unseen_m:
        bits = []
        if unseen_q:
            bits.append(f"{len(unseen_q)} new quotation(s) from " + ", ".join(q["name"] for q in unseen_q))
        if unseen_m:
            bits.append(f"{unseen_m} new message(s) from mechanics")
        st.session_state[notif_key] = "🔔 " + "  •  ".join(bits)
        conn.execute("UPDATE quotes SET seen = 1 WHERE report_id = ?", (report_id,))
        conn.execute("UPDATE messages SET seen = 1 WHERE report_id = ? AND sender = 'mechanic'", (report_id,))
        conn.commit()
    conn.close()
    if st.session_state.get(notif_key):
        st.info(st.session_state[notif_key])

    chosen = next((q for q in quotes if q["selected"]), None)
    if chosen:
        status = report["repair_status"] if report["repair_status"] in STATUS_FLOW else "Selected"
        idx = STATUS_FLOW.index(status)
        html = "".join(f"<div class='stp {'done' if i < idx else 'now' if i == idx else ''}'>"
                       f"<span>{'✓' if i < idx else i + 1}</span>{n}</div>" for i, n in enumerate(STATUS_FLOW))
        st.markdown(f"### Repair tracker — {chosen['name']}")
        st.markdown(f"<div class='stepbar'>{html}</div>", unsafe_allow_html=True)
        lo_e, hi_e = report["min_cost"] or 0, report["max_cost"] or 0
        if hi_e:
            mid_e = (lo_e + hi_e) / 2
            diff = (chosen["amount"] - mid_e) / mid_e * 100 if mid_e else 0
            st.caption(f"Accepted quote {money(chosen['amount'])} is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} "
                       f"the AI midpoint estimate ({money(mid_e)}) and "
                       f"{'inside' if lo_e <= chosen['amount'] <= hi_e else 'outside'} the AI range.")
        photos = get_photos(report_id)
        before_p = [p for p in photos if p["kind"] == "before"]
        after_p = [p for p in photos if p["kind"] == "after"]
        if before_p or after_p:
            st.markdown(tr("**📷 Repair photos**"))
            pc1, pc2 = st.columns(2)
            with pc1:
                for p in before_p:
                    st.image(p["image"], caption="Before", use_container_width=True)
            with pc2:
                for p in after_p:
                    st.image(p["image"], caption="After", use_container_width=True)

        if status == "Ready":
            conn = db_connect()
            done = conn.execute("SELECT 1 FROM reviews WHERE report_id = ?", (report_id,)).fetchone()
            conn.close()
            if done:
                st.success("Thanks for rating this workshop.")
            else:
                with st.form("review_form"):
                    st.markdown(tr("**Rate this repair**"))
                    stars = st.select_slider("Rating", [1, 2, 3, 4, 5], value=5)
                    comment = st.text_input(tr("Comment (optional)"))
                    if st.form_submit_button(tr("SUBMIT REVIEW"), use_container_width=True):
                        conn = db_connect()
                        conn.execute("INSERT INTO reviews (report_id, mechanic_id, rating, comment) VALUES (?, ?, ?, ?)",
                                     (report_id, chosen["mechanic_id"], stars, comment.strip()))
                        conn.commit()
                        conn.close()
                        st.rerun()

    quote_ctx = {"damage": str(report["damage"]).replace("_", " "), "severity": report["severity"],
                 "min_cost": report["min_cost"], "max_cost": report["max_cost"],
                 "quotes": [{"name": q["name"], "amount": q["amount"], "days": q["days"],
                             "rating": q["rating"], "warranty": q["warranty"]} for q in quotes]}
    with st.expander(tr("💬 Ask the AutoVision assistant")):
        render_assistant("quotes", quote_ctx)
    with st.expander(tr("🗺 Where are these workshops?")):
        render_workshop_map(mech_ids={q["mechanic_id"] for q in quotes}, customer_area=report["area"])

    sort_by = st.selectbox(tr("Sort quotes by"), ["Lowest price", "Fastest", "Best rated", "Nearest"])
    key = {"Lowest price": lambda q: q["amount"],
           "Fastest": lambda q: q["days"] if q["days"] else 999,
           "Best rated": lambda q: -(q["rating"] or 0),
           "Nearest": lambda q: area_distance(report["area"], q["marea"]) or 999}[sort_by]
    quotes = sorted(quotes, key=key)

    if len(quotes) > 1:
        lo_p, hi_p = int(min(q["amount"] for q in quotes)), int(max(q["amount"] for q in quotes))
        if hi_p > lo_p:
            budget = st.slider(tr("Maximum budget (₹)"), lo_p, hi_p, hi_p, step=100)
            quotes = [q for q in quotes if q["amount"] <= budget]

    cheapest = min(q["amount"] for q in quotes)
    days_vals = [q["days"] for q in quotes if q["days"]]
    fastest = min(days_vals) if days_vals else None
    rated = [q["rating"] for q in quotes if q["rating"]]
    best = max(rated) if rated else None

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0

    amt = [q["amount"] for q in quotes]
    dys = [q["days"] or 0 for q in quotes]

    def value_score(q):   # lower is better: price 50%, speed 20%, rating 30%
        r = 0.5 if not q["rating"] else 1 - q["rating"] / 5
        return (0.5 * norm(q["amount"], min(amt), max(amt))
                + 0.2 * norm(q["days"] or 0, min(dys), max(dys)) + 0.3 * r)

    rec_id = min(quotes, key=value_score)["id"] if len(quotes) > 1 else None

    with st.expander(tr("📊 Compare side by side")):
        table = [{"Mechanic": q["name"], "Area": q["marea"], "Total (₹)": int(q["amount"]),
                  "Parts": int(q["parts"] or 0), "Labour": int(q["labour"] or 0), "Paint": int(q["paint"] or 0),
                  "Days": q["days"] or "–", "Warranty": q["warranty"] or "–",
                  "Rating": round(q["rating"], 1) if q["rating"] else "–"} for q in quotes]
        st.dataframe(table, use_container_width=True, hide_index=True)
        import pandas as pd
        cost_df = pd.DataFrame(table).set_index("Mechanic")[["Parts", "Labour", "Paint"]]
        if cost_df.sum().sum() > 0:
            st.caption(tr("Cost breakdown (parts, labour, paint)"))
            st.bar_chart(cost_df)
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
        st.download_button(tr("⬇ DOWNLOAD COMPARISON (CSV)"), buf.getvalue(), f"quotes_report_{report_id}.csv", "text/csv")

    for q in quotes:
        tags = badge("Recommended", "green") if q["id"] == rec_id else ""
        if q["amount"] == cheapest and len(quotes) > 1:
            tags += badge("Lowest price", "green")
        if fastest and q["days"] == fastest and len(days_vals) > 1:
            tags += badge("Fastest", "blue")
        if best and q["rating"] == best and len(rated) > 1:
            tags += badge("Best rated", "orange")
        if (q["services"] or "").strip() and (q["hours"] or "").strip():
            tags += badge("Verified", "gray")
        if q["available"] == 0:
            tags += badge("Not taking new jobs", "red")

        with st.container(border=True):
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"### 🔧 {q['name']}  {tags}", unsafe_allow_html=True)
                stars = f"⭐ {q['rating']:.1f} ({q['rcount']})" if q["rcount"] else "No ratings yet"
                dist = area_distance(report["area"], q["marea"])
                st.write(f"📍 {q['marea']}  •  {stars}" + (f"  •  📏 ~{dist} km" if dist is not None else ""))
                meta = []
                if q["days"]:
                    meta.append(f"🕒 {q['days']} day(s)")
                if q["warranty"]:
                    meta.append(f"🛡 {q['warranty']}")
                if meta:
                    st.write("  •  ".join(meta))
                if q["parts"] or q["labour"] or q["paint"]:
                    st.caption(f"Parts {money(q['parts'])} + Labour {money(q['labour'])} + Paint {money(q['paint'])}")
                st.write(q["details"] or "No repair details provided.")
                if q["price_list"]:
                    st.caption("💰 Price list: " + q["price_list"])
                with st.expander(tr("💬 Message this workshop")):
                    render_thread(report_id, q["mechanic_id"], "customer",
                                  f"c_{report_id}_{q['mechanic_id']}", block_phone=not q["selected"])
                st.markdown(ai_band_html(q["amount"], report["min_cost"] or 0, report["max_cost"] or 0),
                            unsafe_allow_html=True)
            with right:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>MECHANIC QUOTATION</div>"
                            f"<div class='metric-value'>{money(q['amount'])}</div></div>", unsafe_allow_html=True)
                if q["selected"]:
                    st.success("Selected")
                    st.markdown(f"<a class='link-btn' href='tel:{q['mphone']}'>📞 Call</a>"
                                f"<a class='link-btn' target='_blank' href='https://wa.me/91{q['mphone']}'>💬 WhatsApp</a>",
                                unsafe_allow_html=True)
                    if (report["repair_status"] or "Open") in ("Open", "Selected") and \
                            st.button(tr("CHANGE MECHANIC"), key=f"unselect_{q['id']}"):
                        conn = db_connect()
                        conn.execute("UPDATE quotes SET selected = 0 WHERE report_id = ?", (report_id,))
                        conn.execute("UPDATE reports SET repair_status = 'Open' WHERE id = ?", (report_id,))
                        conn.commit()
                        conn.close()
                        st.rerun()
                elif not chosen and st.button(tr("SELECT THIS MECHANIC"), key=f"choose_quote_{q['id']}"):
                    conn = db_connect()
                    conn.execute("UPDATE quotes SET selected = 0 WHERE report_id = ?", (report_id,))
                    conn.execute("UPDATE quotes SET selected = 1 WHERE id = ?", (q["id"],))
                    conn.execute("UPDATE reports SET repair_status = 'Selected' WHERE id = ?", (report_id,))
                    conn.commit()
                    conn.close()
                    st.rerun()


# ------------------------- ABOUT PAGE -------------------------

def render_about():
    st.markdown('<div class="section-heading">ABOUT THE PROJECT</div>',
                unsafe_allow_html=True)
    st.title(tr("AI Vehicle Damage Inspection System"))
    st.write(
        "AutoVision AI is a vehicle damage assessment and repair-network "
        "prototype. It combines a trained YOLO model with severity mapping, "
        "indicative repair-cost estimation and a mechanic quotation workflow."
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown("<div class='portal-card'><div class='portal-icon'>🧠</div>"
                    "<h3>AI Detection</h3><p>Identifies damage categories supported by the trained model.</p></div>",
                    unsafe_allow_html=True)
    with b:
        st.markdown("<div class='portal-card'><div class='portal-icon'>📊</div>"
                    "<h3>Severity & Cost</h3><p>Provides indicative severity and repair-cost ranges.</p></div>",
                    unsafe_allow_html=True)
    with c:
        st.markdown("<div class='portal-card'><div class='portal-icon'>🤝</div>"
                    "<h3>Mechanic Network</h3><p>Allows workshops to quote on customer reports.</p></div>",
                    unsafe_allow_html=True)

    st.warning(
        "The current model detects only the damage classes it was trained on. "
        "It does not independently verify that every uploaded image is a vehicle. "
        "AI estimates are not a substitute for a physical inspection."
    )

    with st.expander(tr("🔐 Admin dashboard")):
        render_admin_dashboard()


# ------------------------- PAGE ROUTER -------------------------

page = st.session_state.get("page", "Home")

if page == "Home":
    render_home()

elif page == "Customer":
    render_customer()

elif page == "Mechanic":
    render_mechanic()

elif page == "Quotations":
    render_quotations()

elif page == "About":
    render_about()
# ------------------------- FOOTER -------------------------

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#777276;padding:14px;font-size:11px;letter-spacing:1px;">
  <b style="color:#bd263d;">AUTOVISION AI</b><br>
  DETECT&nbsp; • &nbsp;ANALYSE&nbsp; • &nbsp;REPAIR SMARTER
</div>
""", unsafe_allow_html=True)





import streamlit as st
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO

from severity import assess_severity
from cost_estimation import estimate_repair_cost


# PAGE SETTINGS
st.set_page_config(
    page_title="AI Vehicle Damage Assessment",
    page_icon="🚗",
    layout="wide"
)

# DATABASE
def db():
    return sqlite3.connect("vehicle_damage.db", check_same_thread=False)


with db() as con:
    con.execute("""
        CREATE TABLE IF NOT EXISTS mechanics(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT UNIQUE,
            area TEXT,
            password TEXT
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS reports(
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

    con.execute("""
        CREATE TABLE IF NOT EXISTS quotes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            mechanic_id INTEGER,
            amount INTEGER,
            details TEXT,
            selected INTEGER DEFAULT 0,
            UNIQUE(report_id, mechanic_id)
        )
    """)


# AI MODEL
MODEL_PATH = Path(
    "best.pt"
)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return None


# DESIGN
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#10152b,#172554,#241342);
    color: white;
}
.hero {
    background: linear-gradient(100deg,#0066ff,#793bdb,#ed3b91);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
}
.hero h1 {
    color: white;
}
.section {
    color: #8ec5ff;
    font-size: 25px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
<h1>🚗 AI VEHICLE DAMAGE ASSESSMENT</h1>
<p>Damage Detection | Severity Analysis | Cost Estimation | Mechanic Network</p>
</div>
""", unsafe_allow_html=True)


# LOGIN SESSION
if "mechanic" not in st.session_state:
    st.session_state.mechanic = None


customer_tab, mechanic_tab, quotation_tab = st.tabs([
    "🚗 Customer",
    "🔧 Mechanic Portal",
    "📋 Quotations"
])


# CUSTOMER SECTION
with customer_tab:

    st.markdown(
        '<div class="section">📸 Upload Vehicle Image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload damaged vehicle image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image_bytes = uploaded.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        st.image(image, use_container_width=True)

        if st.button("Analyse Damage"):
            model = load_model()

            if model is None:
                st.error(f"Model not found: {MODEL_PATH}")
            else:
                with st.spinner("Analysing image..."):
                    results = model.predict(
                        np.array(image),
                        conf=0.25,
                        verbose=False
                    )

                    result = results[0]
                    detections = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            damage = model.names[class_id]

                            detections.append(
                                (damage, confidence)
                            )

                    st.session_state.detections = detections
                    st.session_state.image_bytes = image_bytes
                    st.session_state.result_image = result.plot()

        if "detections" in st.session_state:
            st.markdown("### 🧠 AI Detection Result")

            st.image(
                st.session_state.result_image,
                channels="BGR",
                use_container_width=True
            )

            detections = st.session_state.detections

            if not detections:
                st.warning("No damage detected. Try another image.")

            else:
                for damage, confidence in detections:
                    severity, description = assess_severity(damage)
                    low, high = estimate_repair_cost(damage)

                    st.write("**Damage:**", damage.replace("_", " ").title())
                    st.write(f"**Confidence:** {confidence:.0%}")
                    st.write("**Severity:**", severity)
                    st.write(description)

                    if low:
                        st.write(f"**Estimated Cost:** ₹{low:,} – ₹{high:,}")
                    else:
                        st.write("Manual cost assessment required.")

                    st.divider()

                st.caption(
                    "AI results are preliminary. Final repair costs "
                    "must be confirmed by a mechanic."
                )

                st.markdown("### 📤 Send Report to Mechanics")

                with st.form("customer_report"):
                    customer = st.text_input("Customer Name")
                    phone = st.text_input("Phone Number")
                    vehicle = st.text_input("Vehicle Model")
                    area = st.text_input("Your Bengaluru Area")

                    send = st.form_submit_button("Submit Damage Report")

                    if send:
                        if not all([
                            customer.strip(),
                            phone.strip(),
                            vehicle.strip(),
                            area.strip()
                        ]):
                            st.error("Please fill all details.")
                        else:
                            damage = detections[0][0]
                            severity, _ = assess_severity(damage)
                            low, high = estimate_repair_cost(damage)

                            with db() as con:
                                cur = con.execute("""
                                    INSERT INTO reports
                                    (customer, phone, area, vehicle,
                                     damage, severity, min_cost, max_cost,
                                     image, created)
                                    VALUES(?,?,?,?,?,?,?,?,?,?)
                                """, (
                                    customer, phone, area, vehicle,
                                    damage, severity, low, high,
                                    st.session_state.image_bytes,
                                    datetime.now().strftime("%Y-%m-%d %H:%M")
                                ))

                                report_id = cur.lastrowid

                            st.success(
                                f"Report submitted! Your Report ID is {report_id}"
                            )


# MECHANIC PORTAL
with mechanic_tab:

    st.markdown(
        '<div class="section">🔧 Mechanic Portal</div>',
        unsafe_allow_html=True
    )

    if st.session_state.mechanic is None:

        register, login = st.tabs(["Register", "Login"])

        with register:
            with st.form("register_mechanic"):
                name = st.text_input("Mechanic Name")
                phone = st.text_input("Phone Number")
                area = st.text_input("Workshop Area")
                password = st.text_input("Password", type="password")

                submit = st.form_submit_button("Register")

                if submit:
                    if not all([name, phone, area, password]):
                        st.error("Fill all details.")
                    else:
                        try:
                            with db() as con:
                                con.execute("""
                                    INSERT INTO mechanics
                                    (name, phone, area, password)
                                    VALUES(?,?,?,?)
                                """, (
                                    name.strip(),
                                    phone.strip(),
                                    area.strip(),
                                    hashlib.sha256(
                                        password.encode()
                                    ).hexdigest()
                                ))

                            st.success("Registration successful!")

                        except sqlite3.IntegrityError:
                            st.error("Phone number already registered.")

        with login:
            with st.form("mechanic_login"):
                phone = st.text_input("Registered Phone")
                password = st.text_input("Password", type="password")

                submit = st.form_submit_button("Login")

                if submit:
                    hashed = hashlib.sha256(
                        password.encode()
                    ).hexdigest()

                    with db() as con:
                        user = con.execute("""
                            SELECT id, name, area FROM mechanics
                            WHERE phone=? AND password=?
                        """, (phone.strip(), hashed)).fetchone()

                    if user:
                        st.session_state.mechanic = user
                        st.rerun()
                    else:
                        st.error("Invalid login details.")

    else:
        mechanic_id, mechanic_name, mechanic_area = (
            st.session_state.mechanic
        )

        st.success(f"Welcome, {mechanic_name} — {mechanic_area}")

        if st.button("Logout"):
            st.session_state.mechanic = None
            st.rerun()

        with db() as con:
            reports = con.execute("""
                SELECT id, customer, phone, area, vehicle, damage,
                       severity, min_cost, max_cost, image
                FROM reports
                WHERE LOWER(area) LIKE ?
                ORDER BY id DESC
            """, (f"%{mechanic_area.lower()}%",)).fetchall()

        st.markdown("### Customer Reports")

        if not reports:
            st.info("No reports available in your matching area.")

        for r in reports:
            with st.expander(f"Report #{r[0]} — {r[4]} — {r[3]}"):

                st.write("Customer:", r[1])
                st.write("Phone:", r[2])
                st.write("Vehicle:", r[4])
                st.write("Damage:", r[5])
                st.write("Severity:", r[6])

                if r[8]:
                    st.write(f"AI Cost Estimate: ₹{r[7]:,} – ₹{r[8]:,}")

                st.image(
                    Image.open(BytesIO(r[9])),
                    use_container_width=True
                )

                with st.form(f"quote_{r[0]}"):
                    amount = st.number_input(
                        "Your Repair Quotation (₹)",
                        min_value=1,
                        step=500
                    )

                    details = st.text_area("Repair Details")

                    submit = st.form_submit_button("Send Quotation")

                    if submit:
                        with db() as con:
                            con.execute("""
                                INSERT INTO quotes
                                (report_id, mechanic_id, amount, details)
                                VALUES(?,?,?,?)
                                ON CONFLICT(report_id, mechanic_id)
                                DO UPDATE SET
                                amount=excluded.amount,
                                details=excluded.details
                            """, (
                                r[0], mechanic_id, int(amount), details
                            ))

                        st.success("Quotation submitted!")


# CUSTOMER QUOTATION COMPARISON
# CUSTOMER QUOTATION COMPARISON
with quotation_tab:

    st.markdown(
        '<div class="section">💰 Compare Mechanic Quotations</div>',
        unsafe_allow_html=True
    )

    report_id = st.number_input(
        "Enter Your Report ID",
        min_value=1,
        step=1,
        key="quotation_report_id"
    )

    if st.button("View Quotations"):
        st.session_state["view_report_id"] = int(report_id)

    if st.session_state.get("view_report_id"):

        selected_report_id = st.session_state["view_report_id"]

        with db() as con:
            report = con.execute("""
                SELECT customer, vehicle, area
                FROM reports WHERE id=?
            """, (selected_report_id,)).fetchone()

            quotes = con.execute("""
                SELECT q.id, m.name, m.area, m.phone,
                       q.amount, q.details, q.selected
                FROM quotes q
                JOIN mechanics m ON q.mechanic_id=m.id
                WHERE q.report_id=?
                ORDER BY q.amount
            """, (selected_report_id,)).fetchall()

        if not report:
            st.error("Report ID not found.")

        elif not quotes:
            st.info("No quotations received yet.")

        else:
            st.success(
                f"Vehicle: {report[1]} | Area: {report[2]}"
            )

            for q in quotes:
                quote_id, name, area, phone, amount, details, selected = q

                with st.container(border=True):
                    st.subheader(name)
                    st.write("Workshop:", area)
                    st.write("Phone:", phone)
                    st.write(f"Quotation: ₹{amount:,}")
                    st.write("Details:", details or "Not provided")

                    if selected:
                        st.success("✅ Selected Mechanic")
                    else:
                        if st.button(
                            "Select Mechanic",
                            key=f"select_{quote_id}"
                        ):
                            with db() as con:
                                con.execute("""
                                    UPDATE quotes
                                    SET selected=0
                                    WHERE report_id=?
                                """, (selected_report_id,))

                                con.execute("""
                                    UPDATE quotes
                                    SET selected=1
                                    WHERE id=? AND report_id=?
                                """, (quote_id, selected_report_id))

                            st.session_state["mechanic_selected"] = True
                            st.rerun()

            if st.session_state.get("mechanic_selected"):
                st.success("Mechanic selection saved successfully!")
                st.session_state["mechanic_selected"] = False

st.markdown("""
<center>
🚘 AI-Based Intelligent Vehicle Damage Assessment System
<br>
Smart Technology for Smarter Vehicle Repair
</center>
""", unsafe_allow_html=True)

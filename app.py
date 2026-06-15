import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
from io import BytesIO
import hashlib

# ------------------------------------------------------
# PAGINA-INSTELLINGEN
# ------------------------------------------------------
st.set_page_config(
    page_title="AI Recycle Coach",
    page_icon="♻️",
    layout="wide"
)

# ------------------------------------------------------
# AI-MODEL LADEN
# ------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ------------------------------------------------------
# AFVALDATABASE
# ------------------------------------------------------
afval_info = {
    "banana": {
        "naam": "Banaan",
        "bak": "GFT",
        "icoon": "🍌",
        "bak_icoon": "🟢",
        "impact_icoon": "🌱",
        "tip_icoon": "💡",
        "impact": "Bananenschillen kunnen verwerkt worden tot compost of biogas.",
        "tip": "Gooi fruitresten bij het GFT-afval.",
        "ai_uitleg": "De AI herkent vooral de vorm en kleur van de banaan.",
        "score": 10
    },
    "apple": {
        "naam": "Appel",
        "bak": "GFT",
        "icoon": "🍎",
        "bak_icoon": "🟢",
        "impact_icoon": "🌱",
        "tip_icoon": "💡",
        "impact": "Appelresten zijn biologisch afbreekbaar en geschikt voor compost.",
        "tip": "Gooi klokhuizen en fruitschillen bij het GFT.",
        "ai_uitleg": "De AI herkent de ronde vorm en kleuren die vaak bij een appel horen.",
        "score": 10
    },
    "orange": {
        "naam": "Sinaasappel",
        "bak": "GFT",
        "icoon": "🍊",
        "bak_icoon": "🟢",
        "impact_icoon": "🌱",
        "tip_icoon": "💡",
        "impact": "Sinaasappelschillen kunnen opnieuw gebruikt worden als compost.",
        "tip": "Gooi fruitschillen bij het GFT-afval.",
        "ai_uitleg": "De AI kijkt naar de kleur, ronde vorm en herkenbare kenmerken van fruit.",
        "score": 10
    },
    "bottle": {
        "naam": "Fles",
        "bak": "PMD / Plastic",
        "icoon": "🧴",
        "bak_icoon": "🔵",
        "impact_icoon": "♻️",
        "tip_icoon": "💧",
        "impact": "Plastic kan opnieuw gebruikt worden als het goed wordt gescheiden.",
        "tip": "Maak de fles leeg voordat je hem weggooit.",
        "ai_uitleg": "De AI herkent de fles aan de langwerpige vorm, dop en doorzichtige structuur.",
        "score": 15
    },
    "cup": {
        "naam": "Beker",
        "bak": "Restafval of papier",
        "icoon": "🥤",
        "bak_icoon": "🟠",
        "impact_icoon": "⚠️",
        "tip_icoon": "💡",
        "impact": "Bekers met een plastic laagje zijn vaak moeilijk te recyclen.",
        "tip": "Gebruik liever een herbruikbare beker.",
        "ai_uitleg": "De AI herkent de beker aan de vorm en opening aan de bovenkant.",
        "score": 8
    },
    "book": {
        "naam": "Boek",
        "bak": "Papier / Karton",
        "icoon": "📘",
        "bak_icoon": "🔵",
        "impact_icoon": "📄",
        "tip_icoon": "💡",
        "impact": "Papier kan goed gerecycled worden als het schoon en droog blijft.",
        "tip": "Lever boeken die nog goed zijn liever in bij een kringloop of bibliotheek.",
        "ai_uitleg": "De AI herkent het boek aan de rechthoekige vorm en platte structuur.",
        "score": 12
    },
    "cell phone": {
        "naam": "Mobiele telefoon",
        "bak": "Elektronisch afval",
        "icoon": "📱",
        "bak_icoon": "🟣",
        "impact_icoon": "🔋",
        "tip_icoon": "📍",
        "impact": "Elektronica bevat waardevolle metalen die opnieuw gebruikt kunnen worden.",
        "tip": "Lever oude telefoons in bij een inzamelpunt voor e-waste.",
        "ai_uitleg": "De AI herkent de telefoon aan de smalle rechthoekige vorm en het scherm.",
        "score": 20
    },
    "laptop": {
        "naam": "Laptop",
        "bak": "Elektronisch afval",
        "icoon": "💻",
        "bak_icoon": "🟣",
        "impact_icoon": "🔋",
        "tip_icoon": "📍",
        "impact": "Laptops bevatten grondstoffen die hergebruikt kunnen worden.",
        "tip": "Breng oude laptops naar een milieustraat of elektronica-inzamelpunt.",
        "ai_uitleg": "De AI herkent de laptop aan het scherm, toetsenbord en de hoekige vorm.",
        "score": 20
    },
    "keyboard": {
        "naam": "Toetsenbord",
        "bak": "Elektronisch afval",
        "icoon": "⌨️",
        "bak_icoon": "🟣",
        "impact_icoon": "🔋",
        "tip_icoon": "📍",
        "impact": "Elektronische apparaten horen niet bij het restafval.",
        "tip": "Lever oude randapparatuur in bij een e-waste punt.",
        "ai_uitleg": "De AI herkent het toetsenbord aan de vorm en de herhaling van toetsen.",
        "score": 15
    },
    "mouse": {
        "naam": "Computermuis",
        "bak": "Elektronisch afval",
        "icoon": "🖱️",
        "bak_icoon": "🟣",
        "impact_icoon": "🔋",
        "tip_icoon": "📍",
        "impact": "Kleine elektronica bevat materialen die opnieuw gebruikt kunnen worden.",
        "tip": "Gooi kleine elektronica niet zomaar in de prullenbak.",
        "ai_uitleg": "De AI herkent de computermuis aan de ronde vorm en het compacte ontwerp.",
        "score": 15
    },
    "scissors": {
        "naam": "Schaar",
        "bak": "Metaal / Restafval",
        "icoon": "✂️",
        "bak_icoon": "⚫",
        "impact_icoon": "🔩",
        "tip_icoon": "💡",
        "impact": "Metaal kan vaak opnieuw worden verwerkt.",
        "tip": "Controleer lokaal of kleine metalen voorwerpen apart ingeleverd kunnen worden.",
        "ai_uitleg": "De AI herkent de schaar aan de twee grepen en metalen bladen.",
        "score": 8
    },
    "vase": {
        "naam": "Glazen object",
        "bak": "Glasbak of restafval",
        "icoon": "🏺",
        "bak_icoon": "⚫",
        "impact_icoon": "🍾",
        "tip_icoon": "💡",
        "impact": "Niet al het glas hoort in de glasbak. Drinkglazen en vazen horen vaak bij restafval.",
        "tip": "Controleer of het om verpakkingsglas gaat. Alleen dat hoort meestal in de glasbak.",
        "ai_uitleg": "De AI herkent het object aan de vorm en het glasachtige uiterlijk.",
        "score": 8
    },
    "bowl": {
        "naam": "Kom",
        "bak": "Restafval",
        "icoon": "🥣",
        "bak_icoon": "⚫",
        "impact_icoon": "⚠️",
        "tip_icoon": "💡",
        "impact": "Servies zoals kommen en borden hoort meestal niet in de glasbak.",
        "tip": "Is het nog bruikbaar? Breng het dan naar de kringloop.",
        "ai_uitleg": "De AI herkent de kom aan de ronde vorm en open bovenkant.",
        "score": 8
    }
}

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "total_score" not in st.session_state:
    st.session_state.total_score = 0

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

if "last_object" not in st.session_state:
    st.session_state.last_object = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "current_scan_confirmed" not in st.session_state:
    st.session_state.current_scan_confirmed = False

if "current_image_id" not in st.session_state:
    st.session_state.current_image_id = None

if "feedback_saved" not in st.session_state:
    st.session_state.feedback_saved = False

# ------------------------------------------------------
# FUNCTIES
# ------------------------------------------------------
def reset_uploader():
    st.session_state.uploader_key += 1
    st.session_state.last_object = None
    st.session_state.current_scan_confirmed = False
    st.session_state.current_image_id = None
    st.session_state.feedback_saved = False
    st.rerun()

def reset_score():
    st.session_state.total_score = 0
    st.session_state.scan_count = 0
    st.session_state.last_object = None
    st.session_state.current_scan_confirmed = False
    st.session_state.current_image_id = None
    st.session_state.feedback_saved = False
    st.session_state.uploader_key += 1
    st.rerun()

def confidence_label(confidence):
    percentage = round(confidence * 100, 1)

    if percentage >= 70:
        return "Hoge zekerheid"
    elif percentage >= 40:
        return "Redelijke zekerheid"
    else:
        return "Lage zekerheid"

def show_score_button(info):
    st.markdown(
        f"""
        <div class="score-banner">
            <div>
                <div class="score-label">Score voor deze scan</div>
                <div class="score-text">Je krijgt <strong>+{info['score']} punten</strong> als je deze scan bevestigt.</div>
            </div>
            <div class="score-pill">+{info['score']} punten</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="button-space"></div>', unsafe_allow_html=True)

    if not st.session_state.current_scan_confirmed:
        if st.button(f"Score toevoegen (+{info['score']} punten)", type="primary"):
            st.session_state.total_score += info["score"]
            st.session_state.scan_count += 1
            st.session_state.current_scan_confirmed = True
            st.rerun()
    else:
        st.success(f"Scan bevestigd. +{info['score']} duurzaamheidspunten toegevoegd.")

def show_test_feedback(variant):
    st.divider()
    st.markdown("### Korte testvraag")

    st.caption("Help ons verbeteren door deze scan kort te beoordelen.")

    col1, col2, col3 = st.columns(3)

    with col1:
        duidelijkheid = st.slider("Duidelijkheid", 1, 5, 3)

    with col2:
        gemak = st.slider("Gebruiksgemak", 1, 5, 3)

    with col3:
        aantrekkelijk = st.slider("Aantrekkelijkheid", 1, 5, 3)

    begrip_ai = st.radio(
        "Was de AI-uitleg begrijpelijk?",
        ["Ja", "Een beetje", "Nee"],
        horizontal=True
    )

    if st.button("Feedback opslaan"):
        st.session_state.feedback_saved = True
        st.success(
            f"Bedankt! Feedback genoteerd voor {variant}. "
            f"Duidelijkheid: {duidelijkheid}/5, "
            f"Gebruiksgemak: {gemak}/5, "
            f"Aantrekkelijkheid: {aantrekkelijk}/5, "
            f"AI-uitleg: {begrip_ai}."
        )

# ------------------------------------------------------
# CSS
# ------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 0px;
        color: inherit;
    }

    .subtitle {
        font-size: 18px;
        color: #d1d5db;
        margin-bottom: 25px;
    }

    .white-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #111827 !important;
        margin-bottom: 14px;
        line-height: 1.5;
    }

    .white-card * {
        color: #111827 !important;
    }

    .plain-card {
        padding: 18px;
        border-radius: 10px;
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        color: #111827 !important;
        margin-bottom: 14px;
        line-height: 1.6;
    }

    .plain-card * {
        color: #111827 !important;
    }

    .hero-card {
        padding: 26px;
        border-radius: 24px;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        color: #064e3b !important;
        margin-bottom: 16px;
        text-align: center;
    }

    .hero-card * {
        color: #064e3b !important;
    }

    .hero-icon {
        font-size: 58px;
        margin-bottom: 6px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        font-weight: 600;
    }

    .visual-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #111827 !important;
        height: 250px;
        margin-bottom: 12px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .visual-card * {
        color: #111827 !important;
    }

    .visual-icon {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .small-label {
        font-size: 14px;
        color: #4b5563 !important;
        margin-bottom: 8px;
        font-weight: 700;
    }

    .big-result {
        font-size: 23px;
        font-weight: 900;
        margin-bottom: 8px;
        color: #111827 !important;
    }

    .compact-text {
        font-size: 16px;
        line-height: 1.5;
        color: #111827 !important;
    }

    .score-banner {
        padding: 16px;
        border-radius: 16px;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        color: #78350f !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 14px;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .score-banner * {
        color: #78350f !important;
    }

    .score-label {
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .score-text {
        font-size: 16px;
        line-height: 1.4;
    }

    .score-pill {
        background-color: #ffffff;
        border: 1px solid #f59e0b;
        border-radius: 999px;
        padding: 10px 14px;
        font-weight: 900;
        white-space: nowrap;
    }

    .button-space {
        height: 8px;
    }

    .warning-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404 !important;
        margin-top: 14px;
        font-size: 16px;
        line-height: 1.5;
    }

    .warning-box * {
        color: #856404 !important;
    }

    .info-box {
        padding: 16px;
        border-radius: 14px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460 !important;
        margin-top: 14px;
        margin-bottom: 18px;
        font-size: 16px;
        line-height: 1.5;
    }

    .info-box * {
        color: #0c5460 !important;
    }

    .ab-a-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        color: #111827 !important;
        margin-bottom: 12px;
    }

    .ab-a-box * {
        color: #111827 !important;
    }

    .ab-b-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #064e3b !important;
        margin-bottom: 12px;
    }

    .ab-b-box * {
        color: #064e3b !important;
    }

    .bottom-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #111827;
        border: 1px solid #374151;
        color: #f9fafb !important;
        min-height: 170px;
        line-height: 1.5;
    }

    .bottom-card * {
        color: #f9fafb !important;
    }

    .bottom-card h3 {
        margin-top: 0;
        margin-bottom: 10px;
    }

    @media screen and (max-width: 768px) {
        .main-title {
            font-size: 34px;
        }

        .hero-title {
            font-size: 25px;
        }

        .visual-card {
            height: auto;
            min-height: 210px;
        }

        .score-banner {
            flex-direction: column;
            align-items: flex-start;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.markdown('<div class="main-title">♻️ AI Recycle Coach</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Maak of upload een foto van afval en krijg direct duidelijk recycleadvies.</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------
# KPI'S
# ------------------------------------------------------
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Bevestigde scans", st.session_state.scan_count)

with kpi2:
    st.metric("Duurzaamheidsscore", st.session_state.total_score)

with kpi3:
    st.metric("Status", "Prototype")

st.divider()

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
with st.sidebar:
    st.header("Onderzoek & A/B-test")

    variant = st.radio(
        "Kies testvariant",
        ["Variant A - tekstueel", "Variant B - visuele coach"],
        help="Variant A toont vooral tekst. Variant B toont dezelfde informatie visueel en interactief."
    )

    if variant == "Variant A - tekstueel":
        st.markdown(
            """
            <div class="ab-a-box">
            <strong>Variant A:</strong><br>
            Klassieke tekstuele uitleg. De gebruiker moet vooral lezen.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="ab-b-box">
            <strong>Variant B:</strong><br>
            Visuele coach met iconen, grote advieskaart, score en korte uitleg.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("**Wat testen we?**")
    st.write(
        "We vergelijken of gebruikers het advies beter begrijpen via tekst of via een visuele recyclecoach."
    )

    st.write("**KPI's:**")
    st.write("- Duidelijkheid")
    st.write("- Gebruiksgemak")
    st.write("- Aantrekkelijkheid")
    st.write("- Begrip van AI")

    st.divider()

    st.write("**Privacy:**")
    st.write("Er worden geen persoonsgegevens opgeslagen.")
    st.write("Geüploade foto's worden alleen gebruikt voor de scan en worden niet opgeslagen.")
    st.write("Feedback wordt alleen in deze sessie getoond.")

    if st.button("Reset score"):
        reset_score()

# ------------------------------------------------------
# HOOFDLAYOUT
# ------------------------------------------------------
left_col, right_col = st.columns([1, 2])

uploaded_file = None
image = None

with left_col:
    st.subheader("1. Upload product")

    upload_method = st.radio(
        "Kies invoermethode",
        ["Foto uploaden", "Camera gebruiken"],
        horizontal=True
    )

    if upload_method == "Foto uploaden":
        uploaded_file = st.file_uploader(
            "Kies een afbeelding",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
    else:
        uploaded_file = st.camera_input(
            "Maak direct een foto",
            key=f"camera_{st.session_state.uploader_key}"
        )

    st.caption("Tip: maak een duidelijke foto van één product op een rustige achtergrond.")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        image_id = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.current_image_id != image_id:
            st.session_state.current_image_id = image_id
            st.session_state.current_scan_confirmed = False
            st.session_state.last_object = None
            st.session_state.feedback_saved = False

        image = ImageOps.exif_transpose(
            Image.open(BytesIO(file_bytes))
        ).convert("RGB")

        st.image(
            image,
            caption="Geüploade afbeelding",
            width=260
        )

        if st.button("Scan volgende product"):
            reset_uploader()

# ------------------------------------------------------
# RESULTAAT
# ------------------------------------------------------
with right_col:
    st.subheader("2. Resultaat")

    if uploaded_file is None:
        st.info("Upload links een afbeelding om de AI-scan te starten.")

        st.markdown(
            """
            <div class="white-card">
                <strong>Wat krijg je te zien?</strong><br><br>
                1. De AI probeert het product te herkennen.<br>
                2. Je ziet meteen in welke afvalbak het hoort.<br>
                3. Je krijgt een korte milieu-impact en een duurzame tip.<br>
                4. Je kunt scorepunten toevoegen na een scan.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        with st.spinner("AI analyseert de afbeelding..."):
            results = model(image)

        detected_objects = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = model.names[class_id]

                detected_objects.append({
                    "object_name": object_name,
                    "confidence": confidence
                })

        if len(detected_objects) == 0:
            st.error("De AI heeft geen object kunnen herkennen.")
            st.warning("Gebruik een duidelijke foto met één object op een rustige achtergrond.")

        else:
            best_detection = max(detected_objects, key=lambda x: x["confidence"])
            object_name = best_detection["object_name"]
            confidence = best_detection["confidence"]
            percentage = round(confidence * 100, 1)

            st.session_state.last_object = object_name

            if object_name in afval_info:
                info = afval_info[object_name]

                # ------------------------------------------------------
                # VARIANT A: TEKSTUELE VERSIE
                # ------------------------------------------------------
                if variant == "Variant A - tekstueel":
                    st.markdown("### Variant A: tekstuele uitleg")

                    st.markdown(
                        f"""
                        <div class="plain-card">
                            <strong>AI-herkenning</strong><br><br>
                            De AI denkt dat het object waarschijnlijk een <strong>{object_name}</strong> is.
                            De zekerheid van deze voorspelling is <strong>{percentage}%</strong>.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(min(confidence, 1.0))
                    st.caption(f"AI-zekerheid: {confidence_label(confidence)}")

                    st.markdown(
                        f"""
                        <div class="plain-card">
                            <strong>Afvaladvies</strong><br><br>
                            Herkend object: <strong>{info['naam']}</strong><br>
                            Juiste afvalbak: <strong>{info['bak']}</strong><br><br>
                            Milieu-impact: {info['impact']}<br><br>
                            Duurzame tip: {info['tip']}<br><br>
                            Punten voor deze scan: <strong>+{info['score']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="info-box">
                            <strong>AI-uitleg:</strong><br>
                            {info['ai_uitleg']} Het model vergelijkt patronen in de foto met objecten die het eerder heeft geleerd.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------------
                # VARIANT B: VISUELE VERSIE
                # ------------------------------------------------------
                else:
                    st.markdown("### Variant B: visuele recyclecoach")

                    st.markdown(
                        f"""
                        <div class="hero-card">
                            <div class="hero-icon">{info['icoon']}</div>
                            <div class="hero-title">{info['bak_icoon']} {info['bak']}</div>
                            <div class="hero-subtitle">Herkend als: {info['naam']} · Zekerheid: {percentage}% · +{info['score']} punten</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(min(confidence, 1.0))
                    st.caption(f"AI-zekerheid: {confidence_label(confidence)}")

                    card1, card2, card3 = st.columns(3)

                    with card1:
                        st.markdown(
                            f"""
                            <div class="visual-card">
                                <div class="visual-icon">{info['bak_icoon']}</div>
                                <div class="small-label">Afvalbak</div>
                                <div class="big-result">{info['bak']}</div>
                                <div class="compact-text">Dit is de juiste plek voor dit product.</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with card2:
                        st.markdown(
                            f"""
                            <div class="visual-card">
                                <div class="visual-icon">{info['impact_icoon']}</div>
                                <div class="small-label">Milieu-impact</div>
                                <div class="big-result">Waarom?</div>
                                <div class="compact-text">{info['impact']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with card3:
                        st.markdown(
                            f"""
                            <div class="visual-card">
                                <div class="visual-icon">{info['tip_icoon']}</div>
                                <div class="small-label">Duurzame tip</div>
                                <div class="big-result">Probeer dit</div>
                                <div class="compact-text">{info['tip']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <div class="info-box">
                            <strong>AI-uitleg:</strong><br>
                            {info['ai_uitleg']} De AI vergelijkt de foto met voorbeelden en kiest het object dat het meest lijkt op jouw afbeelding.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ------------------------------------------------------
                # SCORE
                # ------------------------------------------------------
                show_score_button(info)

                # ------------------------------------------------------
                # WAARSCHUWING
                # ------------------------------------------------------
                st.markdown(
                    """
                    <div class="warning-box">
                        <strong>Let op:</strong> dit is een prototype. AI-herkenning is niet altijd perfect.
                        Controleer bij twijfel altijd de lokale afvalregels.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------------------
                # TESTFEEDBACK
                # ------------------------------------------------------
                show_test_feedback(variant)

            else:
                st.warning("Dit object wordt nog niet ondersteund in onze afvaldatabase.")

                st.markdown(
                    f"""
                    <div class="white-card">
                        De AI heeft het object herkend als <strong>{object_name}</strong>
                        met een zekerheid van <strong>{percentage}%</strong>.
                        Voor dit object is nog geen afvaladvies toegevoegd.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <div class="info-box">
                        Voor de demo werken vooral deze objecten goed:
                        banana, apple, orange, bottle, cup, book, cell phone, laptop, keyboard en mouse.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ------------------------------------------------------
# ONDERSTE GEBRUIKERSINFORMATIE
# ------------------------------------------------------
st.divider()

bottom1, bottom2, bottom3 = st.columns(3)

with bottom1:
    st.markdown(
        """
        <div class="bottom-card">
            <h3>📸 Betere scan</h3>
            Maak een foto van één product tegelijk. Zorg voor genoeg licht en een rustige achtergrond.
            Zo kan de AI het object beter herkennen.
        </div>
        """,
        unsafe_allow_html=True
    )

with bottom2:
    st.markdown(
        """
        <div class="bottom-card">
            <h3>🏆 Score verzamelen</h3>
            Na elke ondersteunde scan kun je punten toevoegen. Zo wordt afval scheiden iets actiever
            en leuker om te proberen.
        </div>
        """,
        unsafe_allow_html=True
    )

with bottom3:
    st.markdown(
        """
        <div class="bottom-card">
            <h3>♻️ Snel advies</h3>
            Je krijgt direct de afvalbak, milieu-impact en een praktische tip te zien.
            Geen lange uitleg, maar meteen bruikbare informatie.
        </div>
        """,
        unsafe_allow_html=True
    )

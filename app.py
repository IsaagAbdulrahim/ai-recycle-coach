import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps

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
        "kleur": "green",
        "impact": "Bananenschillen kunnen verwerkt worden tot compost of biogas.",
        "tip": "Gooi fruitresten bij het GFT-afval.",
        "score": 10
    },
    "apple": {
        "naam": "Appel",
        "bak": "GFT",
        "icoon": "🍎",
        "kleur": "green",
        "impact": "Appelresten zijn biologisch afbreekbaar en geschikt voor compost.",
        "tip": "Gooi klokhuizen en fruitschillen bij het GFT.",
        "score": 10
    },
    "orange": {
        "naam": "Sinaasappel",
        "bak": "GFT",
        "icoon": "🍊",
        "kleur": "green",
        "impact": "Sinaasappelschillen kunnen opnieuw gebruikt worden als compost.",
        "tip": "Gooi fruitschillen bij het GFT-afval.",
        "score": 10
    },
    "bottle": {
        "naam": "Fles",
        "bak": "PMD / Plastic",
        "icoon": "🧴",
        "kleur": "blue",
        "impact": "Plastic kan opnieuw gebruikt worden als het goed wordt gescheiden.",
        "tip": "Maak de fles leeg voordat je hem weggooit.",
        "score": 15
    },
    "cup": {
        "naam": "Beker",
        "bak": "Restafval of papier",
        "icoon": "🥤",
        "kleur": "orange",
        "impact": "Bekers met een plastic laagje zijn vaak moeilijk te recyclen.",
        "tip": "Gebruik liever een herbruikbare beker.",
        "score": 8
    },
    "book": {
        "naam": "Boek",
        "bak": "Papier / Karton",
        "icoon": "📘",
        "kleur": "blue",
        "impact": "Papier kan goed gerecycled worden als het schoon en droog blijft.",
        "tip": "Lever boeken die nog goed zijn liever in bij een kringloop of bibliotheek.",
        "score": 12
    },
    "cell phone": {
        "naam": "Mobiele telefoon",
        "bak": "Elektronisch afval",
        "icoon": "📱",
        "kleur": "violet",
        "impact": "Elektronica bevat waardevolle metalen die opnieuw gebruikt kunnen worden.",
        "tip": "Lever oude telefoons in bij een inzamelpunt voor e-waste.",
        "score": 20
    },
    "laptop": {
        "naam": "Laptop",
        "bak": "Elektronisch afval",
        "icoon": "💻",
        "kleur": "violet",
        "impact": "Laptops bevatten grondstoffen die hergebruikt kunnen worden.",
        "tip": "Breng oude laptops naar een milieustraat of elektronica-inzamelpunt.",
        "score": 20
    },
    "keyboard": {
        "naam": "Toetsenbord",
        "bak": "Elektronisch afval",
        "icoon": "⌨️",
        "kleur": "violet",
        "impact": "Elektronische apparaten horen niet bij het restafval.",
        "tip": "Lever oude randapparatuur in bij een e-waste punt.",
        "score": 15
    },
    "mouse": {
        "naam": "Computermuis",
        "bak": "Elektronisch afval",
        "icoon": "🖱️",
        "kleur": "violet",
        "impact": "Kleine elektronica bevat materialen die opnieuw gebruikt kunnen worden.",
        "tip": "Gooi kleine elektronica niet zomaar in de prullenbak.",
        "score": 15
    },
    "scissors": {
        "naam": "Schaar",
        "bak": "Metaal / Restafval",
        "icoon": "✂️",
        "kleur": "gray",
        "impact": "Metaal kan vaak opnieuw worden verwerkt.",
        "tip": "Controleer lokaal of kleine metalen voorwerpen apart ingeleverd kunnen worden.",
        "score": 8
    },
    "vase": {
        "naam": "Glazen object",
        "bak": "Glasbak of restafval",
        "icoon": "🏺",
        "kleur": "gray",
        "impact": "Niet al het glas hoort in de glasbak. Drinkglazen en vazen horen vaak bij restafval.",
        "tip": "Controleer of het om verpakkingsglas gaat. Alleen dat hoort meestal in de glasbak.",
        "score": 8
    },
    "bowl": {
        "naam": "Kom",
        "bak": "Restafval",
        "icoon": "🥣",
        "kleur": "gray",
        "impact": "Servies zoals kommen en borden hoort meestal niet in de glasbak.",
        "tip": "Is het nog bruikbaar? Breng het dan naar de kringloop.",
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

# ------------------------------------------------------
# FUNCTIES
# ------------------------------------------------------
def reset_uploader():
    st.session_state.uploader_key += 1
    st.session_state.last_object = None
    st.session_state.current_scan_confirmed = False
    st.rerun()

def reset_score():
    st.session_state.total_score = 0
    st.session_state.scan_count = 0
    st.session_state.last_object = None
    st.session_state.current_scan_confirmed = False
    st.session_state.uploader_key += 1
    st.rerun()

# ------------------------------------------------------
# CSS
# ------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
        color: #ffffff;
    }

    .subtitle {
        font-size: 18px;
        color: #d1d5db;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 18px;
        border-radius: 16px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
        color: #111827 !important;
        min-height: 120px;
    }

    .result-card * {
        color: #111827 !important;
    }

    .small-label {
        font-size: 14px;
        color: #4b5563 !important;
        margin-bottom: 8px;
    }

    .big-result {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #111827 !important;
    }

    .compact-text {
        font-size: 16px;
        line-height: 1.5;
        color: #111827 !important;
    }

    .warning-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404 !important;
        margin-top: 12px;
        font-size: 16px;
        line-height: 1.5;
    }

    .warning-box * {
        color: #856404 !important;
    }

    .success-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724 !important;
        margin-top: 12px;
        font-size: 16px;
        line-height: 1.5;
    }

    .success-box * {
        color: #155724 !important;
    }

    .info-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460 !important;
        margin-top: 12px;
        font-size: 16px;
        line-height: 1.5;
    }

    .info-box * {
        color: #0c5460 !important;
    }

    .prototype-box {
        padding: 14px;
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #111827 !important;
        margin-bottom: 12px;
        font-size: 16px;
        line-height: 1.5;
    }

    .prototype-box * {
        color: #111827 !important;
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
    '<div class="subtitle">Upload een afvalproduct en ontdek direct waar het hoort, wat de milieu-impact is en welke duurzame tip erbij past.</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------
# KPI'S
# ------------------------------------------------------
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Aantal bevestigde scans", st.session_state.scan_count)

with kpi2:
    st.metric("Duurzaamheidsscore", st.session_state.total_score)

with kpi3:
    st.metric("Prototypefase", "MVP")

st.divider()

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
with st.sidebar:
    st.header("Onderzoek & test")

    variant = st.radio(
        "A/B-test variant",
        ["Variant A - tekstueel", "Variant B - visueel"],
        help="Gebruik dit om te testen welke uitleg gebruikers duidelijker vinden."
    )

    st.write("**Doel A/B-test:**")
    st.write("Onderzoeken of gebruikers informatie beter begrijpen via tekstuele uitleg of visuele blokken.")

    st.write("**Te meten KPI's:**")
    st.write("- Duidelijkheid")
    st.write("- Gebruiksgemak")
    st.write("- Aantrekkelijkheid")
    st.write("- Begrip van AI")

    st.divider()

    st.write("**Privacy:**")
    st.write("Er worden geen persoonsgegevens opgeslagen.")

    if st.button("Reset score"):
        reset_score()

# ------------------------------------------------------
# HOOFDLAYOUT
# ------------------------------------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("1. Upload product")

    uploaded_file = st.file_uploader(
        "Kies een afbeelding",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    st.caption("Tip: gebruik duidelijke foto's van bijvoorbeeld banaan, fles, boek, beker of telefoon.")

    image = None

    if uploaded_file:
    # Corrigeert automatisch iPhone/Android rotatie
    image = ImageOps.exif_transpose(
        Image.open(uploaded_file)
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

    if not uploaded_file:
        st.info("Upload links een afbeelding om de AI-scan te starten.")

        st.markdown(
            """
            <div class="prototype-box">
                <strong>Wat doet dit prototype?</strong><br><br>
                - Herkent objecten met AI.<br>
                - Koppelt objecten aan een afvalcategorie.<br>
                - Geeft korte uitleg over milieu-impact.<br>
                - Geeft een duurzame tip.<br>
                - Ondersteunt A/B-testen.
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

            st.session_state.last_object = object_name

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="small-label">AI-herkenning</div>
                    <div class="big-result">AI herkent waarschijnlijk: {object_name}</div>
                    <div class="compact-text">Zekerheid: {round(confidence * 100, 1)}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if object_name in afval_info:
                info = afval_info[object_name]

                if not st.session_state.current_scan_confirmed:
                    if st.button("Bevestig scan en voeg score toe"):
                        st.session_state.total_score += info["score"]
                        st.session_state.scan_count += 1
                        st.session_state.current_scan_confirmed = True
                        st.rerun()
                else:
                    st.success(f"Scan bevestigd. +{info['score']} duurzaamheidspunten toegevoegd.")

                if variant == "Variant A - tekstueel":
                    # ------------------------------------------------------
                    # VARIANT A: TEKSTUEEL
                    # ------------------------------------------------------
                    st.markdown("### Advies")

                    st.markdown(
                        f"""
                        <div class="prototype-box">
                            <strong>Herkend object:</strong> {info['naam']}<br>
                            <strong>Afvalbak:</strong> {info['bak']}<br>
                            <strong>Milieu-impact:</strong> {info['impact']}<br>
                            <strong>Duurzame tip:</strong> {info['tip']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <div class="info-box">
                            <strong>Hoe helpt AI hierbij?</strong><br>
                            De AI vergelijkt de afbeelding met patronen die het model eerder heeft geleerd.
                            Daardoor kan het systeem inschatten welk object op de foto staat.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    # ------------------------------------------------------
                    # VARIANT B: VISUEEL
                    # ------------------------------------------------------
                    card1, card2, card3 = st.columns(3)

                    with card1:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="small-label">Afvalbak</div>
                                <div class="big-result">{info['icoon']} {info['bak']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with card2:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="small-label">Milieu-impact</div>
                                <div class="compact-text">{info['impact']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with card3:
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="small-label">Duurzame tip</div>
                                <div class="compact-text">{info['tip']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        """
                        <div class="info-box">
                            <strong>AI-uitleg:</strong> De AI herkent patronen in de afbeelding en voorspelt welk object het meest waarschijnlijk zichtbaar is.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    """
                    <div class="warning-box">
                        <strong>Let op:</strong> dit is een prototype. AI-herkenning is niet altijd perfect.
                        Controleer bij twijfel altijd de lokale afvalregels.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.warning("Dit object wordt nog niet ondersteund in onze afvaldatabase.")

                st.markdown(
                    """
                    <div class="prototype-box">
                        De AI heeft wel iets herkend, maar wij hebben hier nog geen afvaladvies voor toegevoegd.
                        Dit prototype ondersteunt bewust een beperkt aantal voorbeeldobjecten.
                        Het doel is niet om alle afvalsoorten perfect te herkennen, maar om te testen of gebruikers begrijpen hoe AI kan helpen bij afvalscheiding.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <div class="info-box">
                        Voor de demo werken vooral deze objecten goed:
                        banana, apple, orange, bottle, cup, book, cell phone en laptop.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ------------------------------------------------------
# ONDERSTE UITLEG
# ------------------------------------------------------
st.divider()

bottom1, bottom2, bottom3 = st.columns(3)

with bottom1:
    st.markdown("### Waarom deze app?")
    st.write(
        "Uit de enquête bleek dat afval scheiden soms te veel moeite kost. "
        "Deze app maakt het makkelijker door direct advies te geven."
    )

with bottom2:
    st.markdown("### Wat wordt getest?")
    st.write(
        "Met de A/B-test vergelijken we tekstuele uitleg met visuele uitleg. "
        "Zo onderzoeken we welke versie duidelijker is."
    )

with bottom3:
    st.markdown("### Beperking")
    st.write(
        "Het prototype gebruikt een bestaand AI-model en ondersteunt een beperkt aantal objecten. "
        "De focus ligt op onderzoek en gebruikerservaring."
    )

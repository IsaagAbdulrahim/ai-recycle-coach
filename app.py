import streamlit as st
from ultralytics import YOLO
from google import genai
from PIL import Image, ImageOps
from io import BytesIO
from datetime import datetime
from urllib.parse import quote_plus
import hashlib
import pandas as pd

# ======================================================
# APP-INSTELLINGEN
# ======================================================
# Eindversie: de A/B-test is afgerond. De app toont alleen de visuele coachvariant.
GEMINI_MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title="AI Recycle Coach",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# AI-MODEL LADEN
# ======================================================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

@st.cache_resource
def load_gemini_client():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        return None

    return genai.Client(api_key=api_key)

gemini_client = load_gemini_client()

# ======================================================
# AFVALDATABASE
# ======================================================
# YOLO gebruikt Engelse labels. Daarom zijn de keys hieronder Engels.
WASTE_DB = {
    "banana": {
        "name": "banaan",
        "title": "Banaan",
        "category": "GFT-afval",
        "bin_text": "Groene GFT-bak",
        "where": "Gooi dit in de groene GFT-bak.",
        "icon": "🍌",
        "category_icon": "🟢",
        "impact": "Fruitresten kunnen worden verwerkt tot compost of biogas.",
        "tip": "Haal eventuele stickers of verpakking eraf en gooi alleen het fruitafval bij GFT.",
        "coach": "Goed bezig! Door GFT apart te scheiden help je mee om voedselresten opnieuw te gebruiken.",
        "ai_explanation": "Ik herken dit vooral aan de vorm en kleur.",
        "maps": "GFT afvalbak in de buurt"
    },
    "apple": {
        "name": "appel",
        "title": "Appel",
        "category": "GFT-afval",
        "bin_text": "Groene GFT-bak",
        "where": "Gooi dit in de groene GFT-bak.",
        "icon": "🍎",
        "category_icon": "🟢",
        "impact": "Appelresten zijn biologisch afbreekbaar en geschikt voor compost.",
        "tip": "Gooi klokhuizen en fruitschillen bij het GFT.",
        "coach": "Mooi! Kleine duurzame keuzes tellen mee als je ze vaker doet.",
        "ai_explanation": "Ik herken dit vooral aan de ronde vorm en kleur.",
        "maps": "GFT afvalbak in de buurt"
    },
    "orange": {
        "name": "sinaasappel",
        "title": "Sinaasappel",
        "category": "GFT-afval",
        "bin_text": "Groene GFT-bak",
        "where": "Gooi dit in de groene GFT-bak.",
        "icon": "🍊",
        "category_icon": "🟢",
        "impact": "Schillen van fruit kunnen opnieuw worden gebruikt als compost.",
        "tip": "Gooi fruitschillen bij het GFT-afval.",
        "coach": "Goed gescheiden! Zo voorkom je dat organisch afval onnodig bij restafval komt.",
        "ai_explanation": "Ik herken dit vooral aan de ronde vorm en oranje kleur.",
        "maps": "GFT afvalbak in de buurt"
    },
    "bottle": {
        "name": "fles",
        "title": "Fles",
        "category": "PMD / plastic",
        "bin_text": "PMD-bak of PMD-zak",
        "where": "Gooi dit bij PMD als het om een plastic verpakking gaat.",
        "icon": "🧴",
        "category_icon": "🔵",
        "impact": "Plastic kan opnieuw worden gebruikt als het goed wordt gescheiden.",
        "tip": "Maak de fles leeg en druk hem plat voordat je hem weggooit.",
        "coach": "Sterk! Lege plastic verpakkingen kunnen beter worden verwerkt als ze in de juiste afvalstroom terechtkomen.",
        "ai_explanation": "Ik herken dit vooral aan de langwerpige vorm en de dop.",
        "maps": "PMD afvalbak in de buurt"
    },
    "cup": {
        "name": "beker",
        "title": "Beker",
        "category": "Restafval of papier",
        "bin_text": "Controleer het materiaal",
        "where": "Bekers met een plastic laagje horen meestal bij restafval.",
        "icon": "🥤",
        "category_icon": "🟠",
        "impact": "Bekers met een plastic laagje zijn vaak moeilijk te recyclen.",
        "tip": "Gebruik liever een herbruikbare beker.",
        "coach": "Let op: bekers zijn soms lastig. Hergebruiken is hier vaak de beste keuze.",
        "ai_explanation": "Ik herken dit vooral aan de ronde opening en bekervorm.",
        "maps": "restafval afvalbak in de buurt"
    },
    "book": {
        "name": "boek",
        "title": "Boek",
        "category": "Papier / karton",
        "bin_text": "Papierbak of kringloop",
        "where": "Gooi oud papier in de papierbak. Is het boek nog bruikbaar? Breng het dan liever naar de kringloop.",
        "icon": "📘",
        "category_icon": "🔵",
        "impact": "Papier kan goed worden gerecycled als het schoon en droog blijft.",
        "tip": "Hergebruik is vaak duurzamer dan direct weggooien.",
        "coach": "Goede keuze! Eerst kijken naar hergebruik is vaak nog beter dan recyclen.",
        "ai_explanation": "Ik herken dit vooral aan de rechthoekige vorm en platte structuur.",
        "maps": "papierbak of kringloopwinkel in de buurt"
    },
    "cell phone": {
        "name": "mobiele telefoon",
        "title": "Mobiele telefoon",
        "category": "Elektronisch afval",
        "bin_text": "E-waste inzamelpunt",
        "where": "Lever dit in bij een e-waste inzamelpunt, milieustraat of elektronicawinkel.",
        "icon": "📱",
        "category_icon": "🟣",
        "impact": "Elektronica bevat waardevolle metalen die opnieuw gebruikt kunnen worden.",
        "tip": "Gooi oude telefoons nooit bij het restafval.",
        "coach": "Goed dat je dit apart houdt. Elektronica bevat waardevolle én soms schadelijke stoffen.",
        "ai_explanation": "Ik herken dit vooral aan de smalle rechthoekige vorm en het scherm.",
        "maps": "e-waste inzamelpunt in de buurt",
        "travel_tip": "Is het inleverpunt dichtbij? Pak de fiets of ga lopend. Zo blijft je duurzame keuze ook duurzaam onderweg."
    },
    "laptop": {
        "name": "laptop",
        "title": "Laptop",
        "category": "Elektronisch afval",
        "bin_text": "E-waste inzamelpunt",
        "where": "Breng dit naar een milieustraat, elektronicawinkel of e-waste inzamelpunt.",
        "icon": "💻",
        "category_icon": "🟣",
        "impact": "Laptops bevatten grondstoffen die hergebruikt kunnen worden.",
        "tip": "Verwijder persoonlijke data voordat je een laptop inlevert.",
        "coach": "Goed bezig! Elektronica hoort niet bij normaal huisafval.",
        "ai_explanation": "Ik herken dit vooral aan het scherm, toetsenbord en de hoekige vorm.",
        "maps": "e-waste inzamelpunt in de buurt",
        "travel_tip": "Is het inleverpunt dichtbij? Pak de fiets of ga lopend. Zo blijft je duurzame keuze ook duurzaam onderweg."
    },
    "keyboard": {
        "name": "toetsenbord",
        "title": "Toetsenbord",
        "category": "Elektronisch afval",
        "bin_text": "E-waste inzamelpunt",
        "where": "Lever dit in bij een e-waste inzamelpunt of milieustraat.",
        "icon": "⌨️",
        "category_icon": "🟣",
        "impact": "Elektronische apparaten bevatten materialen die opnieuw gebruikt kunnen worden.",
        "tip": "Lever oude randapparatuur in bij een e-waste punt.",
        "coach": "Goed bezig! Ook kleine elektronica verdient een aparte afvalstroom.",
        "ai_explanation": "Ik herken dit vooral aan de vorm en de herhaling van toetsen.",
        "maps": "e-waste inzamelpunt in de buurt",
        "travel_tip": "Is het inleverpunt dichtbij? Pak de fiets of ga lopend. Zo blijft je duurzame keuze ook duurzaam onderweg."
    },
    "mouse": {
        "name": "computermuis",
        "title": "Computermuis",
        "category": "Elektronisch afval",
        "bin_text": "E-waste inzamelpunt",
        "where": "Lever dit in bij een e-waste inzamelpunt of milieustraat.",
        "icon": "🖱️",
        "category_icon": "🟣",
        "impact": "Kleine elektronica bevat materialen die opnieuw gebruikt kunnen worden.",
        "tip": "Gooi kleine elektronica niet zomaar in de prullenbak.",
        "coach": "Mooi! Kleine elektronica wordt vaak vergeten, maar hoort niet bij restafval.",
        "ai_explanation": "Ik herken dit vooral aan de ronde vorm en het compacte ontwerp.",
        "maps": "e-waste inzamelpunt in de buurt",
        "travel_tip": "Is het inleverpunt dichtbij? Pak de fiets of ga lopend. Zo blijft je duurzame keuze ook duurzaam onderweg."
    },
    "chair": {
        "name": "stoel",
        "title": "Stoel",
        "category": "Grofvuil / kringloop",
        "bin_text": "Kringloop of milieustraat",
        "where": "Is de stoel nog bruikbaar? Breng hem naar de kringloop. Anders hoort hij meestal bij grofvuil of de milieustraat.",
        "icon": "🪑",
        "category_icon": "🟤",
        "impact": "Een stoel kan uit hout, metaal, plastic of stof bestaan. Daarom verschilt de juiste verwerking per materiaal.",
        "tip": "Controleer eerst of de stoel hergebruikt of gerepareerd kan worden.",
        "coach": "Goed dat je hier extra op let. Bij meubels is hergebruik vaak duurzamer dan direct weggooien.",
        "ai_explanation": "Ik herken dit vooral aan de poten, zitting en rugleuning.",
        "maps": "kringloopwinkel of milieustraat in de buurt",
        "travel_tip": "Is de kringloop of milieustraat dichtbij? Ga lopend, met de fiets of combineer het met een rit die je toch al moest maken."
    },
    "dining table": {
        "name": "tafel",
        "title": "Tafel",
        "category": "Grofvuil / kringloop",
        "bin_text": "Kringloop of milieustraat",
        "where": "Is de tafel nog bruikbaar? Breng hem naar de kringloop. Anders hoort hij meestal bij grofvuil of de milieustraat.",
        "icon": "🪵",
        "category_icon": "🟤",
        "impact": "Een tafel kan uit hout, metaal, glas of kunststof bestaan. De juiste verwerking hangt af van het materiaal.",
        "tip": "Kijk eerst of iemand anders de tafel nog kan gebruiken.",
        "coach": "Slim dat je dit controleert. Bij meubels is hergebruik meestal de beste duurzame stap.",
        "ai_explanation": "Ik herken dit vooral aan het tafelblad en de poten.",
        "maps": "kringloopwinkel of milieustraat in de buurt",
        "travel_tip": "Is de kringloop of milieustraat dichtbij? Ga lopend, met de fiets of combineer het met een rit die je toch al moest maken."
    },
    "scissors": {
        "name": "schaar",
        "title": "Schaar",
        "category": "Metaal / restafval",
        "bin_text": "Metaalbak, milieustraat of restafval",
        "where": "Controleer lokaal of kleine metalen voorwerpen apart ingeleverd kunnen worden.",
        "icon": "✂️",
        "category_icon": "⚫",
        "impact": "Metaal kan vaak opnieuw worden verwerkt.",
        "tip": "Is het nog bruikbaar? Geef het door of breng het naar de kringloop.",
        "coach": "Goed dat je controleert. Niet elk object past automatisch in een standaard afvalbak.",
        "ai_explanation": "Ik herken dit vooral aan de twee grepen en metalen bladen.",
        "maps": "milieustraat metaal afval in de buurt",
        "travel_tip": "Is de milieustraat dichtbij? Neem de fiets of combineer het met een andere rit."
    },
    "vase": {
        "name": "glazen object",
        "title": "Glazen object",
        "category": "Glasbak of restafval",
        "bin_text": "Controleer het soort glas",
        "where": "Alleen verpakkingsglas hoort meestal in de glasbak. Drinkglazen en vazen vaak niet.",
        "icon": "🏺",
        "category_icon": "⚫",
        "impact": "Verkeerd glas kan recycling verstoren.",
        "tip": "Controleer of het om verpakkingsglas gaat.",
        "coach": "Goed dat je dit controleert. Bij glas is het verschil tussen verpakkingsglas en servies belangrijk.",
        "ai_explanation": "Ik herken dit vooral aan de vorm en het glasachtige uiterlijk.",
        "maps": "glasbak in de buurt"
    },
    "bowl": {
        "name": "kom",
        "title": "Kom",
        "category": "Restafval of kringloop",
        "bin_text": "Restafval of kringloop",
        "where": "Servies zoals kommen en borden hoort meestal bij restafval of naar de kringloop als het nog bruikbaar is.",
        "icon": "🥣",
        "category_icon": "⚫",
        "impact": "Servies hoort meestal niet in de glasbak, omdat het een andere samenstelling heeft.",
        "tip": "Is het nog bruikbaar? Breng het dan naar de kringloop.",
        "coach": "Duurzaam denken is ook kijken of iets opnieuw gebruikt kan worden.",
        "ai_explanation": "Ik herken dit vooral aan de ronde vorm en open bovenkant.",
        "maps": "kringloopwinkel in de buurt"
    }
}

EXAMPLES = [
    ("🍌", "Banaan"), ("🍎", "Appel"), ("🍊", "Sinaasappel"), ("🧴", "Fles"),
    ("🥤", "Beker"), ("📘", "Boek"), ("📱", "Telefoon"), ("💻", "Laptop"),
    ("⌨️", "Toetsenbord"), ("🖱️", "Muis"), ("🪑", "Stoel"), ("🪵", "Tafel")
]

# ======================================================
# SESSION STATE
# ======================================================
DEFAULTS = {
    "page": "Home",
    "scan_count": 0,
    "daily_goal": 3,
    "uploader_key": 0,
    "current_image_id": None,
    "current_scan_confirmed": False,
    "scan_log": [],
    "chat_messages": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ======================================================
# FUNCTIES
# ======================================================
def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

def reset_current_scan():
    st.session_state.uploader_key += 1
    st.session_state.current_image_id = None
    st.session_state.current_scan_confirmed = False
    st.rerun()

def reset_all():
    st.session_state.scan_count = 0
    st.session_state.uploader_key += 1
    st.session_state.current_image_id = None
    st.session_state.current_scan_confirmed = False
    st.session_state.scan_log = []
    st.session_state.chat_messages = []
    st.rerun()

def maps_url(query):
    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(query)


def sustainable_travel_tip(info):
    """
    Geeft een extra coachende vervolgstap.
    Vooral nuttig bij producten die naar een inleverpunt, kringloop of milieustraat moeten.
    """
    return info.get(
        "travel_tip",
        "Moet je dit ergens inleveren? Kijk of je kunt lopen, fietsen of het kunt combineren met een rit die je toch al maakt."
    )

def progress_value():
    goal = max(int(st.session_state.daily_goal), 1)
    return min(st.session_state.scan_count / goal, 1.0)

def progress_percent():
    return int(progress_value() * 100)


def remaining_scans():
    goal = max(int(st.session_state.daily_goal), 1)
    return max(goal - st.session_state.scan_count, 0)


def progress_text():
    remaining = remaining_scans()

    if remaining == 0:
        return "Dagdoel gehaald. Lekker bezig!"
    if remaining == 1:
        return "Nog 1 product tot je dagdoel."
    return f"Nog {remaining} producten tot je dagdoel."


def coach_level():
    progress = progress_value()

    if progress >= 1:
        return "Dagdoel gehaald"
    if progress >= 0.66:
        return "Goed op weg"
    if progress >= 0.33:
        return "Lekker bezig"
    return "Begin vandaag"


def progress_ring_html(label="producten"):
    degrees = progress_percent() * 3.6
    return (
        f'<div class="progress-ring" style="background: conic-gradient(#22c55e {degrees}deg, #1f2937 0deg);">'
        f'<div class="progress-ring-inner">'
        f'<div>'
        f'<div class="ring-number">{st.session_state.scan_count}/{st.session_state.daily_goal}</div>'
        f'<div class="ring-label">{label}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

def confidence_label(confidence):
    percentage = round(confidence * 100, 1)

    if percentage >= 70:
        return "Hoge zekerheid"
    if percentage >= 40:
        return "Redelijke zekerheid"
    return "Lage zekerheid"

def create_image_id(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return hashlib.md5(file_bytes).hexdigest(), file_bytes

def detect_objects(image):
    results = model(image)
    detected = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = model.names[class_id]
            detected.append({"object_name": object_name, "confidence": confidence})

    return detected

def best_detection(detected):
    if not detected:
        return None
    return max(detected, key=lambda x: x["confidence"])

def confirm_scan(info, confidence):
    """Telt een scan één keer mee voor het dagdoel en bewaart alleen scaninformatie in de sessie."""
    if st.session_state.current_scan_confirmed:
        return

    st.session_state.scan_count += 1
    st.session_state.current_scan_confirmed = True

    st.session_state.scan_log.append({
        "tijd": datetime.now().strftime("%H:%M"),
        "product": info["title"],
        "categorie": info["category"],
        "zekerheid": round(confidence * 100, 1),
    })


def scan_log_df():
    visible_columns = ["tijd", "product", "categorie", "zekerheid"]

    if not st.session_state.scan_log:
        return pd.DataFrame(columns=visible_columns)

    df = pd.DataFrame(st.session_state.scan_log)

    for column in visible_columns:
        if column not in df.columns:
            df[column] = ""

    return df[visible_columns]

def category_df():
    df = scan_log_df()
    if df.empty:
        return pd.DataFrame({"categorie": ["Nog geen scans"], "aantal": [0]})
    out = df["categorie"].value_counts().reset_index()
    out.columns = ["categorie", "aantal"]
    return out

def ask_recycle_coach(question):
    if gemini_client is None:
        return (
            "De AI-chatcoach is tijdelijk niet beschikbaar."
        )

    prompt = f"""
    Jij bent een vriendelijke AI Recycle Coach voor gebruikers in Nederland.

    Je helpt gebruikers met:
    - afval scheiden;
    - recycling;
    - duurzaamheid;
    - motivatie om het dagdoel te halen;
    - uitleg over voortgang;
    - uitleg over hoe deze app werkt.

    Belangrijke regels:
    - Antwoord kort en duidelijk.
    - Gebruik eenvoudige taal voor mensen zonder technische kennis.
    - Geef maximaal 5 zinnen.
    - Geef altijd een praktische vervolgstap.
    - Zeg eerlijk dat afvalregels per gemeente kunnen verschillen.
    - Vraag niet om persoonsgegevens.
    - Geef geen medisch, juridisch of financieel advies.
    - Noem geen motivatiescore of punten; de app werkt met dagdoel en voortgang.
    - Als je twijfelt, zeg dat de gebruiker de lokale afvalregels moet controleren.

    Context van deze app:
    - De gebruiker kan een product scannen met AI.
    - De gebruiker kan vragen stellen aan deze chatcoach.
    - De gebruiker heeft een dagdoel.
    - De app toont voortgang met een voortgangsring.
    - De app slaat geen foto's of persoonsgegevens op.

    Gebruikerscontext:
    - Bevestigde scans: {st.session_state.scan_count}
    - Dagdoel: {st.session_state.daily_goal}
    - Resterende producten tot dagdoel: {remaining_scans()}
    - Coachniveau: {coach_level()}

    Vraag van de gebruiker:
    {question}
    """

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as error:
        return f"De AI-chatcoach is tijdelijk niet beschikbaar. Probeer het later opnieuw."


# ======================================================
# CSS
# ======================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    div[data-testid="column"] {
        min-width: 0;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    [data-testid="collapsedControl"] {
        top: 0.9rem;
        left: 0.9rem;
    }

    .app-title {
        font-size: 42px;
        font-weight: 950;
        letter-spacing: -1px;
        line-height: 1.55;
        margin-top: 0;
        margin-bottom: 6px;
        padding-top: 22px;
        overflow: visible;
    }

    .app-subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 18px;
        max-width: 950px;
    }

    .hero {
        padding: 30px;
        border-radius: 28px;
        background: linear-gradient(135deg, #052e2b 0%, #065f46 52%, #047857 100%);
        border: 1px solid rgba(167, 243, 208, 0.35);
        color: #ecfdf5 !important;
        margin-bottom: 20px;
        box-shadow: 0 14px 36px rgba(0,0,0,0.20);
    }

    .hero * {
        color: #ecfdf5 !important;
    }

    .hero-small {
        font-size: 14px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        opacity: 0.9;
        margin-bottom: 8px;
    }

    .hero-title {
        font-size: 39px;
        line-height: 1.05;
        font-weight: 950;
        margin-bottom: 12px;
    }

    .hero-text {
        font-size: 18px;
        line-height: 1.5;
        max-width: 820px;
    }

    .product-card {
        padding: 20px;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        color: #111827 !important;
        box-shadow: 0 7px 22px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .product-card * {
        color: #111827 !important;
    }

    .equal-card {
        min-height: 230px;
    }

    .coach-card {
        padding: 22px;
        border-radius: 24px;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        color: #064e3b !important;
        margin-bottom: 16px;
    }

    .coach-card * {
        color: #064e3b !important;
    }

    .coach-avatar {
        width: 68px;
        height: 68px;
        border-radius: 22px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        margin-bottom: 10px;
        border: 1px solid #bbf7d0;
    }


    .today-card {
        padding: 26px;
        border-radius: 28px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        border: 1px solid #374151;
        color: #f9fafb !important;
        margin-bottom: 18px;
        box-shadow: 0 14px 30px rgba(0,0,0,0.18);
    }

    .today-card * {
        color: #f9fafb !important;
    }

    .ring-wrap {
        display: flex;
        align-items: center;
        gap: 26px;
        flex-wrap: wrap;
    }

    .progress-ring {
        width: 178px;
        height: 178px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 0 0 1px #374151;
    }

    .progress-ring-inner {
        width: 132px;
        height: 132px;
        border-radius: 50%;
        background: #0b1120;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .ring-number {
        font-size: 34px;
        font-weight: 950;
        line-height: 1;
    }

    .ring-label {
        font-size: 13px;
        color: #9ca3af !important;
        margin-top: 6px;
    }

    .today-title {
        font-size: 28px;
        font-weight: 950;
        margin-bottom: 8px;
    }

    .today-text {
        color: #d1d5db !important;
        font-size: 17px;
        line-height: 1.5;
    }

    .badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 14px;
    }

    .badge {
        border-radius: 999px;
        background: #064e3b;
        border: 1px solid #10b981;
        padding: 8px 12px;
        font-weight: 800;
        color: #d1fae5 !important;
        font-size: 14px;
    }

    .scan-result {
        padding: 26px;
        border-radius: 28px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        color: #1e3a8a !important;
        margin-bottom: 16px;
        text-align: center;
        overflow-wrap: anywhere;
    }

    .scan-result * {
        color: #1e3a8a !important;
    }

    .result-icon {
        font-size: 56px;
        margin-bottom: 5px;
    }

    .result-title {
        font-size: 32px;
        font-weight: 950;
        margin-bottom: 8px;
    }

    .result-subtitle {
        font-size: 18px;
        line-height: 1.4;
        font-weight: 650;
    }

    .card-number {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background: #111827;
        color: #ffffff !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .card-label {
        font-size: 14px;
        font-weight: 900;
        color: #4b5563 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }

    .card-big {
        font-size: 23px;
        font-weight: 950;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .card-text {
        font-size: 16px;
        line-height: 1.45;
    }

    .coach-message {
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        color: #78350f !important;
        margin-top: 12px;
        margin-bottom: 12px;
        font-size: 17px;
        line-height: 1.5;
    }

    .coach-message * {
        color: #78350f !important;
    }

    .warning-box {
        padding: 14px;
        border-radius: 14px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412 !important;
        margin-top: 14px;
        font-size: 15px;
        line-height: 1.5;
    }

    .warning-box * {
        color: #9a3412 !important;
    }

    .info-box {
        padding: 16px;
        border-radius: 14px;
        background-color: #ecfeff;
        border: 1px solid #a5f3fc;
        color: #164e63 !important;
        margin-top: 14px;
        margin-bottom: 14px;
        font-size: 15px;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }

    .info-box * {
        color: #164e63 !important;
    }

    .pill-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 26px;
        max-width: 920px;
    }

    .pill {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 999px;
        padding: 10px 15px;
        font-size: 15px;
        color: #111827 !important;
        white-space: normal;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    .pill * {
        color: #111827 !important;
    }

    .action-space {
        height: 18px;
    }

    .footer-card {
        padding: 18px;
        border-radius: 20px;
        background: #111827;
        border: 1px solid #374151;
        color: #f9fafb !important;
        min-height: 145px;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    .footer-card * {
        color: #f9fafb !important;
    }

    @media screen and (max-width: 900px) {
        .block-container { max-width: 100%; }
        .app-title { font-size: 34px; }
        .hero-title { font-size: 30px; }
        .result-title { font-size: 25px; }
        .equal-card { min-height: auto; }
        .pill-wrap { gap: 8px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# SIDEBAR INSTELLINGEN + BOVENNAVIGATIE
# ======================================================
# In de eindversie staat alleen de visuele coachvariant aan.
# De A/B-testkeuze en feedbackknop zijn verwijderd.

with st.sidebar:
    st.markdown("## ⚙️ Instellingen")
    st.caption("Pas je dagdoel aan en bekijk de app-instellingen.")

    st.markdown("### Dagdoel")
    st.slider(
        "Hoeveel producten wil je vandaag goed scheiden?",
        min_value=1,
        max_value=20,
        step=1,
        key="daily_goal"
    )

    st.progress(progress_value())
    st.caption(progress_text())

    st.divider()

    st.markdown("### Chatcoach")
    if gemini_client is None:
        st.warning("De AI-chatcoach is tijdelijk niet beschikbaar.")
    else:
        st.success("De AI-chatcoach is klaar voor gebruik.")

    st.divider()

    st.markdown("### Privacy")
    st.caption(
        "Foto's worden alleen gebruikt voor de scan en worden niet opgeslagen. "
        "Stel geen persoonlijke gegevens in de chat."
    )

    st.divider()

    if st.button("Sessie resetten", use_container_width=True):
        reset_all()

# Header
st.markdown('<div class="app-title">♻️ AI Recycle Coach</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Scan een product, stel vragen aan de AI-coach en werk aan je eigen dagdoel.</div>',
    unsafe_allow_html=True
)

# Navigatie bovenin
nav1, nav2, nav3, nav4 = st.columns([1, 1, 1, 1])

with nav1:
    if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state.page == "Home" else "secondary"):
        set_page("Home")

with nav2:
    if st.button("📸 Scannen", use_container_width=True, type="primary" if st.session_state.page == "Scannen" else "secondary"):
        set_page("Scannen")

with nav3:
    if st.button("💬 Chatcoach", use_container_width=True, type="primary" if st.session_state.page == "Chatcoach" else "secondary"):
        set_page("Chatcoach")

with nav4:
    if st.button("📊 Voortgang", use_container_width=True, type="primary" if st.session_state.page == "Voortgang" else "secondary"):
        set_page("Voortgang")

st.markdown("")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Gescheiden", st.session_state.scan_count)
with m2:
    st.metric("Dagdoel", st.session_state.daily_goal)
with m3:
    st.metric("Resterend", remaining_scans())
with m4:
    st.metric("Voortgang", f"{progress_percent()}%")

st.progress(progress_value())

# ======================================================
# HOME
# ======================================================
if st.session_state.page == "Home":
    left, right = st.columns([1.45, 1])

    with left:
        st.markdown(
            """
            <div class="hero">
                <div class="hero-small">Slimmer afval scheiden</div>
                <div class="hero-title">Hoi, ik help je met afval scheiden.</div>
                <div class="hero-text">
                    Maak of upload een foto van een product, of stel direct een vraag aan de chatcoach.
                    Je krijgt eenvoudig advies over de juiste afvalbak, hergebruik en duurzame keuzes.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                <div class="product-card equal-card">
                    <div class="card-number">1</div>
                    <div class="card-label">Scan</div>
                    <div class="card-big">Voeg een foto toe</div>
                    <div class="card-text">Gebruik één product tegelijk op een rustige achtergrond.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                """
                <div class="product-card equal-card">
                    <div class="card-number">2</div>
                    <div class="card-label">Vraag</div>
                    <div class="card-big">Gebruik de chatcoach</div>
                    <div class="card-text">Geen foto bij de hand? Stel gewoon je afvalvraag.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                """
                <div class="product-card equal-card">
                    <div class="card-number">3</div>
                    <div class="card-label">Coach</div>
                    <div class="card-big">Haal je dagdoel</div>
                    <div class="card-text">Bevestig correcte scans en volg je voortgang.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Voorbeelden die goed werken")

        pills = "".join([f'<span class="pill">{icon} {name}</span>' for icon, name in EXAMPLES])
        st.markdown(f'<div class="pill-wrap">{pills}</div>', unsafe_allow_html=True)

        st.markdown('<div class="action-space"></div>', unsafe_allow_html=True)

        btn1, btn2 = st.columns([1, 1])
        with btn1:
            if st.button("Start met scannen", type="primary", use_container_width=True):
                set_page("Scannen")
        with btn2:
            if st.button("Vraag het aan de chatcoach", use_container_width=True):
                set_page("Chatcoach")

    with right:
        st.markdown(
            f"""
            <div class="today-card">
                <div class="ring-wrap">
                    {progress_ring_html()}
                    <div>
                        <div class="today-title">Vandaag</div>
                        <div class="today-text">{progress_text()}</div>
                        <div class="badge-row">
                            <span class="badge">🎯 {coach_level()}</span>
                            <span class="badge">♻️ {progress_percent()}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box">
                <strong>Tip van je coach</strong><br>
                Elke goede keuze telt. Scan vooral producten waar je over twijfelt en ontdek direct wat de beste vervolgstap is.
            </div>
            """,
            unsafe_allow_html=True
        )

# ======================================================
# SCANNEN
# ======================================================
elif st.session_state.page == "Scannen":
    upload_col, result_col = st.columns([0.9, 2.1])

    uploaded_file = None
    image = None

    with upload_col:
        st.subheader("Foto toevoegen")

        method = st.radio(
            "Invoer",
            ["Uploaden", "Camera"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if method == "Uploaden":
            uploaded_file = st.file_uploader(
                "Kies een afbeelding",
                type=["jpg", "jpeg", "png"],
                key=f"file_uploader_{st.session_state.uploader_key}"
            )
        else:
            uploaded_file = st.camera_input(
                "Maak direct een foto",
                key=f"camera_{st.session_state.uploader_key}"
            )

        st.caption("Zet het product duidelijk in beeld. Vermijd dat personen groot in beeld staan.")

        if uploaded_file is not None:
            image_id, file_bytes = create_image_id(uploaded_file)

            if st.session_state.current_image_id != image_id:
                st.session_state.current_image_id = image_id
                st.session_state.current_scan_confirmed = False
            
            image = ImageOps.exif_transpose(Image.open(BytesIO(file_bytes))).convert("RGB")
            st.image(image, caption="Preview", width=240)

            if st.button("Nieuw product scannen"):
                reset_current_scan()

        st.markdown(
            """
            <div class="product-card">
                <strong>Privacy</strong><br>
                De foto wordt alleen gebruikt voor deze scan en wordt niet opgeslagen.
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col:
        st.subheader("Advies van je coach")

        if uploaded_file is None:
            st.markdown(
                """
                <div class="coach-card">
                    <div class="coach-avatar">🤖</div>
                    <h3>Ik ben klaar voor je scan</h3>
                    <p>Voeg links een foto toe. Daarna geef ik direct advies.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            with st.spinner("AI analyseert je foto..."):
                detected_objects = detect_objects(image)
                detection = best_detection(detected_objects)

            if detection is None:
                st.error("Ik kon geen object herkennen.")
                st.warning("Maak een duidelijkere foto met één product op een rustige achtergrond.")
            else:
                object_name = detection["object_name"]
                confidence = detection["confidence"]
                percentage = round(confidence * 100, 1)
                st.session_state.last_detection = detection

                person_detected = any(
                    item["object_name"] == "person" and item["confidence"] >= 0.35
                    for item in detected_objects
                )

                if object_name == "person":
                    st.markdown(
                        """
                        <div class="warning-box">
                            <strong>Ik zie vooral een persoon in beeld.</strong><br>
                            Zet het afvalproduct duidelijker in beeld en zorg dat de persoon niet het grootste deel van de foto inneemt.
                            Maak daarna opnieuw een foto.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif object_name not in WASTE_DB:
                    st.warning("Ik herkende iets, maar ik heb hier nog geen afvaladvies voor.")
                    st.markdown(
                        f"""
                        <div class="product-card">
                            Herkend als: <strong>{object_name}</strong><br>
                            Zekerheid: <strong>{percentage}%</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write("Probeer bijvoorbeeld een banaan, fles, boek, beker, telefoon, stoel of tafel.")
                else:
                    info = WASTE_DB[object_name]

                    if person_detected:
                        st.markdown(
                            """
                            <div class="warning-box">
                                <strong>Tip:</strong> ik zie ook een persoon in beeld.
                                Voor een betere scan kun je het product dichterbij houden en de achtergrond rustiger maken.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <div class="scan-result">
                            <div class="result-icon">{info['icon']}</div>
                            <div class="result-title">Dit is een {info['name']}</div>
                            <div class="result-subtitle">
                                Dit hoort bij <strong>{info['category']}</strong><br>
                                AI-zekerheid: {percentage}% · {confidence_label(confidence)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(min(confidence, 1.0))

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.markdown(
                            f"""
                            <div class="product-card equal-card">
                                <div class="card-number">1</div>
                                <div class="card-label">Wat is het?</div>
                                <div class="card-big">{info['icon']} {info['title']}</div>
                                <div class="card-text">{info['ai_explanation']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with r2:
                        st.markdown(
                            f"""
                            <div class="product-card equal-card">
                                <div class="card-number">2</div>
                                <div class="card-label">Waar hoort het?</div>
                                <div class="card-big">{info['category_icon']} {info['bin_text']}</div>
                                <div class="card-text">{info['where']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with r3:
                        st.markdown(
                            f"""
                            <div class="product-card equal-card">
                                <div class="card-number">3</div>
                                <div class="card-label">Waarom duurzaam?</div>
                                <div class="card-big">♻️ Impact</div>
                                <div class="card-text">{info['impact']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <div class="coach-message">
                            <strong>Coach zegt:</strong><br>
                            {info['coach']}<br><br>
                            <strong>Volgende stap:</strong> {info['tip']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    b1, b2 = st.columns([1, 1])
                    with b1:
                        st.link_button(
                            "Zoek inleverpunt in de buurt",
                            maps_url(info["maps"]),
                            use_container_width=True
                        )
                    with b2:
                        if not st.session_state.current_scan_confirmed:
                            if st.button("Ik heb dit correct weggegooid", type="primary", use_container_width=True):
                                confirm_scan(info, confidence)
                                st.rerun()
                        else:
                            st.success("Opgeslagen. Je voortgang is bijgewerkt.")

                    st.markdown(
                        """
                        <div class="info-box">
                            <strong>Voortgang bijgewerkt</strong><br>
                            Als je bevestigt dat je dit correct hebt weggegooid, telt deze scan mee voor je dagdoel.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div class="info-box">
                            <strong>Duurzame vervolgstap</strong><br>
                            {sustainable_travel_tip(info)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        """
                        <div class="warning-box">
                            AI kan fouten maken. Controleer bij twijfel altijd de lokale afvalregels.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ======================================================
# CHATCOACH
# ======================================================
elif st.session_state.page == "Chatcoach":
    st.subheader("💬 Chat met je AI Recycle Coach")

    intro_left, intro_right = st.columns([1.35, 1])

    with intro_left:
        st.markdown(
            """
            <div class="coach-card">
                <div class="coach-avatar">🤖</div>
                <h3>Vraag mij alles over afval</h3>
                <p>
                    Stel een vraag over afval scheiden, recycling, hergebruik of je dagdoel.
                    Ik geef kort advies en een duidelijke vervolgstap.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with intro_right:
        st.markdown(
            f"""
            <div class="today-card">
                <div class="ring-wrap">
                    {progress_ring_html("vandaag")}
                    <div>
                        <div class="today-title">{coach_level()}</div>
                        <div class="today-text">{progress_text()}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Snelle vragen")

    quick1, quick2, quick3, quick4 = st.columns(4)

    with quick1:
        quick_battery = st.button("🔋 Batterij", use_container_width=True)

    with quick2:
        quick_progress = st.button("📊 Mijn voortgang", use_container_width=True)

    with quick3:
        quick_goal = st.button("🎯 Dagdoel halen", use_container_width=True)

    with quick4:
        quick_pizza = st.button("🍕 Pizzadoos", use_container_width=True)

    quick_question = None

    if quick_battery:
        quick_question = "Waar hoort een batterij?"
    elif quick_progress:
        quick_question = "Hoe werkt mijn voortgang in deze app?"
    elif quick_goal:
        quick_question = "Hoe kan ik mijn dagdoel halen?"
    elif quick_pizza:
        quick_question = "Waar hoort een lege pizzadoos?"

    if quick_question:
        with st.spinner("De AI-coach denkt na..."):
            answer = ask_recycle_coach(quick_question)

        st.session_state.chat_messages.append({
            "question": quick_question,
            "answer": answer,
            "time": datetime.now().strftime("%H:%M")
        })

        st.rerun()

    st.markdown("### Gesprek")

    chat_box = st.container(height=520, border=True)

    with chat_box:
        if not st.session_state.chat_messages:
            st.info("Typ onderaan je eerste vraag. Bijvoorbeeld: Waar hoort een plastic fles?")
        else:
            for message in st.session_state.chat_messages:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**Jij · {message['time']}**")
                    st.markdown(message["question"])

                with st.chat_message("assistant", avatar="♻️"):
                    st.markdown("**AI Recycle Coach**")
                    st.markdown(message["answer"])

    user_prompt = st.chat_input("Typ je vraag aan de AI Recycle Coach...")

    if user_prompt:
        user_prompt = user_prompt.strip()

        if user_prompt:
            with st.spinner("De AI-coach denkt na..."):
                answer = ask_recycle_coach(user_prompt)

            st.session_state.chat_messages.append({
                "question": user_prompt,
                "answer": answer,
                "time": datetime.now().strftime("%H:%M")
            })

            st.rerun()

    chat_left, chat_right = st.columns([1, 2])

    with chat_left:
        if st.button("Chat wissen", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    with chat_right:
        st.markdown(
            """
            <div class="warning-box">
                De AI-coach kan fouten maken. Controleer bij twijfel altijd de afvalregels van je gemeente.
            </div>
            """,
            unsafe_allow_html=True
        )

# ======================================================
# VOORTGANG
# ======================================================
elif st.session_state.page == "Voortgang":
    st.subheader("📊 Jouw voortgang")

    df = scan_log_df()

    top_progress, top_tip = st.columns([1.25, 1])

    with top_progress:
        st.markdown(
            f"""
            <div class="today-card">
                <div class="ring-wrap">
                    {progress_ring_html()}
                    <div>
                        <div class="today-title">Vandaag</div>
                        <div class="today-text">{progress_text()}</div>
                        <div class="badge-row">
                            <span class="badge">🎯 {coach_level()}</span>
                            <span class="badge">♻️ {progress_percent()}%</span>
                            <span class="badge">✅ {st.session_state.scan_count} gescheiden</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with top_tip:
        if remaining_scans() == 0:
            tip_text = "Je dagdoel is gehaald. Kies morgen opnieuw een doel en probeer het vol te houden."
        else:
            tip_text = "Elke duurzame keuze telt. Scan producten waar je over twijfelt en ontdek direct de beste vervolgstap."

        st.markdown(
            f"""
            <div class="coach-card">
                <div class="coach-avatar">🏆</div>
                <h3>Coachadvies</h3>
                <p>{tip_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if df.empty:
        st.info("Je hebt nog geen bevestigde scans. Scan eerst een product.")
        if st.button("Ga naar scannen", type="primary"):
            set_page("Scannen")
    else:
        left, right = st.columns([1, 1])

        with left:
            st.markdown("#### Aantal scans per categorie")
            st.bar_chart(category_df().set_index("categorie"))

        with right:
            st.markdown("#### Recente scans")
            st.dataframe(df.tail(5), use_container_width=True, hide_index=True)

# ======================================================
# FOOTER
# ======================================================
st.divider()

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(
        """
        <div class="footer-card">
            <h3>🤖 Coach</h3>
            Je krijgt direct advies via scan of chat, motivatie en een duidelijke volgende stap.
        </div>
        """,
        unsafe_allow_html=True
    )
with f2:
    st.markdown(
        """
        <div class="footer-card">
            <h3>🔒 Privacy</h3>
            De app vraagt geen persoonsgegevens en slaat geen foto's op.
        </div>
        """,
        unsafe_allow_html=True
    )
with f3:
    st.markdown(
        """
        <div class="footer-card">
            <h3>📊 Dagdoel</h3>
            Kies zelf je doel en volg hoeveel producten je correct hebt gescheiden.
        </div>
        """,
        unsafe_allow_html=True
    )

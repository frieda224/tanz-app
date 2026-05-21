import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import math

# --- SEITEN-SETUP ---
st.set_page_config(page_title="Tanz-Trainer KI", layout="wide")

st.title("🩰 Tanz-Trainer KI")
st.write("Lade ein Foto deiner Pose hoch. Die KI vergleicht sie mit einem Profi und gibt dir Feedback!")

# --- KI-MODELL LADEN ---
# Wir laden das YOLOv8-Pose Modell direkt von Hugging Face / Ultralytics
@st.cache_resource
def load_pose_model():
    # Lädt automatisch das vortrainierte Pose-Modell (ca. 50 MB)
    return YOLO('yolov8n-pose.pt') 

with st.spinner("Tanz-Lehrer wird vorbereitet..."):
    model = load_pose_model()

# --- HILFSFUNKTION: WINKEL BERECHNEN ---
def calculate_angle(a, b, c):
    """Berechnet den Winkel zwischen drei Punkten (z.B. Schulter, Ellenbogen, Handgelenk)"""
    a = np.array(a) # Punkt A
    b = np.array(b) # Orientierungspunkt B
    c = np.array(c) # Punkt C
    
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# --- DEMO: PROFI-FOTO (REFERENZ) ---
# Für eine echte App lädst du hier ein Bild eines professionellen Tänzers hoch
st.subheader("1. Referenz-Pose des Profis")
profi_file = st.file_uploader("Schritt A: Profi-Foto hochladen (oder Beispiel nutzen)", type=["jpg", "png", "jpeg"], key="profi")

st.subheader("2. Dein Foto")
user_file = st.file_uploader("Schritt B: Lade dein Foto hoch", type=["jpg", "png", "jpeg"], key="user")

if profi_file and user_file:
    profi_img = Image.open(profi_file)
    user_img = Image.open(user_file)
    
    # KI analysiert beide Bilder
    res_profi = model(profi_img)
    res_user = model(user_img)
    
    # Spalten für die Anzeige
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(res_profi[0].plot(), caption="Profi-Pose (Analysiert)", use_container_width=True)
    with col2:
        st.image(res_user[0].plot(), caption="Deine Pose (Analysiert)", use_container_width=True)
        
    # Extrahiere die Keypoints (Gelenke)
    # Gelenke-Index bei YOLO: 5=L_Schulter, 7=L_Ellenbogen, 9=L_Handgelenk, 11=L_Hüfte, 13=L_Knie, 15=L_Knöchel
    try:
        kp_profi = res_profi[0].keypoints.xy[0].cpu().numpy()
        kp_user = res_user[0].keypoints.xy[0].cpu().numpy()
        
        # Beispiel-Vergleich: Linker Ellenbogen-Winkel
        # Punkte: Schulter (5), Ellenbogen (7), Handgelenk (9)
        winkel_profi = calculate_angle(kp_profi[5], kp_profi[7], kp_profi[9])
        winkel_user = calculate_angle(kp_user[5], kp_user[7], kp_user[9])
        
        # Abweichung berechnen
        unterschied = abs(winkel_profi - winkel_user)
        score = max(0, 100 - unterschied) # 100 ist perfekt
        
        # --- FEEDBACK AUSGABE ---
        st.divider()
        st.header("📋 Dein KI-Feedback")
        
        if score > 85:
            st.success(f"🌟 Großartig! Deine Haltung stimmt zu {score:.1f}% mit dem Profi überein!")
        elif score > 60:
            st.warning(f"👍 Gut, aber ausbaufähig (Übereinstimmung: {score:.1f}%). Achte auf deine Arme/Ellenbogen!")
        else:
            st.error(f"❌ Übereinstimmung: {score:.1f}%. Versuche, den Winkel deiner Arme stärker an das Profi-Bild anzupassen.")
            
        st.info(f"Winkel beim Profi: {winkel_profi:.1f}° | Dein Winkel: {winkel_user:.1f}°")
        
    except Exception as e:
        st.error("Die KI konnte auf einem der Bilder nicht alle Gelenke erkennen. Stelle sicher, dass der ganze Körper sichtbar ist!")

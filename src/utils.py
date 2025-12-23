import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import streamlit as st
import sys

# --- KONFIGURASI ROBOFLOW (LOAD DARI SECRETS) ---
try:
    # Mengambil value dari .streamlit/secrets.toml
    API_KEY = st.secrets["ROBOFLOW_API_KEY"]
    WORKSPACE_NAME = st.secrets["ROBOFLOW_WORKSPACE"]
    WORKFLOW_ID = st.secrets["ROBOFLOW_WORKFLOW"]
except FileNotFoundError:
    st.error("❌ File .streamlit/secrets.toml tidak ditemukan!")
    st.stop()
except KeyError as e:
    st.error(f"❌ Key {e} tidak ditemukan di secrets.toml")
    st.stop()

# Inisialisasi Client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# Warna Bounding Box
COLOR_BOX = (0, 0, 255) 

def run_ai_workflow(frame): 
    predictions = []
    
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": frame}
        )
        
        # ... (Lanjutkan kode parsing seperti sebelumnya) ...
        prediction_result = result[0]
        raw_preds = prediction_result.get("predictions", [])
        
        if not raw_preds:
            for key, val in prediction_result.items():
                if isinstance(val, dict) and "predictions" in val:
                    raw_preds = val["predictions"]
                    break
        
        for p in raw_preds:
            x, y, w, h = p['x'], p['y'], p['width'], p['height']
            label = p['class']
            conf = p['confidence']
            
            predictions.append({
                "class": label,
                "confidence": conf
            })

            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
            
            text = f"{label} {int(conf*100)}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), COLOR_BOX, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    except Exception as e:
        print(f"Workflow Error: {e}")
    
    return frame, predictions
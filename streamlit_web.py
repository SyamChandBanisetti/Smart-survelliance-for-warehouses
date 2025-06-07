import streamlit as st
import subprocess
import threading

st.set_page_config(page_title="🛡️ Smart Warehouse Surveillance", layout="wide")

st.title("📦 Smart Warehouse Surveillance Dashboard")
st.markdown("""
This system monitors:
- ✅ Gunny Bags & Boxes
- 🚗 Vehicles
- 😊 Faces

Click the button below to start **all modules at once**.
""")

# Define function to run each module in its own thread
def run_script(script_name):
    subprocess.run(["python", script_name])

if st.button("▶ Start Full Surveillance"):
    st.success("Surveillance started. Please check the OpenCV windows.")
    
    # Run each detection module in a separate thread
    threading.Thread(target=run_script, args=("gunny_box_detector.py",), daemon=True).start()
    threading.Thread(target=run_script, args=("vehicle_detector.py",), daemon=True).start()
    threading.Thread(target=run_script, args=("face_detector.py",), daemon=True).start()

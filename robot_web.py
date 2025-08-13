# robot_web.py
import subprocess
import streamlit as st

st.title("Robot Control Web Interface")

if st.button("Launch Desktop GUI"):
    subprocess.Popen(["python3", "robot_gui.py"])
    st.success("Desktop GUI launched in separate process")

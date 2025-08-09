import os
import subprocess

def run_streamlit():
    """
    Run the Streamlit app.
    """
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    subprocess.run(["streamlit", "run", "ui/streamlit_app.py"])

if __name__ == "__main__":
    run_streamlit()

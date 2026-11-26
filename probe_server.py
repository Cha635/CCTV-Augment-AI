"""
Probe script to test Streamlit server startup.
"""
import subprocess
import time
import urllib.request
import sys

def probe():
    print("Starting Streamlit in background...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
        cwd=r"C:\Users\Sripragna\Downloads\GenAI_Project\CCTV_Augment_AI",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    status_code = None
    for attempt in range(12):
        time.sleep(1.5)
        try:
            req = urllib.request.Request("http://localhost:8501")
            with urllib.request.urlopen(req, timeout=3) as resp:
                status_code = resp.status
                print(f"Attempt {attempt+1}: Streamlit HTTP Status = {status_code}")
                if status_code == 200:
                    break
        except Exception as e:
            # print(f"Attempt {attempt+1}: waiting... ({e})")
            pass
            
    # Terminate process
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
        
    if status_code == 200:
        print("SUCCESS: Streamlit web server is fully operational!")
        return 0
    else:
        print(f"FAIL: Server did not return HTTP 200. Final code: {status_code}")
        return 1

if __name__ == "__main__":
    sys.exit(probe())

import os
import subprocess
import webbrowser

def launch_ollama():
    print("Launching Ollama...")
    # Replace "Ollama" with the actual path if necessary
    subprocess.Popen(["open", "-a", "Ollama"])  

def launch_open_webui():
    print("Launching Open-WebUI...")
    # Replace with the correct URL or command to start Open-WebUI
    webbrowser.open("http://localhost:7860")  

if __name__ == "__main__":
    launch_ollama()
    launch_open_webui()
    print("Both applications have been launched!")
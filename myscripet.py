import pyautogui
import time
import subprocess
import webbrowser

# Function to open a new Terminal window
def open_terminal_and_run_command():
    # Open a new Terminal window
    subprocess.run(["open", "-a", "Terminal"])
    time.sleep(1)  # Wait for the terminal to open

    # Type the command to start the server
    pyautogui.typewrite("open-webui serve")
    pyautogui.press("enter")

# Function to open the default browser with a specific URL
def open_browser_with_url():
    url = "http://localhost:8080"
    time.sleep(5)  # Wait for the server to start
    webbrowser.open(url)

# Execute both actions
open_terminal_and_run_command()
open_browser_with_url()

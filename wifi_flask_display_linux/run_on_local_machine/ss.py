import mss
import mss.tools
import requests
import ctypes

def capture_screenshot():
    with mss.mss() as sct:
        # Choose the appropriate monitor index.
        monitor = sct.monitors[2]  # Adjust index as needed.
        screenshot = sct.grab(monitor)
        image_png = mss.tools.to_png(screenshot.rgb, screenshot.size)

    arrow_cursor = ctypes.windll.user32.LoadCursorW(None, 32512)
    ctypes.windll.user32.SetCursor(arrow_cursor)
    
    return image_png

def send_to_pi(image_bytes):
    url = "http://XXX.XXX.X.XXX:XXXX/upload"  # Replace with external device ip address
    files = {'image': ('screenshot.png', image_bytes, 'image/png')}
    response = requests.post(url, files=files)
    print("Response:", response.status_code)

if __name__ == "__main__":
    image_bytes = capture_screenshot()
    send_to_pi(image_bytes)

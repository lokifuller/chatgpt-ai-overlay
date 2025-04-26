#!/usr/bin/env python3
import os
import sys
import time
import threading
import base64
import openai
from PIL import Image, ImageGrab
from mimetypes import guess_type
from PyQt5 import QtWidgets, QtGui, QtCore

# Set your OpenAI API key here
openai.api_key = "sk-proj-############################################"

# Globals
ai_response = "Waiting for input..."
ai_response_updated = False
ai_processing = False
screenshot_path = ""
screenshot_dir = os.path.join(os.path.expanduser("~"), "Desktop", "screenshots")
os.makedirs(screenshot_dir, exist_ok=True)

class AIProcessor:
    @staticmethod
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @staticmethod
    def local_image_to_data_url(image_path):
        mime, _ = guess_type(image_path)
        if not mime: mime = 'application/octet-stream'
        b64 = AIProcessor.encode_image_to_base64(image_path)
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def get_text_from_image(image_path, prompt):
        global ai_response, ai_processing
        ai_response = "Processing image..."
        ai_processing = True
        data_uri = AIProcessor.local_image_to_data_url(image_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in extracting text from images."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}}
            ]}
        ]
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=5000
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"
        finally:
            ai_processing = False

    @staticmethod
    def generate_response(extracted_text, code_prompt_instruction):
        global ai_response, ai_processing
        ai_response = "Generating answers..."
        ai_processing = True
        prompt = (
            f"{code_prompt_instruction}\n\n"
            f"IMPORTANT: Respond in JSON format.\n\n"
            f"Input JSON:\n{extracted_text}"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="o1-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"
        finally:
            ai_processing = False

    @staticmethod
    def process_image(image_path):
        global ai_response, ai_response_updated
        ai_response = "AI processing started..."
        ai_response_updated = True

        try:
            # 1)
            data = AIProcessor.local_image_to_data_url(image_path)

            # 2) Extract questions and answer them
            system = {
                "role": "system",
                "content": "You are an assistant that looks at screenshots, finds *all* the questions in them, and then answers each question in turn.  For each question you find, output exactly two lines: the question (unchanged) on line 1, then its answer on line 2.  Do not wrap in JSON, do not output any extra text, just the alternating Q&A lines."
            }
            user = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data, "detail": "high"}}
                ]
            }

            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[system, user],
                max_tokens=1000
            )
            # 3) Directly display
            ai_response = resp.choices[0].message.content.strip()

        except Exception as e:
            ai_response = f"Error processing image: {e}"
        finally:
            ai_response_updated = True

class OverlayWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_width = parent.width()
        self.window_height = parent.height()
        self.hud_pos = QtCore.QPoint(20, 20)
        self.scroll = 0
        self.max_scroll = 0
        self.scroll_step = 30
        self.last_text = ""
        self.all_lines = []
        self.line_spacing = 0

        # Font
        self.font = QtGui.QFont("Arial", 12)
        self.fm = QtGui.QFontMetrics(self.font)

        # Timer to trigger update
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

    def update_frame(self):
        global ai_response, ai_response_updated

        self.scroll = min(self.scroll, self.max_scroll)

        x, margin = self.hud_pos.x(), 20
        max_w = self.window_width - x - margin

        self.all_lines = []
        title = "AI Assistant HUD" + (" (Processing...)" if ai_processing else "")
        ts    = time.strftime("%Y-%m-%d %H:%M:%S")

        self.all_lines.extend(self.wrap(title, max_w))
        self.all_lines.extend(self.wrap(ts,    max_w))

        for raw in ai_response.split("\n"):
            self.all_lines.extend(self.wrap(raw, max_w))

        self.line_spacing = int(self.fm.height() * 1.2)
        total_h = len(self.all_lines) * self.line_spacing
        self.max_scroll = max(0,
            total_h - (self.window_height - self.hud_pos.y() - margin)
        )

        ai_response_updated = False

        self.update()


    def wrap(self, text, max_width):
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            test = cur + (" " if cur else "") + w
            if self.fm.horizontalAdvance(test) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur or not lines:
            lines.append(cur)
        return lines

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # translucent navy background
        bg = QtGui.QColor(0, 0, 128, int(255 * 0.2))
        painter.fillRect(self.rect(), bg)

        painter.setFont(self.font)
        painter.setPen(QtGui.QColor(0, 0, 0))

        x, y = self.hud_pos.x(), self.hud_pos.y() - self.scroll
        for line in self.all_lines:
            if y + self.line_spacing < 0:
                y += self.line_spacing
                continue
            if y > self.window_height:
                break
            painter.drawText(x, y + self.fm.ascent(), line)
            y += self.line_spacing

        painter.end()

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        if delta > 0:
            self.scroll = max(0, self.scroll - self.scroll_step)
        else:
            self.scroll = min(self.max_scroll, self.scroll + self.scroll_step)
        self.update()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Overlay")
        self.setFixedSize(500, 700)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.overlay = OverlayWidget(self)
        self.setCentralWidget(self.overlay)
        self._drag_pos = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def keyPressEvent(self, e):
        global screenshot_path
        k = e.text().lower()
        if k == 'q':
            QtWidgets.QApplication.quit()
        elif k == 'i':
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"{timestamp}.png")
            ImageGrab.grab().save(path)
            print("Saved screenshot to", path)
            screenshot_path = path
            if not ai_processing:
                threading.Thread(
                    target=AIProcessor.process_image, args=(path,), daemon=True
                ).start()
        else:
            super().keyPressEvent(e)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

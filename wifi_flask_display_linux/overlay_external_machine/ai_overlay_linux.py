#!/usr/bin/env python3
import os
# Force the use of the X11 backend (XCB)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys
import cv2
import time
import threading
import openai
import base64
from PIL import Image
from mimetypes import guess_type
from PyQt5 import QtWidgets, QtGui, QtCore
from flask import Flask, request

# Set your OpenAI API
openai.api_key = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXX"

# Global variables for sharing data between Flask and PyQt
ai_response = "Waiting for input..."
ai_response_updated = False
ai_code_response = ""
ai_processing = False
screenshot_path = "/tmp/incoming.png"

class AIProcessor:
    @staticmethod
    def encode_image_to_base64(image_path):
        """Reads an image file and returns a base64-encoded string."""
        with open(image_path, "rb") as img_file:
            b64_bytes = base64.b64encode(img_file.read())
        return b64_bytes.decode('utf-8')

    @staticmethod
    def local_image_to_data_url(image_path):
        """Encodes a local image file as a data URL."""
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        image_b64 = AIProcessor.encode_image_to_base64(image_path)
        return f"data:{mime_type};base64,{image_b64}"

    @staticmethod
    def get_text_from_image(image_path, vision_prompt):
        """Uses ChatGPT API to extract text from an image."""
        global ai_response, ai_processing
        
        # Update status
        ai_response = "Processing image with ChatGPT API..."
        
        # Encode image as base64 data URI.
        data_uri = AIProcessor.local_image_to_data_url(image_path)

        # Build messages:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in extracting text from images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}}
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Pick your choice of ChatGPT model.
                messages=messages,
                max_tokens=5000
            )
            # Extract reply text.
            extracted_text = response['choices'][0]['message']['content']
            return extracted_text.strip()
        except Exception as e:
            print("Error during GPT-4o call:", e)
            return f"Error: {e}"

    @staticmethod
    def get_code_from_gpt3(extracted_text, code_prompt_instruction):
        """Uses GPT-3.5-turbo to generate code (or other answers)."""
        global ai_response
        
        # Update status
        ai_response = "Generating code solution..."
        
        # Modify prompt as needed
        prompt = (
            f"{code_prompt_instruction}\n\n"
            f"IMPORTANT: If this is code, respond using the SAME PROGRAMMING LANGUAGE "
            f"as shown in the problem. If you see C++ code, respond with C++ code. "
            f"If you see Python code, respond with Python code, etc.\n\n"
            f"Extracted Problem:\n{extracted_text}"
        )

        try:
            response = openai.ChatCompletion.create(
                model="o1-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            code_answer = response['choices'][0]['message']['content']
            return code_answer.strip()
        except Exception as e:
            print("Error during AI code generation:", e)
            return f"Error during AI code generation: {e}"

    @staticmethod
    def process_image(image_path):
        """Process the image with AI and return the results."""
        global ai_response, ai_code_response, ai_response_updated, ai_processing
        
        ai_processing = True
        ai_response = "AI processing started..."
        ai_response_updated = True
        
        try:
            img = Image.open(image_path)
            print(f"Image {image_path} opened successfully. Size: {img.size}")
            
            vision_prompt = (
                "Extract the **entire** visible questions, instructions, AND code, extract ALL words, this including function definitions, from top to bottom. Do not miss any lines on the right side, even if they’re partially visible."
            )
            extracted_text = AIProcessor.get_text_from_image(image_path, vision_prompt)
            print("Extracted Text from GPT-4 Vision:")
            print(extracted_text)

            code_prompt_instruction = (
                "Analyze the extracted text carefully to determine what programming language is being used. "
                "If this is a coding problem, give a complete code solution IN THE SAME PROGRAMMING LANGUAGE as the problem. "
                "Pay special attention to language-specific syntax like '#include', 'vector<>', 'cout', etc. for C++, "
                "or 'import', 'def', 'print()' for Python. "
                "Match the programming language exactly - if you see C++ code, respond with C++ code. "
                "Otherwise, provide the best short answer or analysis."
            )
            code_answer = AIProcessor.get_code_from_gpt3(extracted_text, code_prompt_instruction)
            print("\nAI Answer:")
            print(code_answer)
            
            ai_response = f"Prompt Extracted:\n\n{extracted_text}\n\nAnswer:\n\n{code_answer}"
            ai_response_updated = True
            
        except Exception as e:
            ai_response = f"Error processing image: {str(e)}"
            ai_response_updated = True
            print(f"Error processing image: {e}")
        
        finally:
            ai_processing = False
            return ai_response

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_image():
    global screenshot_path, ai_response, ai_response_updated, ai_processing
    
    if 'image' not in request.files:
        return "No image received", 400
    
    image = Image.open(request.files['image'].stream)
    image.save(screenshot_path)
    print("Screenshot received and saved to", screenshot_path)
    
    if not ai_processing:
        threading.Thread(target=AIProcessor.process_image, args=(screenshot_path,), daemon=True).start()
    
    return "Image received and processing started", 200

def start_flask():
    app.run(host="0.0.0.0", port=5000)

class VideoWidget(QtWidgets.QLabel):
    def __init__(self, video_source, parent=None):
        super(VideoWidget, self).__init__(parent)
        
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"Error opening live video device {video_source}")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set input format
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('R', 'G', 'B', '3'))
        
        # Get capture properties to verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = (
            chr(fourcc_code & 0xFF)
            + chr((fourcc_code >> 8) & 0xFF)
            + chr((fourcc_code >> 16) & 0xFF)
            + chr((fourcc_code >> 24) & 0xFF)
        )
        
        print(f"Camera initialized with: {actual_width}x{actual_height} @ {actual_fps}fps, format: {fourcc_str}")
        
        # Set display window
        self.window_width = 1920
        self.window_height = 1080
        
        # HUD settings
        self.hud_scale = 1.0
        self.hud_position_y = 50
        self.hud_position_x = 50
        self.hud_visible = True
        
        # HUD width
        self.hud_width = 840
        
        self.hud_background_opacity = 0.10
        
        # Color scheme
        self.hud_font = cv2.FONT_HERSHEY_SIMPLEX
        self.hud_color = (180, 230, 180)
        self.code_color = (230, 230, 180)
        self.comment_color = (180, 230, 230)
        self.keyword_color = (230, 200, 180)
        self.timestamp_color = (230, 230, 230)
        self.background_color = (0, 0, 0)
        
        # Timestamp position
        self.timestamp_y_offset = int(50 * self.hud_scale)
        
        # Frame counters
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_check = time.time()
        self.fps = 0
        
        # Timer to update video frames
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # 60 FPS
        
        # Window-drag variables
        self.drag_position = None
        self.setMouseTracking(True)
        
        # AI response display and scrolling
        self.last_ai_response = ""
        self.last_check_time = time.time()
        self.dragging_hud = False
        self.drag_start_pos = None
        self.hud_scroll_offset = 0
        self.max_scroll_offset = 0
        self.scroll_step = 30

    def mousePressEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        
        hud_x = self.hud_position_x
        hud_y = self.hud_position_y
        
        hud_bounding_height = 550
        
        if (hud_x <= x <= hud_x + self.hud_width and
            hud_y <= y <= hud_y + hud_bounding_height):
            self.dragging_hud = True
            self.drag_start_pos = (x - self.hud_position_x, y - self.hud_position_y)
            event.accept()
            self.setFocus()
        else:
            self.parent().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_hud and self.drag_start_pos is not None:
            new_x = event.pos().x() - self.drag_start_pos[0]
            new_y = event.pos().y() - self.drag_start_pos[1]
            
            new_x = max(10, min(new_x, self.window_width - self.hud_width - 10))
            new_y = max(10, min(new_y, self.window_height - 200))
            
            self.hud_position_x = new_x
            self.hud_position_y = new_y
            event.accept()
        else:
            self.parent().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging_hud:
            self.dragging_hud = False
            event.accept()
        else:
            self.parent().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        x, y = event.pos().x(), event.pos().y()

        hud_x = self.hud_position_x
        hud_y = self.hud_position_y
        hud_width = self.hud_width
        
        max_visible_height = 500
        bounding_hud_height = max_visible_height + 50
        
        if (hud_x <= x <= hud_x + hud_width and
            hud_y <= y <= hud_y + bounding_hud_height):
            
            delta = event.angleDelta().y()
            
            line_spacing = 1.15
            font_scale = 0.8 * self.hud_scale
            line_height = int(30 * font_scale * line_spacing)
            
            lines = self.last_ai_response.count('\n') + 3
            content_height = lines * line_height
            visible_height = max_visible_height
            
            self.max_scroll_offset = max(0, content_height - visible_height)
            
            if delta > 0:  # Scroll up
                self.hud_scroll_offset = max(0, self.hud_scroll_offset - self.scroll_step)
            else:  # Scroll down
                self.hud_scroll_offset = min(self.max_scroll_offset, self.hud_scroll_offset + self.scroll_step)
            
            print(
                f"Scroll: {self.hud_scroll_offset}/{self.max_scroll_offset}, "
                f"Content height: {content_height}, Visible: {visible_height}"
            )
            event.accept()
        else:
            event.ignore()

    def draw_multiline_text(self, img, text, start_position, font, font_scale, color, thickness):
        """
        Draw multiline text with syntax highlighting for code, 
        with a slight line spacing increase. 
        This version processes code-fence lines even if they're offscreen,
        ensuring code-mode toggles remain correct when scrolling.
        """
        line_spacing = 1.15
        line_height = int(30 * font_scale * line_spacing)
        
        max_width = self.hud_width - 10
        lines = text.split('\n')
        
        total_content_height = len(lines) * line_height
        visible_height = 500
        self.max_scroll_offset = max(0, total_content_height - visible_height)
        
        clip_start_y = start_position[1] - 30
        clip_end_y = start_position[1] + visible_height
        
        rect_start = (start_position[0] - 5, clip_start_y)
        rect_end = (start_position[0] + max_width, clip_end_y)
        
        overlay = img.copy()
        cv2.rectangle(overlay, rect_start, rect_end, self.background_color, -1)
        cv2.addWeighted(overlay, self.hud_background_opacity, img, 1 - self.hud_background_opacity, 0, img)
        
        y_pos = start_position[1] - self.hud_scroll_offset
        code_mode = False
        
        for line in lines:
            if '```' in line:
                code_mode = not code_mode
                y_pos += line_height
                continue
            
            if y_pos + line_height < clip_start_y:
                y_pos += line_height
                continue
            
            if y_pos > clip_end_y:
                break
            
            if code_mode:
                if "//" in line or "#" in line:
                    text_color = self.comment_color
                elif any(keyword in line for keyword in ["def ", "class ", "function", "for ", "while ", "if ", "else", "return"]):
                    text_color = self.keyword_color
                else:
                    text_color = self.code_color
            else:
                text_color = self.hud_color
            
            # Wrapping
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            if text_size[0] > max_width:
                words = line.split(' ')
                current_line = ""
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if cv2.getTextSize(test_line, font, font_scale, thickness)[0][0] <= max_width:
                        current_line = test_line
                    else:
                        if clip_start_y <= y_pos <= clip_end_y:
                            cv2.putText(img, current_line, (start_position[0], y_pos), font, font_scale, text_color, thickness)
                        y_pos += line_height
                        current_line = word
                
                if current_line:
                    if clip_start_y <= y_pos <= clip_end_y:
                        cv2.putText(img, current_line, (start_position[0], y_pos), font, font_scale, text_color, thickness)
                    y_pos += line_height
            else:
                if clip_start_y <= y_pos <= clip_end_y:
                    cv2.putText(img, line, (start_position[0], y_pos), font, font_scale, text_color, thickness)
                y_pos += line_height
        
        # Scroll indicator
        if self.max_scroll_offset > 0:
            scroll_percent = min(100, int((self.hud_scroll_offset / self.max_scroll_offset) * 100))
            scroll_indicator = f"Scroll: {scroll_percent}%"
            cv2.putText(
                img,
                scroll_indicator,
                (start_position[0] + max_width - 120, clip_end_y - 10),
                font,
                font_scale * 0.7,
                (200, 200, 255),
                thickness
            )
        return y_pos

    def update_frame(self):
        global ai_response, ai_response_updated, ai_processing
        
        frame_start = time.time()
        
        if not self.cap.grab():
            print("Frame capture failed. Retrying...")
            if self.frame_count == 0:
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('R', 'G', 'B', '3'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return
            
        ret, frame = self.cap.retrieve()
        if not ret:
            return

        self.frame_count += 1
        
        if self.frame_count % 60 == 0:
            now = time.time()
            elapsed = now - self.last_fps_check
            self.fps = 60 / elapsed if elapsed > 0 else 0
            print(f"Processing FPS: {self.fps:.2f}")
            self.last_fps_check = now

        if frame.shape[0] != self.window_height or frame.shape[1] != self.window_width:
            frame = cv2.resize(frame, (self.window_width, self.window_height),
                               interpolation=cv2.INTER_NEAREST)
        
        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, g, b])
        
        if ai_response_updated or self.last_ai_response != ai_response:
            self.last_ai_response = ai_response
            ai_response_updated = False
            self.hud_scroll_offset = 0
        
        if self.hud_visible:
            font_scale = 0.6 * self.hud_scale
            line_thickness = max(1, int(1 * self.hud_scale))
            y_pos = self.hud_position_y
            x_pos = self.hud_position_x
            
            title = "AI Assistant HUD"
            if ai_processing:
                title += " (Processing...)"
            title_height = self.add_text_with_background(
                frame, title, (x_pos, y_pos),
                self.hud_font, font_scale * 1.2,
                self.hud_color, line_thickness
            )
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_height = self.add_text_with_background(
                frame, timestamp, (x_pos, y_pos + title_height),
                self.hud_font, font_scale * 0.8,
                self.timestamp_color, line_thickness
            )
            
            if self.max_scroll_offset > 0:
                scroll_pct = int((self.hud_scroll_offset / self.max_scroll_offset) * 100) if self.max_scroll_offset else 0
                scroll_text = f"[Use mouse wheel to scroll - {scroll_pct}%]"
                self.add_text_with_background(
                    frame, scroll_text,
                    (x_pos + self.hud_width - 300, y_pos),
                    self.hud_font, font_scale * 0.7,
                    (180, 180, 230), line_thickness
                )
            
            content_y = y_pos + title_height + timestamp_height + 10
            self.draw_multiline_text(
                frame, self.last_ai_response,
                (x_pos, content_y),
                self.hud_font,
                font_scale * 0.8,
                self.hud_color,
                line_thickness
            )
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        self.qimage = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.pixmap)
        
        frame_time = time.time() - frame_start
        next_delay = max(1, int(16 - frame_time * 1000))
        self.timer.setInterval(next_delay)

    def add_text_with_background(self, img, text, position, font, font_scale, color, thickness):
        """Add text with a semi-transparent background for better readability."""
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        rect_start = (position[0] - 5, position[1] - text_size[1] - 5)
        rect_end = (position[0] + text_size[0] + 5, position[1] + 5)
        
        overlay = img.copy()
        cv2.rectangle(overlay, rect_start, rect_end, self.background_color, -1)
        
        cv2.addWeighted(overlay, self.hud_background_opacity, img, 1 - self.hud_background_opacity, 0, img)
        
        cv2.putText(img, text, position, font, font_scale, color, thickness)
        
        return text_size[1] + 10

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, video_source):
        super(MainWindow, self).__init__()
        self.setWindowTitle("AI Overlay Output")
        
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        x_position = (screen.width() - 1920) // 2
        y_position = (screen.height() - 1080) // 2
        self.setGeometry(x_position, y_position, 1920, 1080)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()
        
        self.video_widget = VideoWidget(video_source)
        self.setCentralWidget(self.video_widget)
        
        self.setWindowOpacity(1.0)
        self.setMouseTracking(True)
        self.drag_position = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton and self.drag_position is not None:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def keyPressEvent(self, event):
        if event.text().lower() == 'q':
            self.video_widget.cap.release()
            QtWidgets.QApplication.quit()
        elif event.text() == '=' or event.key() == QtCore.Qt.Key_Plus:
            self.video_widget.hud_scale = min(2.0, self.video_widget.hud_scale + 0.1)
            self.video_widget.timestamp_y_offset = int(50 * self.video_widget.hud_scale)
            print(f"HUD Scale: {self.video_widget.hud_scale:.1f}")
        elif event.text() == '-' or event.key() == QtCore.Qt.Key_Minus:
            self.video_widget.hud_scale = max(0.5, self.video_widget.hud_scale - 0.1)
            self.video_widget.timestamp_y_offset = int(50 * self.video_widget.hud_scale)
            print(f"HUD Scale: {self.video_widget.hud_scale:.1f}")
        elif event.text() == '[':
            self.video_widget.hud_position_y = max(10, self.video_widget.hud_position_y - 10)
            print(f"HUD Position Y: {self.video_widget.hud_position_y}")
        elif event.text() == ']':
            self.video_widget.hud_position_y = min(
                self.video_widget.window_height - 100,
                self.video_widget.hud_position_y + 10
            )
            print(f"HUD Position Y: {self.video_widget.hud_position_y}")
        elif event.text() == '{':
            self.video_widget.hud_position_x = max(10, self.video_widget.hud_position_x - 10)
            print(f"HUD Position X: {self.video_widget.hud_position_x}")
        elif event.text() == '}':
            self.video_widget.hud_position_x = min(
                self.video_widget.window_width - 100,
                self.video_widget.hud_position_x + 10
            )
            print(f"HUD Position X: {self.video_widget.hud_position_x}")
        elif event.key() == QtCore.Qt.Key_Up and event.modifiers() & QtCore.Qt.ShiftModifier:
            self.video_widget.hud_visible = not self.video_widget.hud_visible
            print(f"HUD Visible: {self.video_widget.hud_visible}")
        elif event.text().lower() == 'w':
            self.video_widget.hud_width = min(1800, self.video_widget.hud_width + 50)
            print(f"HUD Width: {self.video_widget.hud_width}")
        elif event.text().lower() == 'n':
            self.video_widget.hud_width = max(400, self.video_widget.hud_width - 50)
            print(f"HUD Width: {self.video_widget.hud_width}")
        elif event.text().lower() == 'p':
            print("Manual AI processing triggered")
            if not ai_processing and os.path.exists(screenshot_path):
                threading.Thread(target=AIProcessor.process_image, args=(screenshot_path,), daemon=True).start()
        elif event.key() == QtCore.Qt.Key_Up and not event.modifiers() & QtCore.Qt.ShiftModifier:
            self.video_widget.hud_scroll_offset = max(
                0, self.video_widget.hud_scroll_offset - self.video_widget.scroll_step
            )
            print(f"Scroll offset: {self.video_widget.hud_scroll_offset}")
        elif event.key() == QtCore.Qt.Key_Down:
            max_offset = self.video_widget.max_scroll_offset
            self.video_widget.hud_scroll_offset = min(
                max_offset, self.video_widget.hud_scroll_offset + self.video_widget.scroll_step
            )
            print(f"Scroll offset: {self.video_widget.hud_scroll_offset}")
        elif event.key() == QtCore.Qt.Key_PageUp:
            self.video_widget.hud_scroll_offset = max(
                0, self.video_widget.hud_scroll_offset - self.video_widget.scroll_step * 5
            )
            print(f"Scroll offset: {self.video_widget.hud_scroll_offset}")
        elif event.key() == QtCore.Qt.Key_PageDown:
            max_offset = self.video_widget.max_scroll_offset
            self.video_widget.hud_scroll_offset = min(
                max_offset, self.video_widget.hud_scroll_offset + self.video_widget.scroll_step * 5
            )
            print(f"Scroll offset: {self.video_widget.hud_scroll_offset}")
        elif event.key() == QtCore.Qt.Key_Home:
            self.video_widget.hud_scroll_offset = 0
            print("Scrolled to top")
        elif event.key() == QtCore.Qt.Key_End:
            self.video_widget.hud_scroll_offset = self.video_widget.max_scroll_offset
            print("Scrolled to bottom")
        else:
            super(MainWindow, self).keyPressEvent(event)

def main():
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    print("Flask server started on port 5000")
    
    app = QtWidgets.QApplication(sys.argv)
    
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
    else:
        video_source = 0  # Default camera
    
    print(f"Opening video source: {video_source}")
    window = MainWindow(video_source)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

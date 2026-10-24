"""
High-Performance Motion Server for Code Eye.

Uses the advanced computer vision engine from motion_detector.py:
- Threaded non-blocking webcam grabber (zero hardware buffer delay)
- Advanced background subtraction (MOG2) and contour cluster analysis
- Velocity vector calculation and adaptive saccadic tracking
- Real-time MJPEG live stream with cybernetic reticle HUD
- Sub-millisecond JSON state API for high-refresh visualizer

Usage:
  python motion_server.py [--camera 0] [--port 8000]
"""

import argparse
import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from motion_detector import AdvancedMotionDetector, MotionState, ThreadedCamera

HERE = Path(__file__).resolve().parent

# Global shared state
state = asdict(MotionState())
lock = threading.Lock()

# Latest preview JPEG
latest_jpeg = None
jpeg_lock = threading.Lock()


def camera_loop(camera_index: int, min_area: int, history: int, var_threshold: int, blink_cooldown: float):
    global latest_jpeg, state

    cam = ThreadedCamera(camera_index)
    if not cam.is_opened():
        print(f"Could not open camera {camera_index}.")
        return

    detector = AdvancedMotionDetector(
        min_area=min_area,
        history=history,
        var_threshold=var_threshold,
        blink_cooldown=blink_cooldown,
    )

    print("Motion engine active: MOG2 background modeling + adaptive saccadic tracking.")

    while cam.running:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        motion_st, jpeg_bytes = detector.process(frame)

        with lock:
            state = asdict(motion_st)

        if jpeg_bytes:
            with jpeg_lock:
                latest_jpeg = jpeg_bytes

        # Lightweight yield to allow smooth concurrency
        time.sleep(0.01)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/motion":
            with lock:
                body = json.dumps(state).encode()
            self._send(200, body, "application/json")
        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with jpeg_lock:
                        frame_bytes = latest_jpeg
                    if frame_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n")
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS preview stream
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        elif path in ("/camera_frame.jpg", "/snapshot"):
            with jpeg_lock:
                frame_bytes = latest_jpeg
            if frame_bytes:
                self._send(200, frame_bytes, "image/jpeg")
            else:
                self._send(503, b"No frame available yet", "text/plain")
        elif path in ("/", "/code_eye.html"):
            page = HERE / "code_eye.html"
            if page.exists():
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"code_eye.html not found next to this script", "text/plain")
        else:
            self._send(404, b"Not found", "text/plain")

    def log_message(self, *args):
        pass  # Keep console clean


def main():
    ap = argparse.ArgumentParser(description="High-performance webcam motion detector for Code Eye")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--port", type=int, default=8000, help="server port (default 8000)")
    ap.add_argument("--min-area", type=int, default=80, help="min motion area threshold (default 80)")
    ap.add_argument("--history", type=int, default=150, help="MOG2 background history frames (default 150)")
    ap.add_argument("--var-threshold", type=int, default=25, help="MOG2 variance threshold (default 25)")
    ap.add_argument("--blink-cooldown", type=float, default=0.7, help="min seconds between blinks")
    args = ap.parse_args()

    threading.Thread(
        target=camera_loop,
        args=(args.camera, args.min_area, args.history, args.var_threshold, args.blink_cooldown),
        daemon=True,
    ).start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Server online at http://localhost:{args.port}. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
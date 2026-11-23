import cv2
import time
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from modules.face_detection import FaceROI
from modules.rppg_processing import RPPGEnsemble
from modules.heart_rate import HeartRateEstimator
from modules.hrv_analysis import HRVAnalyzer
from modules.stress_analysis import StressAnalyzer
from ui.dashboard import Dashboard


class RPPGMonitor:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.cap = None

        self.face = FaceROI()
        self.rppg = RPPGEnsemble()
        self.hr_estimator = HeartRateEstimator()
        self.hrv_estimator = HRVAnalyzer()
        self.stress_estimator = StressAnalyzer()

        self.times = []
        self.values = []
        self.hr_history = []
        self.hrv_history = []

        self.bpm = 0.0
        self.hrv = 0.0
        self.confidence = 0.0
        self.stress = "Waiting"
        self.fft_bpm = 0.0
        self.peak_bpm = 0.0

        self.dashboard = Dashboard(root, self)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self.update)

    def start(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Webcam could not be opened.\n"
                                 "Check camera permissions and close other camera apps.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.times.clear()
        self.values.clear()
        self.hr_history.clear()
        self.hrv_history.clear()

        self.bpm = 0
        self.hrv = 0
        self.confidence = 0
        self.stress = "Calibrating"
        self.fft_bpm = 0.0
        self.peak_bpm = 0.0

        self.rppg.reset()
        self.hr_estimator.reset()
        self.hrv_estimator.reset()
        self.dashboard.set_live(True)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.dashboard.set_live(False)

    def update(self):
        if self.running and self.cap is not None:
            ok, frame = self.cap.read()

            if ok:
                # OpenCV captures BGR. Convert ONLY for display/processing.
                # This prevents the blue-tinted camera problem.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                roi = self.face.get_measurement(frame_rgb)

                display = frame.copy()  # Keep the camera in natural BGR color.
                if roi is not None:
                    x1, y1, x2, y2 = roi["box"]
                    cv2.rectangle(display, (x1, y1), (x2, y2),
                                  (255, 255, 255), 2)
                    cv2.putText(display, "rPPG REGION",
                                (x1, max(25, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)

                    sample = roi["mean_rgb"]
                    # Reject frames with excessive motion or poor skin ROI quality.
                    result = None if roi["motion"] > 0.012 or roi["quality"] < 0.42 else self.rppg.update(sample, time.time())

                    if result is not None:
                        self.times.append(result[0])
                        self.values.append(result[1])

                    # Keep approximately the most recent 40 seconds.
                    if len(self.times) > 1200:
                        self.times = self.times[-1200:]
                        self.values = self.values[-1200:]

                    if len(self.times) >= 150:
                        hr_result = self.hr_estimator.estimate(
                            np.asarray(self.times),
                            np.asarray(self.values)
                        )

                        if hr_result["bpm"] > 0:
                            self.bpm = hr_result["bpm"]
                            self.confidence = hr_result["confidence"]
                            self.fft_bpm = hr_result.get("fft_bpm", 0.0)
                            self.peak_bpm = hr_result.get("peak_bpm", 0.0)

                            rr = self.hr_estimator.get_rr_intervals()
                            self.hrv = self.hrv_estimator.rmssd(rr)

                            self.stress = self.stress_estimator.classify(
                                self.bpm, self.hrv, self.confidence
                            )

                            self.hr_history.append(self.bpm)
                            self.hrv_history.append(self.hrv)

                self.dashboard.update_camera(display)
                self.dashboard.update_data(
                    self.bpm, self.hrv, self.confidence, self.stress,
                    self.times, self.values,
                    self.hr_history, self.hrv_history,
                    self.fft_bpm, self.peak_bpm
                )

        self.root.after(30, self.update)

    def save_session(self):
        if len(self.times) < 10:
            messagebox.showinfo("No Session",
                                "Start monitoring and collect some data first.")
            return

        os.makedirs("data/sessions", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"data/sessions/session_{stamp}.csv"

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "rppg_signal"])
            writer.writerows(zip(self.times, self.values))

        messagebox.showinfo("Session Saved", f"Saved to:\n{filename}")

    def close(self):
        self.stop()
        self.face.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("rPPG Vital Signs Monitor — Accuracy Mode")
    root.geometry("1450x900")
    root.minsize(1150, 760)
    RPPGMonitor(root)
    root.mainloop()

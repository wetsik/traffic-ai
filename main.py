import cv2
from ultralytics import YOLO
import yt_dlp

model = YOLO("yolo11n.pt")

youtube_url = "https://www.youtube.com/watch?v=8JCk5M_xrBs"

ydl_opts = {
    "format": "best[ext=mp4]",
    "quiet": True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(youtube_url, download=False)
    video_url = info["url"]

cap = cv2.VideoCapture(video_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

window_name = "YouTube Stream YOLO"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1920, 1080)

while True:
    for _ in range(2):
        cap.grab()

    ret, frame = cap.read()

    if not ret:
        print("Video tugadi yoki frame olinmadi")
        break

    # YOLO uchun kichikroq frame
    detect_frame = cv2.resize(frame, (1920, 1080))

    results = model.predict(
        detect_frame,
        verbose=False,
        classes=[0],
        imgsz=640,
        conf=0.4
    )

    annotated_small = results[0].plot()

    # Ko‘rsatish uchun kattalashtiramiz
    annotated_big = cv2.resize(annotated_small, (1280, 720))

    cv2.imshow(window_name, annotated_big)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
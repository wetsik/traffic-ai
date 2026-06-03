from ultralytics import YOLO

model = YOLO('runs/detect/people_vehicle_detector-5/weights/best.pt')
results = model('video.mp4', show=True, save=True, conf=0.4)
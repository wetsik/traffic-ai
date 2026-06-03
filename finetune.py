from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data='yolo_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        name='people_vehicle_detector'
    )
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model using our vehicle damage dataset
results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    workers=2,
    project="models",
    name="vehicle_damage"
)

print("Vehicle damage model training completed!")
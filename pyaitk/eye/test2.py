from eye import launch_gui, DetectionConfig

config = DetectionConfig(
    model_name="yolov8n.pt",
    confidence_threshold=0.6,
    target_fps=25,
    show_heatmap=True,
    filter_classes=["person", "car"],
)
launch_gui(config)
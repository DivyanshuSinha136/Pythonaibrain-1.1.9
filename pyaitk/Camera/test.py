from mainCamera import Camera

with Camera.open(device=0, output_dir="./out") as cam:
    cam.run()
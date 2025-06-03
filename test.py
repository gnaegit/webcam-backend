from src.camera import CameraIDS

try:
    descriptors = CameraIDS.list_devices()
    print(descriptors)
except Exception as e:
    str(e)




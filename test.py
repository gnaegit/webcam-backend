import os
from src.camera import CameraIDS, CameraConfigurationError
import cv2
import ids_peak.ids_peak as ids_peak

def capture_and_save_image(camera, filename):
    """
    Capture a single image and save it to disk.
    
    Args:
        camera (CameraIDS): Initialized camera object.
        filename (str): Path to save the image.
    
    Returns:
        bool: True if capture and save succeeded, False otherwise.
    """
    try:
        camera.start_acquisition()
        image = camera.capture()  # Use correct capture method
        if image is not None:
            np_arr = image.get_numpy_3D()
            success, jpeg_data = cv2.imencode('.jpg', np_arr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            cv2.imwrite(filename, jpeg_data)
            print(f"Image saved to {filename}")
            return True
        else:
            print("Failed to capture image: No image returned.")
            return False
    except Exception as e:
        print(f"Error capturing image: {e}")
        return False
    finally:
        camera.stop_acquisition()

def main():
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    try:
        # Initialize camera
        camera = CameraIDS(id_device=0)
        print(f"Camera initialized: {camera}")
        print(camera.nodemap.FindNode("BinningHorizontal").Minimum())
        print(camera.nodemap.FindNode("BinningHorizontal").Maximum())

        allEntries = camera.nodemap.FindNode("BinningSelector").Entries()
        availableEntries = []
        for entry in allEntries:
            if (entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                    and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented):
                availableEntries.append(entry.SymbolicValue())
        print(availableEntries)

        print(camera.nodemap.FindNode("BinningHorizontalMode").CurrentEntry().SymbolicValue())
        # camera.nodemap.FindNode("BinningVerticalMode").SetCurrentEntry('Average')
        print(camera.nodemap.FindNode("BinningHorizontalMode").CurrentEntry().SymbolicValue())
        allEntries = camera.nodemap.FindNode("BinningHorizontalMode").Entries()
        availableEntries = []
        for entry in allEntries:
            if (entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                    and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented):
                availableEntries.append(entry.SymbolicValue())
        print(availableEntries)

        # Set 1x1 binning
        print("\nSetting 1x1 binning...")
        if not camera.set_binning(horizontal_bin_factor=1, vertical_bin_factor=1, mode="Average"):
            print("Failed to set 1x1 binning, skipping binned capture.")
        else:
            print(camera.nodemap.FindNode("BinningHorizontal").Value())
            print(camera.nodemap.FindNode("BinningVertical").Value())
            # Capture image with 1x1 binning
            print("Capturing image with 1x1 binning...")
            if not capture_and_save_image(camera, "output/binning_1x1.png"):
                print("Failed to capture image with 2x2 binning.")

        # Set 2x2 binning
        print("\nSetting 2x2 binning...")
        if not camera.set_binning(horizontal_bin_factor=2, vertical_bin_factor=2, mode="Average"):
            print("Failed to set 2x2 binning, skipping binned capture.")
        else:
            print(camera.nodemap.FindNode("BinningHorizontal").Value())
            print(camera.nodemap.FindNode("BinningVertical").Value())
            # Capture image with 2x2 binning
            print("Capturing image with 2x2 binning...")
            if not capture_and_save_image(camera, "output/binning_2x2.png"):
                print("Failed to capture image with 2x2 binning.")

    except CameraConfigurationError as e:
        print(f"Camera configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        if 'camera' in locals():
            camera.close()
            print("Camera closed.")

if __name__ == "__main__":
    main()
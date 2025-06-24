import io
import asyncio
from zipfile import ZipFile
import tempfile
import shutil
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from threading import Condition
from contextlib import asynccontextmanager
import numpy as np
import cv2
import os
import datetime
import time
from pydantic import BaseModel
from pathlib import Path
import logging
import zipstream
import signal
from starlette.websockets import WebSocketState, WebSocketDisconnect
from typing import Dict, List, cast

try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder, Quality
    from picamera2.outputs import FileOutput
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logging.warning("Picamera2 library not installed")

try:
    from src.camera import CameraIDS
    from src.auto_feature_manager import AutoFeatureManager
    CAMERAIDS_AVAILABLE = True
except ImportError:
    CAMERAIDS_AVAILABLE = False
    logging.warning("CameraIDS library not installed")

IMAGES_DIR = Path("images")

# Define GlobalCameraInfo type based on global_camera_info return structure
GlobalCameraInfo = Dict[str, str | int]

class IntervalRequest(BaseModel):
    camera_key: str
    interval: float

class CameraSelectionRequest(BaseModel):
    camera_key: str

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    async def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.frame is not None)
            return self.frame

class JpegStream:
    def __init__(self):
        self.cameras = {}  # Dict to store camera instances: {camera_key: {"camera": ..., "type": ..., "index": ..., ...}}
        self.active_preview = {}  # Per-camera preview status
        self.active_storage = {}  # Per-camera storage status
        self.connections = set()  # WebSocket connections
        self.preview_tasks = {}  # Per-camera preview tasks
        self.storage_tasks = {}  # Per-camera storage tasks
        self.current_folders = {}  # Per-camera storage folders
        self.outputs = {}  # Per-camera StreamingOutput
        self.save_intervals = {}  # Per-camera save intervals
        self.last_save_times = {}  # Per-camera last save time
        self.auto_feature_managers = {}  # Per-camera auto feature manager
        self.camera_status = {"picamera": [], "cameraids": []}  # List of available camera indices
        self.max_folder_size = 1_073_741_824  # 1GB
        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(IMAGES_DIR / "trigger", exist_ok=True)

    def get_available_disk_space(self) -> int:
        """Check available disk space in bytes."""
        stat = shutil.disk_usage(IMAGES_DIR)
        return stat.free

    def get_folder_size(self, folder_path: Path) -> int:
        """Calculate the total size of a folder in bytes."""
        total_size = 0
        for item in folder_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size

    def check_picamera_availability(self) -> List[GlobalCameraInfo]:
        """Check availability of all Raspberry Pi cameras using global_camera_info."""
        if not PICAMERA_AVAILABLE:
            return []

        @staticmethod
        def global_camera_info() -> List[GlobalCameraInfo]:
            """
            Return Id string and Model name for all attached cameras, one dict per camera.
            Ordered correctly by camera number. Also return the location and rotation
            of the camera when known, as these may help distinguish which is which.
            """
            def describe_camera(cam, num):
                info = {k.name: v for k, v in cam.properties.items() if k.name in ("Model", "Location", "Rotation")}
                info["Id"] = cam.id
                info["Num"] = num
                info["index"] = num  # Add index for consistency with CameraIDS
                info["display_name"] = f"Raspberry Pi Camera {num}"
                info["serial"] = cam.id
                info["label"] = f"PiCam {num}"
                return cast(GlobalCameraInfo, info)
            cameras = [describe_camera(cam, i) for i, cam in enumerate(Picamera2._cm.cms.cameras)]
            # Sort alphabetically so they are deterministic, but send USB cams to the back.
            return sorted(cameras, key=lambda cam: ("/usb" not in cam['Id'], cam['Id']), reverse=True)

        try:
            return global_camera_info()
        except Exception as e:
            logging.error(f"Failed to list Picamera2 devices: {str(e)}")
            return []

    def check_cameraids_availability(self):
        return CameraIDS.list_devices() if CAMERAIDS_AVAILABLE else []

    def _get_camera_key(self, camera_type: str, index: int) -> str:
        """Generate a unique key for a camera."""
        return f"{camera_type}_{index}"

    def set_camera(self, camera_type: str, index: int = 0):
        """Initialize a specific camera or return existing handle if already initialized."""
        camera_key = self._get_camera_key(camera_type, index)
        camera_type = camera_type.lower()
        if camera_type not in ["picamera", "cameraids"]:
            raise ValueError("Invalid camera type. Use 'picamera' or 'cameraids'")

        # Return existing camera handle if already initialized
        if camera_key in self.cameras:
            logging.info(f"Camera {camera_key} already initialized, returning existing handle")
            return self.cameras[camera_key]["camera"]

        # Proceed with initialization if camera is not already open
        if camera_type == "picamera":
            if not any(cam["index"] == index for cam in self.camera_status["picamera"]):
                raise RuntimeError(f"Raspberry Pi camera at index {index} is not available")
            try:
                camera = Picamera2(index)
                video_config = camera.create_video_configuration(main={"size": (1920, 1080)})
                camera.configure(video_config)
                self.cameras[camera_key] = {
                    "camera": camera,
                    "type": camera_type,
                    "index": index,
                    "output": StreamingOutput(),
                    "folder": None,
                    "last_save_time": 0,
                    "save_interval": 5,
                    "auto_feature_manager": None
                }
                logging.info(f"Initialized Picamera2 (key: {camera_key})")
                return camera
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Picamera2 at index {index}: {str(e)}")
        elif camera_type == "cameraids":
            if not any(cam["index"] == index for cam in self.camera_status["cameraids"]):
                raise RuntimeError(f"CameraIDS at index {index} is not available")
            try:
                camera = CameraIDS(id_device=index)
                camera.set_roi_max()
                self.cameras[camera_key] = {
                    "camera": camera,
                    "type": camera_type,
                    "index": index,
                    "output": StreamingOutput(),
                    "folder": None,
                    "last_save_time": 0,
                    "save_interval": 5,
                    "auto_feature_manager": AutoFeatureManager(camera)
                }
                self.cameras[camera_key]["auto_feature_manager"].auto_white_balance = 'on'
                self.cameras[camera_key]["auto_feature_manager"].auto_exposure = 'on'
                self.cameras[camera_key]["auto_feature_manager"].auto_gain = 'on'
                logging.info(f"Initialized CameraIDS (key: {camera_key})")
                return camera
            except Exception as e:
                raise RuntimeError(f"Failed to initialize CameraIDS: {str(e)}")

    def close_camera(self, camera_key: str):
        """Close a specific camera and clean up its resources."""
        if camera_key not in self.cameras:
            return
        camera_info = self.cameras[camera_key]
        camera = camera_info["camera"]
        try:
            if isinstance(camera, Picamera2):
                if camera.started:
                    camera.stop()
                camera.close()
                logging.info(f"Closed Picamera2 (key: {camera_key})")
            elif isinstance(camera, CameraIDS):
                if camera.acquiring:
                    camera.stop_capturing()
                    camera.stop_acquisition()
                camera.close()
                logging.info(f"Closed CameraIDS (key: {camera_key})")
        except Exception as e:
            logging.error(f"Error closing camera (key: {camera_key}): {e}")
        finally:
            del self.cameras[camera_key]
            self.active_preview.pop(camera_key, None)
            self.active_storage.pop(camera_key, None)
            self.preview_tasks.pop(camera_key, None)
            self.storage_tasks.pop(camera_key, None)
            self.current_folders.pop(camera_key, None)
            self.outputs.pop(camera_key, None)
            self.save_intervals.pop(camera_key, None)
            self.last_save_times.pop(camera_key, None)
            self.auto_feature_managers.pop(camera_key, None)

    def close(self):
        """Close all cameras."""
        for camera_key in list(self.cameras.keys()):
            self.close_camera(camera_key)

    def reinitialize_camera(self, camera_key: str):
        """Reinitialize a specific camera."""
        if camera_key not in self.cameras:
            return False
        camera_info = self.cameras[camera_key]
        try:
            self.close_camera(camera_key)
            self.set_camera(camera_info["type"], camera_info["index"])
            logging.info(f"Reinitialized camera (key: {camera_key})")
            return True
        except Exception as e:
            logging.error(f"Failed to reinitialize camera (key: {camera_key}): {str(e)}")
            self.close_camera(camera_key)
            return False

    async def set_interval(self, camera_key: str, interval: float):
        """Set save interval for a specific camera."""
        if camera_key not in self.cameras:
            raise ValueError(f"Camera {camera_key} not found")
        if interval <= 0:
            raise ValueError("Interval must be greater than zero.")
        self.cameras[camera_key]["save_interval"] = interval
        await self.notify_clients()

    async def capture_single_image(self, camera_key: str) -> str:
        """Capture a single image and save it to the trigger folder."""
        if camera_key not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_key} not found")

        camera_info = self.cameras[camera_key]
        camera = camera_info["camera"]
        camera_type = camera_info["type"]

        try:
            # Start camera if not already running
            if camera_type == "picamera":
                if not camera.started:
                    camera.start_recording(MJPEGEncoder(), FileOutput(camera_info["output"]), Quality.MEDIUM)
            elif camera_type == "cameraids":
                if not camera.acquiring:
                    camera.start_acquisition()
                if not camera.capturing_threaded:
                    camera.start_capturing(on_capture_callback=lambda img: self._capture_callback(img, camera_key))

            # Capture one frame
            jpeg_data = await camera_info["output"].read()
            img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)

            # Stop camera if it wasn't running before
            if camera_type == "picamera" and not self.active_preview.get(camera_key, False) and not self.active_storage.get(camera_key, False):
                camera.stop_recording()
            elif camera_type == "cameraids" and not self.active_preview.get(camera_key, False) and not self.active_storage.get(camera_key, False):
                try:
                    camera.stop_capturing()
                    camera.stop_acquisition()
                except Exception as e:
                    logging.error(f"Error stopping CameraIDS after single capture for {camera_key}: {str(e)}")

            # Save the image to the trigger folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = IMAGES_DIR / "trigger" / f"{camera_key}_image_{timestamp}.jpg"
            cv2.imwrite(str(image_path), img)

            # Notify clients
            await self.notify_clients(image_path=str(image_path.relative_to(IMAGES_DIR)), camera_key=camera_key)
            return str(image_path.relative_to(IMAGES_DIR))
        
        except Exception as e:
            logging.error(f"Error capturing single image for {camera_key}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")

    def _image_to_jpeg(self, image, camera_key: str):
        camera_info = self.cameras[camera_key]
        if camera_info["auto_feature_manager"]:
            for attempt in range(3):
                try:
                    camera_info["auto_feature_manager"](image)
                    break
                except Exception as e:
                    if "PEAK_AFL_STATUS_BUSY" in str(e):
                        logging.warning(f"PEAK_AFL_STATUS_BUSY for {camera_key}, attempt {attempt + 1}/3")
                        time.sleep(0.1)
                    else:
                        raise
        try:
            if isinstance(image, np.ndarray):
                np_arr = image
            else:
                np_arr = image.get_numpy_3D() if hasattr(image, 'get_numpy_3D') else np.array(image)
            success, jpeg_data = cv2.imencode('.jpg', np_arr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                raise RuntimeError("Failed to encode image to JPEG")
            return jpeg_data.tobytes()
        except Exception as e:
            logging.error(f"Error converting image to JPEG for {camera_key}: {e}")
            raise

    def _capture_callback(self, image, camera_key: str):
        """Capture callback for CameraIDS."""
        if image is not None:
            try:
                self.cameras[camera_key]["output"].write(self._image_to_jpeg(image, camera_key))
            except Exception as e:
                logging.error(f"Error in capture callback (key: {camera_key}): {e}")

    def create_new_folder(self, camera_key: str):
        """Create a new folder for a specific camera."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = Path(f"{camera_key}_{timestamp}")
        os.makedirs(IMAGES_DIR / folder, exist_ok=True)
        self.cameras[camera_key]["folder"] = folder

    async def store_images(self, camera_key: str):
        """Store images for a specific camera."""
        try:
            if camera_key not in self.cameras:
                raise RuntimeError(f"Camera {camera_key} not initialized")
            camera_info = self.cameras[camera_key]
            camera = camera_info["camera"]
            camera_type = camera_info["type"]

            if camera_type == "picamera":
                if not camera.started:
                    camera.start_recording(MJPEGEncoder(), FileOutput(camera_info["output"]), Quality.MEDIUM)
            elif camera_type == "cameraids":
                try:
                    if not camera.acquiring:
                        camera.start_acquisition()
                    if not camera.capturing_threaded:
                        camera.start_capturing(on_capture_callback=lambda img: self._capture_callback(img, camera_key))
                except Exception as e:
                    if "PEAK_RETURN_CODE_INVALID_HANDLE" in str(e):
                        logging.warning(f"Invalid handle for {camera_key}, attempting reinitialization")
                        if not self.reinitialize_camera(camera_key):
                            raise RuntimeError(f"Failed to reinitialize CameraIDS (key: {camera_key})")
                        camera = self.cameras[camera_key]["camera"]
                        camera.start_acquisition()
                        camera.start_capturing(on_capture_callback=lambda img: self._capture_callback(img, camera_key))

            self.create_new_folder(camera_key)

            while self.active_storage.get(camera_key, False):
                if self.get_available_disk_space() < 5 * 1024 * 1024 * 1024:
                    logging.warning(f"Disk space below 5GB for {camera_key}, stopping storage")
                    await self.stop_storage_task(camera_key)
                    await self.notify_clients(stop_reason=f"Insufficient disk space for {camera_key}")
                    return

                folder = self.cameras[camera_key]["folder"]
                if folder and self.get_folder_size(IMAGES_DIR / folder) > self.max_folder_size:
                    logging.info(f"Folder {folder} exceeded 1GB for {camera_key}, creating new folder")
                    self.create_new_folder(camera_key)
                    await self.notify_clients()

                jpeg_data = await camera_info["output"].read()
                img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)

                current_time = time.time()
                if current_time - camera_info["last_save_time"] >= camera_info["save_interval"]:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = IMAGES_DIR / folder / f"image_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), img)
                    camera_info["last_save_time"] = current_time
                    await self.notify_clients(image_path=str(image_path.relative_to(IMAGES_DIR)), camera_key=camera_key)

                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Storage task error for {camera_key}: {str(e)}")
            await self.notify_clients(stop_reason=f"Storage error for {camera_key}: {str(e)}")
        finally:
            if camera_key in self.cameras:
                camera = self.cameras[camera_key]["camera"]
                if camera_type == "picamera":
                    if camera.started and not self.active_preview.get(camera_key, False):
                        camera.stop_recording()
                elif camera_type == "cameraids":
                    if camera.acquiring and not self.active_preview.get(camera_key, False):
                        camera.stop_capturing()
                        camera.stop_acquisition()

    async def stream_preview(self, camera_key: str):
        """Stream preview for a specific camera."""
        try:
            logging.info(f"Starting preview stream for {camera_key}")
            if camera_key not in self.cameras:
                raise RuntimeError(f"Camera {camera_key} not initialized")
            camera_info = self.cameras[camera_key]
            camera = camera_info["camera"]
            camera_type = camera_info["type"]

            if camera_type == "picamera":
                if not camera.started:
                    camera.start_recording(MJPEGEncoder(), FileOutput(camera_info["output"]), Quality.MEDIUM)
            elif camera_type == "cameraids":
                for attempt in range(3):
                    try:
                        if not camera.acquiring:
                            logging.debug(f"Starting acquisition for {camera_key}, attempt {attempt + 1}")
                            camera.start_acquisition()
                        if not camera.capturing_threaded:
                            logging.debug(f"Starting capturing for {camera_key}, attempt {attempt + 1}")
                            camera.start_capturing(on_capture_callback=lambda img: self._capture_callback(img, camera_key))
                        break
                    except Exception as e:
                        if "PEAK_RETURN_CODE_INVALID_HANDLE" in str(e):
                            logging.warning(f"Invalid handle for {camera_key}, attempt {attempt + 1}/3, reinitializing")
                            if not self.reinitialize_camera(camera_key):
                                logging.error(f"Failed to reinitialize CameraIDS (key: {camera_key}) after {attempt + 1} attempts")
                                if attempt == 2:
                                    raise RuntimeError(f"Failed to reinitialize CameraIDS (key: {camera_key})")
                            camera = self.cameras[camera_key]["camera"]
                            await asyncio.sleep(1)
                        else:
                            logging.error(f"CameraIDS error for {camera_key}: {str(e)}")
                            raise

            while self.active_preview.get(camera_key, False) and self.connections:
                try:
                    jpeg_data = await camera_info["output"].read()
                    tasks = [
                        websocket.send_bytes(
                            b"--frame\r\nContent-Type: image/jpeg\r\nX-Camera-Key: " + camera_key.encode() + b"\r\n\r\n" + jpeg_data + b"\r\n"
                        )
                        for websocket in self.connections.copy()
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logging.error(f"Error sending frame for {camera_key}: {str(e)}")
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Preview task error for {camera_key}: {str(e)}")
            await self.notify_clients(stop_reason=f"Preview error for {camera_key}: {str(e)}")
        finally:
            logging.info(f"Stopping preview stream for {camera_key}")
            self.active_preview[camera_key] = False
            if camera_key in self.cameras:
                camera = self.cameras[camera_key]["camera"]
                camera_type = self.cameras[camera_key]["type"]
                if not self.active_storage.get(camera_key, False):
                    if camera_type == "picamera" and camera.started:
                        camera.stop_recording()
                    elif camera_type == "cameraids" and camera.acquiring:
                        try:
                            camera.stop_capturing()
                            camera.stop_acquisition()
                        except Exception as e:
                            logging.error(f"Error stopping CameraIDS for {camera_key}: {str(e)}")
            await self.notify_clients()

    async def start_storage_task(self, camera_key: str):
        """Start storage task for a specific camera."""
        if camera_key not in self.cameras:
            raise HTTPException(status_code=500, detail=f"Camera {camera_key} not initialized")
        if not self.active_storage.get(camera_key, False):
            self.active_storage[camera_key] = True
            self.storage_tasks[camera_key] = asyncio.create_task(self.store_images(camera_key))
            await self.notify_clients()

    async def stop_storage_task(self, camera_key: str):
        """Stop storage task for a specific camera."""
        if camera_key in self.active_storage and self.active_storage[camera_key]:
            self.active_storage[camera_key] = False
            if camera_key in self.storage_tasks:
                self.storage_tasks[camera_key].cancel()
                try:
                    await self.storage_tasks[camera_key]
                except asyncio.CancelledError:
                    pass
                self.storage_tasks.pop(camera_key, None)
            if camera_key in self.cameras:
                camera = self.cameras[camera_key]["camera"]
                camera_type = self.cameras[camera_key]["type"]
                if not self.active_preview.get(camera_key, False):
                    if camera_type == "picamera" and camera.started:
                        camera.stop()
                    elif camera_type == "cameraids" and camera.acquiring:
                        try:
                            camera.stop_capturing()
                            camera.stop_acquisition()
                        except Exception as e:
                            logging.error(f"Error stopping CameraIDS storage for {camera_key}: {str(e)}")
            await self.notify_clients()

    async def start_preview_task(self, camera_key: str):
        """Start preview task for a specific camera."""
        if camera_key not in self.cameras:
            raise HTTPException(status_code=500, detail=f"Camera {camera_key} not initialized")
        if not self.active_preview.get(camera_key, False) and self.connections:
            self.active_preview[camera_key] = True
            self.preview_tasks[camera_key] = asyncio.create_task(self.stream_preview(camera_key))
            await self.notify_clients()

    async def stop_preview_task(self, camera_key: str):
        """Stop preview task for a specific camera."""
        if camera_key in self.active_preview and self.active_preview[camera_key]:
            self.active_preview[camera_key] = False
            if camera_key in self.preview_tasks:
                self.preview_tasks[camera_key].cancel()
                try:
                    await self.preview_tasks[camera_key]
                except asyncio.CancelledError:
                    pass
                self.preview_tasks.pop(camera_key, None)
            if camera_key in self.cameras:
                camera = self.cameras[camera_key]["camera"]
                camera_type = self.cameras[camera_key]["type"]
                if not self.active_storage.get(camera_key, False):
                    if camera_type == "picamera" and camera.started:
                        camera.stop()
                    elif camera_type == "cameraids" and camera.acquiring:
                        try:
                            camera.stop_capturing()
                            camera.stop_acquisition()
                        except Exception as e:
                            logging.error(f"Error stopping CameraIDS preview for {camera_key}: {str(e)}")
            await self.notify_clients()

    async def notify_clients(self, image_path: str = None, stop_reason: str = None, camera_key: str = None):
        """Notify clients of status updates, optionally for a specific camera."""
        status = self.get_stream_status()
        if image_path:
            status["new_image"] = image_path
        if stop_reason:
            status["stop_reason"] = stop_reason
        if camera_key:
            status["camera_key"] = camera_key
        tasks = [websocket.send_json(status) for websocket in self.connections]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_stream_status(self):
        """Return status for all cameras."""
        return {
            "cameras": {
                camera_key: {
                    "preview_status": "running" if self.active_preview.get(camera_key, False) else "stopped",
                    "storage_status": "running" if self.active_storage.get(camera_key, False) else "stopped",
                    "save_interval": self.cameras[camera_key]["save_interval"],
                    "current_folder": str(self.cameras[camera_key]["folder"]) if self.cameras[camera_key]["folder"] else None,
                    "camera_type": self.cameras[camera_key]["type"],
                    "camera_index": self.cameras[camera_key]["index"]
                }
                for camera_key in self.cameras
            },
            "camera_status": self.camera_status
        }

    async def zip_folder_generator(self, folder_path: str):
        """Generator to stream zip file creation with early browser activity."""
        folder_full_path = IMAGES_DIR / folder_path
        if not folder_full_path.exists() or not folder_full_path.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")

        fake_zip = io.BytesIO()
        with ZipFile(fake_zip, 'w') as zf:
            zf.writestr("placeholder.txt", "downloading...")
        fake_zip.seek(0)
        yield fake_zip.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
            with ZipFile(temp_zip.name, "w") as zip_file:
                for root, _, files in os.walk(folder_full_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = str(file_path.relative_to(IMAGES_DIR))
                        zip_file.write(file_path, arcname)
                        await asyncio.sleep(0)

            with open(temp_zip.name, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk

        os.unlink(temp_zip.name)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("myapp")
jpeg_stream = JpegStream()

security = HTTPBearer()
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "supersecretkey":
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check camera availability
    jpeg_stream.camera_status["picamera"] = jpeg_stream.check_picamera_availability()
    jpeg_stream.camera_status["cameraids"] = jpeg_stream.check_cameraids_availability()
    logger.info(f"Camera status: {jpeg_stream.camera_status}")

    # Initialize all available cameras
    for device in jpeg_stream.camera_status["picamera"]:
        try:
            jpeg_stream.set_camera("picamera", index=device["index"])
            logger.info(f"Initialized Raspberry Pi camera at index {device['index']}")
        except Exception as e:
            logger.error(f"Failed to initialize PiCam at index {device['index']}: {str(e)}")

    for device in jpeg_stream.camera_status["cameraids"]:
        try:
            jpeg_stream.set_camera("cameraids", index=device["index"])
            logger.info(f"Initialized CameraIDS (index: {device['index']})")
        except Exception as e:
            logger.error(f"Failed to initialize CameraIDS (index: {device['index']}): {str(e)}")

    yield
    logger.info("Shutting down, cleaning up resources")
    for camera_key in list(jpeg_stream.cameras.keys()):
        await jpeg_stream.stop_storage_task(camera_key)
        await jpeg_stream.stop_preview_task(camera_key)
    jpeg_stream.close()

app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_id = id(websocket)
    logger.debug(f"WebSocket connected: {websocket_id}")
    try:
        status = jpeg_stream.get_stream_status()
        await websocket.send_json(status)
        jpeg_stream.connections.add(websocket)
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                if message.get("type") == "websocket.disconnect":
                    logger.debug(f"WebSocket disconnected: {websocket_id}")
                    break
            except (WebSocketDisconnect, asyncio.TimeoutError):
                continue
            except Exception as e:
                logger.debug(f"WebSocket receive error: {e}")
                break
    finally:
        try:
            jpeg_stream.connections.remove(websocket)
        except KeyError:
            logger.debug(f"WebSocket {websocket_id} already removed")
        if not jpeg_stream.connections:
            logger.info("No more connections, stopping all previews")
            for camera_key in list(jpeg_stream.cameras):
                await jpeg_stream.stop_preview_task(camera_key)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000)
                logger.debug(f"Closed WebSocket: {websocket_id}")
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

@app.post("/select_camera")
async def select_camera(request: CameraSelectionRequest):
    try:
        camera_key = request.camera_key.strip()
        if "_" not in camera_key:
            raise HTTPException(status_code=400, detail=f"Invalid camera key format: {camera_key}")
        
        camera_type, index_str = camera_key.split("_", 1)
        if camera_type not in ["picamera", "cameraids"]:
            raise HTTPException(status_code=400, detail=f"Invalid camera type: {camera_type}")
        
        try:
            index = int(index_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid index format: {index_str}")

        logger.info(f"Parsed camera_key: {camera_key} -> type={camera_type}, index={index}")

        if camera_type == "picamera":
            if not any(cam["index"] == index for cam in jpeg_stream.camera_status["picamera"]):
                raise HTTPException(status_code=400, detail=f"Raspberry Pi camera at index {index} is not available")
        elif camera_type == "cameraids":
            if not any(cam["index"] == index for cam in jpeg_stream.camera_status["cameraids"]):
                raise HTTPException(status_code=400, detail=f"CameraIDS at index {index} is not available")

        camera = jpeg_stream.set_camera(camera_type, index)
        message = f"Using existing camera {camera_key}" if camera_key in jpeg_stream.cameras else f"Initialized new camera {camera_key}"
        await jpeg_stream.notify_clients()
        return {"message": message}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to select {request.camera_key}: {str(e)}")
        await jpeg_stream.notify_clients()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_camera")
async def remove_camera(request: CameraSelectionRequest):
    try:
        if request.camera_key not in jpeg_stream.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {request.camera_key} not found")
        await jpeg_stream.stop_storage_task(request.camera_key)
        await jpeg_stream.stop_preview_task(request.camera_key)
        jpeg_stream.close_camera(request.camera_key)
        await jpeg_stream.notify_clients()
        return {"message": f"Removed {request.camera_key}"}
    except Exception as e:
        logger.error(f"Failed to remove {request.camera_key}: {str(e)}")
        await jpeg_stream.notify_clients()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cameras")
async def get_cameras():
    try:
        cameras = []
        for device in jpeg_stream.camera_status["picamera"]:
            cameras.append({
                "camera_key": jpeg_stream._get_camera_key("picamera", device["index"]),
                "type": "picamera",
                "index": device["index"],
                "display_name": device.get("display_name", f"Raspberry Pi Camera {device['index']}"),
                "model": device.get("Model", "PiCam"),
                "serial": device.get("Id", f"picam_{device['index']}"),
                "label": device.get("label", f"PiCam {device['index']}"),
                "location": device.get("Location"),
                "rotation": device.get("Rotation")
            })
        for device in jpeg_stream.camera_status["cameraids"]:
            cameras.append({
                "camera_key": jpeg_stream._get_camera_key("cameraids", device["index"]),
                **device,
                "type": "cameraids"
            })
        return cameras
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        return []

@app.get("/get_camera_status")
async def get_camera_status():
    return jpeg_stream.camera_status

@app.get("/explorer")
async def list_directory(path: str = ""):
    target_path = IMAGES_DIR / path
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")
    items = [
        {"name": item.name, "path": str(item.relative_to(IMAGES_DIR)), "type": "folder" if item.is_dir() else "file"}
        for item in target_path.iterdir()
    ]
    return {"path": path, "items": items}

@app.get("/images/{filename:path}")
async def get_image(filename: str):
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        headers={"Content-Disposition": f'inline; filename="{image_path.name}"'}
    )

@app.delete("/delete/{filename:path}")
async def delete_file(filename: str):
    file_path = IMAGES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File or folder not found")
    try:
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)
        return {"message": f"'{filename}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_zip/{folder_path:path}")
async def download_zip(folder_path: str):
    full_path = IMAGES_DIR / folder_path
    if not full_path.exists() or not full_path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")

    z = zipstream.ZipFile(mode='w', compression=zipstream.ZIP_DEFLATED)

    for root, _, files in os.walk(full_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, IMAGES_DIR)
            z.write(file_path, arcname=arcname)

    headers = {
        "Content-Disposition": f'attachment; filename="{folder_path.split("/")[-1]}.zip"',
        "Content-Type": "application/zip",
    }

    return StreamingResponse(z, headers=headers)

@app.delete("/delete_folder/{folder_path:path}")
async def delete_folder(folder_path: str):
    target_folder = IMAGES_DIR / folder_path
    if not target_folder.exists() or not target_folder.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    try:
        shutil.rmtree(target_folder)
        return {"message": f"Folder '{folder_path}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")

@app.get("/get_stream_status")
async def get_stream_status():
    return jpeg_stream.get_stream_status()

@app.post("/start_storage")
async def start_storage(request: CameraSelectionRequest):
    try:
        await jpeg_stream.start_storage_task(request.camera_key)
        return {"message": f"Image storage task started for {request.camera_key}"}
    except Exception as e:
        logger.error(f"Failed to start storage for {request.camera_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_storage")
async def stop_storage(request: CameraSelectionRequest):
    try:
        await jpeg_stream.stop_storage_task(request.camera_key)
        return {"message": f"Image storage task stopped for {request.camera_key}"}
    except Exception as e:
        logger.error(f"Failed to stop storage for {request.camera_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_preview")
async def start_preview(request: CameraSelectionRequest):
    logger.info(f"Received start_preview request: {request.camera_key}")
    try:
        await jpeg_stream.start_preview_task(request.camera_key)
        return {"message": f"Preview stream started for {request.camera_key}"}
    except Exception as e:
        logger.error(f"Failed to start preview for {request.camera_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_preview")
async def stop_preview(request: CameraSelectionRequest):
    try:
        await jpeg_stream.stop_preview_task(request.camera_key)
        return {"message": f"Preview stream stopped for {request.camera_key}"}
    except Exception as e:
        logger.error(f"Failed to stop preview for {request.camera_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_interval")
async def set_interval(interval: IntervalRequest):
    try:
        await jpeg_stream.set_interval(interval.camera_key, interval.interval)
        await jpeg_stream.notify_clients()
        return {"message": f"Image save interval set to {interval.interval} seconds for {interval.camera_key}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def shutdown_server():
    logger.info("Initiating graceful shutdown...")
    close_tasks = []
    for ws in list(jpeg_stream.connections.copy()):
        try:
            close_tasks.append(ws.close(code=1000))
            logger.debug(f"Scheduled closure for WebSocket: {id(ws)}")
        except Exception as e:
            logger.debug(f"Skipping WebSocket closure: {e}")
    if close_tasks:
        try:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Error during WebSocket closure: {e}")
    jpeg_stream.connections.clear()
    logger.debug("All WebSocket connections cleared")
    os.kill(os.getpid(), signal.SIGTERM)
    logger.info("Shutdown signal sent")

@app.post("/capture_image")
async def capture_image(request: CameraSelectionRequest):
    try:
        image_path = await jpeg_stream.capture_single_image(request.camera_key)
        return {"message": f"Image captured and saved to {image_path} for {request.camera_key}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to capture image for {request.camera_key}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restart_server", dependencies=[Depends(verify_token)])
async def restart_server():
    try:
        logger.info("Server restart requested")
        for camera_key in list(jpeg_stream.cameras.keys()):
            logger.debug(f"Stopping storage for {camera_key}")
            await jpeg_stream.stop_storage_task(camera_key)
            logger.debug(f"Stopping preview for {camera_key}")
            await jpeg_stream.stop_preview_task(camera_key)
        logger.debug("Closing cameras")
        jpeg_stream.close()
        logger.info("Cleanup complete, initiating shutdown")
        asyncio.create_task(shutdown_server())
        return {"message": "Server restart initiated"}
    except Exception as e:
        logger.error(f"Failed to restart server: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
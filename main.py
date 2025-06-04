import io
import asyncio
from zipfile import ZipFile
import tempfile
import shutil
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from threading import Condition
from contextlib import asynccontextmanager
import numpy as np
import cv2
import os
import datetime
import time
from pydantic import BaseModel
from pathlib import Path
import json
import logging
import glob
import shutil  # Added for disk space checking
from typing import Dict

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
        self.camera_status = {"picamera": False, "cameraids": False}
        self.max_folder_size = 1_073_741_824  # 1GB
        self.last_image_timestamps: Dict[str, float] = {}  # Track last image send time per camera
        os.makedirs("images", exist_ok=True)

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
    
    def _test_picamera(self) -> bool:
        """Test if a Raspberry Pi camera is functional."""
        try:
            video_devices = glob.glob("/dev/video*")
            if not video_devices:
                logging.debug("No video devices found in /dev/video*")
                return False
            with Picamera2() as cam:
                logging.debug("Attempting to configure Picamera2")
                config = cam.create_preview_configuration(main={"size": (640, 480)})
                cam.configure(config)
                logging.debug("Starting Picamera2")
                cam.start()
                time.sleep(0.1)  # Brief delay to ensure camera starts
                cam.stop()
                logging.info("Raspberry Pi camera detected and functional")
                return True
        except Exception as e:
            logging.debug(f"Failed to detect Raspberry Pi camera: {str(e)}")
            return False

    def check_picamera_availability(self):
        # Unchanged from original
        return PICAMERA_AVAILABLE and self._test_picamera()

    def check_cameraids_availability(self):
        # Unchanged from original
        return CameraIDS.list_devices() if CAMERAIDS_AVAILABLE else []

    def _get_camera_key(self, camera_type: str, index: int) -> str:
        """Generate a unique key for a camera."""
        return f"{camera_type}_{index}"

    def set_camera(self, camera_type: str, index: int = 0):
        """Initialize a specific camera and add it to the cameras dict."""
        camera_key = self._get_camera_key(camera_type, index)
        camera_type = camera_type.lower()
        if camera_type not in ["picamera", "cameraids"]:
            raise ValueError("Invalid camera type. Use 'picamera' or 'cameraids'")

        # Close existing camera for this key, if any
        if camera_key in self.cameras:
            self.close_camera(camera_key)

        if camera_type == "picamera":
            if not self.camera_status["picamera"]:
                raise RuntimeError("Raspberry Pi camera is not available")
            if index != 0:
                raise ValueError("PiCam only supports index 0")
            try:
                camera = Picamera2()
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
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Picamera2: {str(e)}")
        elif camera_type == "cameraids":
            if not self.camera_status["cameraids"]:
                raise RuntimeError("CameraIDS is not available")
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

    def _image_to_jpeg(self, image, camera_key: str):
        """Convert image to JPEG for a specific camera."""
        camera_info = self.cameras[camera_key]
        if camera_info["auto_feature_manager"]:
            camera_info["auto_feature_manager"](image)
        np_arr = image.get_numpy_3D()
        success, jpeg_data = cv2.imencode('.jpg', np_arr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")
        return jpeg_data.tobytes()

    def _capture_callback(self, image, camera_key: str):
        """Capture callback for CameraIDS."""
        if image is not None:
            try:
                start_time = time.time()
                self.cameras[camera_key]["output"].write(self._image_to_jpeg(image, camera_key))
                encode_duration = time.time() - start_time
                logging.info(f"Camera {camera_key}: Captured and encoded frame, duration: {encode_duration:.3f}s")
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
                    image_path = IMAGES_DIR / folder / f"original_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), img)
                    camera_info["last_save_time"] = current_time
                    await self.notify_clients(image_path=str(image_path.relative_to(IMAGES_DIR)), camera_key=camera_key)

                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Storage task error for {camera_key}: {e}")
        finally:
            if camera_key in self.cameras:
                camera = self.cameras[camera_key]["camera"]
                if camera_type == "picamera" and camera.started and not self.active_preview.get(camera_key, False):
                    camera.stop_recording()
                elif camera_type == "cameraids" and camera.acquiring and not self.active_preview.get(camera_key, False):
                    camera.stop_capturing()
                    camera.stop_acquisition()

    async def stream_preview(self, camera_key: str):
        """Stream preview for a specific camera."""
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
                        if not self.reinitialize_camera(camera_key):
                            raise RuntimeError(f"Failed to reinitialize CameraIDS (key: {camera_key})")
                        camera = self.cameras[camera_key]["camera"]
                        camera.start_acquisition()
                        camera.start_capturing(on_capture_callback=lambda img: self._capture_callback(img, camera_key))

            while self.active_preview.get(camera_key, False) and self.connections:
                jpeg_data = await camera_info["output"].read()
                tasks = [
                    websocket.send_bytes(
                        b"--frame\r\nContent-Type: image/jpeg\r\nX-Camera-Key: " + camera_key.encode() + b"\r\n\r\n" + jpeg_data + b"\r\n"
                    )
                    for websocket in self.connections.copy()
                ]
                start_time = time.time()
                
                # Log interval between sends
                current_time = start_time
                if camera_key in self.last_image_timestamps:
                    interval = current_time - self.last_image_timestamps[camera_key]
                    frequency = 1.0 / interval if interval > 0 else 0.0
                    logging.info(
                        f"Camera {camera_key}: Sending image at {datetime.datetime.now().isoformat()}, "
                        f"interval: {interval:.3f}s, frequency: {frequency:.2f}Hz"
                    )
                self.last_image_timestamps[camera_key] = current_time

                await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                logging.info(
                    f"Camera {camera_key}: Sent image at {datetime.datetime.now().isoformat()}, "
                    f"send duration: {(end_time - start_time):.3f}s"
                )
                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Preview task error for {camera_key}: {e}")
            raise
        finally:
            self.active_preview[camera_key] = False
            if camera_key in self.cameras:
                camera = self.cameras[camera_key]["camera"]
                camera_type = self.cameras[camera_key]["type"]
                if not self.active_storage.get(camera_key, False):
                    if camera_type == "picamera":
                        camera.stop_recording()
                    elif camera_type == "cameraids":
                        camera.stop_capturing()
                        camera.stop_acquisition()
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
                    if camera_type == "picamera":
                        camera.stop()
                    elif camera_type == "cameraids":
                        camera.stop_capturing()
                        camera.stop_acquisition()
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
                    if camera_type == "picamera":
                        camera.stop()
                    elif camera_type == "cameraids":
                        camera.stop_capturing()
                        camera.stop_acquisition()
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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("myapp")
jpeg_stream = JpegStream()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check camera availability
    jpeg_stream.camera_status["picamera"] = jpeg_stream.check_picamera_availability()
    jpeg_stream.camera_status["cameraids"] = bool(jpeg_stream.check_cameraids_availability())
    logger.info(f"Camera status: {jpeg_stream.camera_status}")

    # Initialize all available cameras
    if jpeg_stream.camera_status["picamera"]:
        try:
            jpeg_stream.set_camera("picamera", index=0)
            logger.info("Initialized Raspberry Pi camera")
        except Exception as e:
            logger.error(f"Failed to initialize PiCam: {str(e)}")

    if jpeg_stream.camera_status["cameraids"]:
        for device in jpeg_stream.check_cameraids_availability():
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
    status = jpeg_stream.get_stream_status()
    await websocket.send_json(status)
    jpeg_stream.connections.add(websocket)
    
    try:
        # Start preview for all initialized cameras
        for camera_key in jpeg_stream.cameras:
            if not jpeg_stream.active_preview.get(camera_key, False):
                await jpeg_stream.start_preview_task(camera_key)
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        jpeg_stream.connections.remove(websocket)
        if not jpeg_stream.connections:
            logger.info("No more connections, stopping all previews")
            for camera_key in jpeg_stream.cameras:
                await jpeg_stream.stop_preview_task(camera_key)

IMAGES_DIR = Path("images")

@app.post("/select_camera")
async def select_camera(request: CameraSelectionRequest):
    try:
        # Parse camera_key (e.g., "cameraids_0" -> type="cameraids", index=0)
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

        # Validate camera availability
        if camera_type == "picamera" and not jpeg_stream.camera_status["picamera"]:
            raise HTTPException(status_code=400, detail="Raspberry Pi camera is not available")
        if camera_type == "cameraids":
            available_ids_cameras = jpeg_stream.check_cameraids_availability()
            if not any(device["index"] == index for device in available_ids_cameras):
                raise HTTPException(status_code=400, detail=f"Invalid CameraIDS index: {index}")

        # Set camera
        jpeg_stream.set_camera(camera_type, index)
        await jpeg_stream.notify_clients()
        return {"message": f"Added {camera_key}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to add {request.camera_key}: {str(e)}")
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
        # Add PiCam if available
        if jpeg_stream.camera_status["picamera"]:
            cameras.append({
                "camera_key": jpeg_stream._get_camera_key("picamera", 0),  # Add camera_key
                "type": "picamera",
                "index": 0,
                "display_name": "Raspberry Pi Camera",
                "model": "PiCam",
                "serial": "N/A",
                "label": "Raspberry Pi Camera"
            })
        # Add IDS cameras
        if jpeg_stream.camera_status["cameraids"]:
            cameras.extend([
                {
                    "camera_key": jpeg_stream._get_camera_key("cameraids", device["index"]),  # Add camera_key
                    **device,
                    "type": "cameraids"
                }
                for device in jpeg_stream.check_cameraids_availability()
            ])
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
    folder_full_path = IMAGES_DIR / folder_path
    if not folder_full_path.exists() or not folder_full_path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with ZipFile(temp_zip.name, "w") as zip_file:
        for root, _, files in os.walk(folder_full_path):
            for file in files:
                file_path = Path(root) / file
                arcname = str(file_path.relative_to(IMAGES_DIR))
                zip_file.write(file_path, arcname)
    
    return FileResponse(
        path=temp_zip.name,
        filename=f"{folder_path.split('/')[-1]}.zip",
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={folder_path.split('/')[-1]}.zip"}
    )

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

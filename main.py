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
    interval: float

class CameraSelectionRequest(BaseModel):
    camera_type: str  # 'picamera' or 'cameraids'
    index: int = 0    # Index for IDS cameras, default 0 for PiCam

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
        self.active_preview = False
        self.active_storage = False
        self.connections = set()
        self.camera = None
        self.camera_type = None
        self.camera_index = 0  # Added to store current camera index
        self.preview_task = None
        self.storage_task = None
        self.current_folder = None
        self.output = None
        self.save_interval = 5
        self.last_save_time = 0
        self.auto_feature_manager = None
        self.camera_status = {"picamera": False, "cameraids": False}
        self.max_folder_size = 1_073_741_824  # 1GB in bytes

        os.makedirs("images", exist_ok=True)
        self.output = StreamingOutput()

    def check_picamera_availability(self):
        """Check if a Raspberry Pi camera is available."""
        if not PICAMERA_AVAILABLE:
            logging.debug("Picamera2 library not available")
            return False
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
                time.sleep(0.1)
                cam.stop()
                logging.info("Raspberry Pi camera detected and functional")
                return True
        except Exception as e:
            logging.debug(f"Failed to detect Raspberry Pi camera: {str(e)}")
            return False

    # Modified to return a list of IDS camera devices
    def check_cameraids_availability(self):
        """Check if CameraIDS is available and return list of detected devices."""
        if not CAMERAIDS_AVAILABLE:
            logging.debug("CameraIDS library not available")
            return []
        try:
            logging.debug("Attempting to list CameraIDS devices")
            devices = CameraIDS.list_devices()
            if devices:
                logging.info(f"CameraIDS detected with {len(devices)} device(s)")
                return devices
            else:
                logging.debug("No CameraIDS devices found")
                return []
        except Exception as e:
            logging.debug(f"Failed to detect CameraIDS: {str(e)}")
            return []

    def close(self):
        """Close the camera and clean up resources."""
        if not self.camera:
            logging.debug("No camera to close")
            return
        try:
            if isinstance(self.camera, Picamera2):
                if hasattr(self.camera, 'started') and self.camera.started:
                    self.camera.stop()
                self.camera.close()
                logging.info("Picamera2 camera closed")
            elif isinstance(self.camera, CameraIDS):
                if hasattr(self.camera, 'acquiring') and self.camera.acquiring:
                    self.camera.stop_capturing()
                    self.camera.stop_acquisition()
                self.camera.close()
                logging.info("CameraIDS camera closed")
            else:
                logging.warning(f"Unknown camera type: {type(self.camera)}")
        except Exception as e:
            logging.error(f"Error closing camera (type: {type(self.camera)}): {e}")
        finally:
            self.camera = None
            self.auto_feature_manager = None

    def reinitialize_camera(self):
        """Reinitialize the camera if the handle is invalid."""
        if not self.camera_type:
            logging.error("No camera type set for reinitialization")
            return False
        logging.debug(f"Reinitializing camera: {self.camera_type} (index: {self.camera_index})")
        try:
            self.close()
            self.set_camera(self.camera_type, self.camera_index)
            logging.info(f"Successfully reinitialized {self.camera_type} camera")
            return True
        except Exception as e:
            logging.error(f"Failed to reinitialize {self.camera_type} camera: {str(e)}")
            self.camera = None
            self.camera_type = None
            self.camera_index = 0
            return False

    def set_camera(self, camera_type: str, index: int = 0):
        """Set the camera type and index, and initialize it."""
        camera_type = camera_type.lower()
        if camera_type not in ["picamera", "cameraids"]:
            raise ValueError("Invalid camera type. Use 'picamera' or 'cameraids'")
        
        logging.debug(f"Closing current camera before switching to {camera_type} (index: {index})")
        self.close()

        self.camera_type = camera_type
        self.camera_index = index  # Store the index
        logging.debug(f"Setting camera to {self.camera_type} (index: {self.camera_index})")

        if self.camera_type == "picamera":
            if not self.camera_status["picamera"]:
                raise RuntimeError("Raspberry Pi camera is not available")
            if index != 0:
                raise ValueError("PiCam only supports index 0")
            try:
                self.camera = Picamera2()
                video_config = self.camera.create_video_configuration(main={"size": (1920, 1080)})
                self.camera.configure(video_config)
                self.auto_feature_manager = None
                logging.info("Initialized Picamera2")
            except Exception as e:
                self.camera = None
                self.camera_type = None
                self.camera_index = 0
                raise RuntimeError(f"Failed to initialize Picamera2: {str(e)}")
        elif self.camera_type == "cameraids":
            if not self.camera_status["cameraids"]:
                raise RuntimeError("CameraIDS is not available")
            try:
                self.camera = CameraIDS(id_device=index)
                self.camera.set_roi_max()
                self.auto_feature_manager = AutoFeatureManager(self.camera)
                self.auto_feature_manager.auto_white_balance = 'on'
                self.auto_feature_manager.auto_exposure = 'on'
                self.auto_feature_manager.auto_gain = 'on'
                logging.info(f"Initialized CameraIDS (index: {index})")
            except Exception as e:
                self.camera = None
                self.camera_type = None
                self.camera_index = 0
                logging.error(f"CameraIDS initialization failed: {str(e)}")
                raise RuntimeError(f"Failed to initialize CameraIDS: {str(e)}")

    def create_new_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_folder = Path(f"stream_{timestamp}")
        os.makedirs(IMAGES_DIR / self.current_folder, exist_ok=True)

    async def set_interval(self, interval: float):
        if interval <= 0:
            raise ValueError("Interval must be greater than zero.")
        self.save_interval = interval

    def _image_to_jpeg(self, image):
        if self.auto_feature_manager:
            self.auto_feature_manager(image)
        np_arr = image.get_numpy_3D()
        success, jpeg_data = cv2.imencode('.jpg', np_arr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            raise RuntimeError("Failed to encode image to JPEG")
        return jpeg_data.tobytes()

    def _capture_callback(self, image):
        if image is not None:
            try:
                self.output.write(self._image_to_jpeg(image))
            except Exception as e:
                logging.error(f"Error in capture callback: {e}")

    def get_available_disk_space(self):
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

    async def store_images(self):
        try:
            if not self.camera or not self.camera_type:
                raise RuntimeError("No camera initialized")
            if self.camera_type == "picamera":
                if not self.camera.started:
                    self.camera.start_recording(MJPEGEncoder(), FileOutput(self.output), Quality.MEDIUM)
            elif self.camera_type == "cameraids":
                try:
                    if not self.camera.acquiring:
                        self.camera.start_acquisition()
                    if not self.camera.capturing_threaded:
                        self.camera.start_capturing(on_capture_callback=self._capture_callback)
                except Exception as e:
                    if "PEAK_RETURN_CODE_INVALID_HANDLE" in str(e):
                        logging.warning("Invalid CameraIDS handle detected, attempting reinitialization")
                        if not self.reinitialize_camera():
                            raise RuntimeError("Failed to reinitialize CameraIDS")
                        self.camera.start_acquisition()
                        self.camera.start_capturing(on_capture_callback=self._capture_callback)

            self.create_new_folder()

            while self.active_storage:
                if self.get_available_disk_space() < 5 * 1024 * 1024 * 1024:
                    logging.warning("Disk space below 5GB, stopping storage")
                    await self.stop_storage_task()
                    await self.notify_clients(stop_reason="Insufficient disk space (less than 5GB)")
                    return

                if self.current_folder and self.get_folder_size(IMAGES_DIR / self.current_folder) > self.max_folder_size:
                    logging.info(f"Folder {self.current_folder} exceeded 1GB, creating new folder")
                    self.create_new_folder()
                    await self.notify_clients()

                jpeg_data = await self.output.read()
                img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)

                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = IMAGES_DIR / self.current_folder / f"original_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), img)
                    self.last_save_time = current_time
                    await self.notify_clients(image_path=str(image_path.relative_to(IMAGES_DIR)))

                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Storage task error: {e}")
        finally:
            if self.camera:
                if self.camera_type == "picamera" and self.camera.started and not self.active_preview:
                    self.camera.stop_recording()
                elif self.camera_type == "cameraids" and self.camera.acquiring and not self.active_preview:
                    self.camera.stop_capturing()
                    self.camera.stop_acquisition()

    async def stream_preview(self):
        try:
            if not self.camera or not self.camera_type:
                raise RuntimeError("No camera initialized")
            if self.camera_type == "picamera":
                if not self.camera.started:
                    self.camera.start_recording(MJPEGEncoder(), FileOutput(self.output), Quality.MEDIUM)
            elif self.camera_type == "cameraids":
                try:
                    if not self.camera.acquiring:
                        self.camera.start_acquisition()
                    if not self.camera.capturing_threaded:
                        self.camera.start_capturing(on_capture_callback=self._capture_callback)
                except Exception as e:
                    if "PEAK_RETURN_CODE_INVALID_HANDLE" in str(e):
                        logging.warning("Invalid CameraIDS handle detected, attempting reinitialization")
                        if not self.reinitialize_camera():
                            raise RuntimeError("Failed to reinitialize CameraIDS")
                        self.camera.start_acquisition()
                        self.camera.start_capturing(on_capture_callback=self._capture_callback)

            while self.active_preview and self.connections:
                jpeg_data = await self.output.read()
                tasks = [websocket.send_bytes(jpeg_data) for websocket in self.connections.copy()]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0.1)
        except Exception as e:
            logging.error(f"Preview task error: {e}")
            raise
        finally:
            self.active_preview = False
            if self.camera:
                if self.camera_type == "picamera" and self.camera.started and not self.active_storage:
                    self.camera.stop_recording()
                elif self.camera_type == "cameraids" and self.camera.acquiring and not self.active_storage:
                    self.camera.stop_capturing()
                    self.camera.stop_acquisition()
            await self.notify_clients()

    async def start_storage_task(self):
        if not self.camera or not self.camera_type:
            raise HTTPException(status_code=500, detail="Camera not initialized")
        if not self.active_storage:
            self.active_storage = True
            self.storage_task = asyncio.create_task(self.store_images())
            await self.notify_clients()

    async def stop_storage_task(self):
        if self.active_storage:
            self.active_storage = False
            if self.storage_task:
                self.storage_task.cancel()
                try:
                    await self.storage_task
                except asyncio.CancelledError:
                    pass
                self.storage_task = None
            if not self.active_preview and self.camera:
                if self.camera_type == "picamera":
                    self.camera.stop()
                elif self.camera_type == "cameraids":
                    self.camera.stop_capturing()
                    self.camera.stop_acquisition()
            await self.notify_clients()

    async def start_preview_task(self):
        if not self.camera or not self.camera_type:
            raise HTTPException(status_code=500, detail="Camera not initialized")
        if not self.active_preview and self.connections:
            self.active_preview = True
            self.preview_task = asyncio.create_task(self.stream_preview())
            await self.notify_clients()

    async def stop_preview_task(self):
        if self.active_preview:
            self.active_preview = False
            if self.preview_task:
                self.preview_task.cancel()
                try:
                    await self.preview_task
                except asyncio.CancelledError:
                    pass
                self.preview_task = None
            if not self.active_storage and self.camera:
                if self.camera_type == "picamera":
                    self.camera.stop()
                elif self.camera_type == "cameraids":
                    self.camera.stop_capturing()
                    self.camera.stop_acquisition()
            await self.notify_clients()

    async def notify_clients(self, image_path: str = None, stop_reason: str = None):
        status = self.get_stream_status()
        if image_path:
            status["new_image"] = image_path
        if stop_reason:
            status["stop_reason"] = stop_reason
        tasks = [websocket.send_json(status) for websocket in self.connections]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_stream_status(self):
        return {
            "preview_status": "running" if self.active_preview else "stopped",
            "storage_status": "running" if self.active_storage else "stopped",
            "save_interval": self.save_interval,
            "current_folder": str(self.current_folder) if self.current_folder else None,
            "camera_type": self.camera_type,
            "camera_index": self.camera_index,  # Added to include index in status
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
    jpeg_stream.camera_status["cameraids"] = bool(jpeg_stream.check_cameraids_availability())  # Modified to boolean

    # Log camera status
    logger.info(f"Camera status: {jpeg_stream.camera_status}")

    # Select default camera
    default_camera = os.getenv("CAMERA_TYPE", "picamera")
    if not jpeg_stream.camera_status["picamera"] and not jpeg_stream.camera_status["cameraids"]:
        logger.warning("No cameras available, starting without default camera")
        yield
        return

    if default_camera == "picamera" and not jpeg_stream.camera_status["picamera"]:
        if jpeg_stream.camera_status["cameraids"]:
            default_camera = "cameraids"
            logger.info("No Raspberry Pi camera available, falling back to CameraIDS")
        else:
            logger.warning("No cameras available, starting without default camera")
            yield
            return

    try:
        logger.info(f"Attempting to initialize {default_camera} camera")
        jpeg_stream.set_camera(default_camera, index=0)  # Default to index 0
        logger.info(f"Successfully initialized {default_camera} camera")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize {default_camera} camera: {str(e)}")
        logger.warning("Continuing without default camera initialization")
        yield
    finally:
        logger.info("Shutting down, cleaning up resources")
        await jpeg_stream.stop_storage_task()
        await jpeg_stream.stop_preview_task()
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
        if not jpeg_stream.active_preview:
            await jpeg_stream.start_preview_task()
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
            logger.info("No more connections, stopping preview")
            await jpeg_stream.stop_preview_task()

IMAGES_DIR = Path("images")

# Modified to handle camera type and index
@app.post("/select_camera")
async def select_camera(request: CameraSelectionRequest):
    try:
        if request.camera_type == "picamera" and not jpeg_stream.camera_status["picamera"]:
            raise HTTPException(status_code=400, detail="Raspberry Pi camera is not available")
        if request.camera_type == "cameraids" and not jpeg_stream.camera_status["cameraids"]:
            raise HTTPException(status_code=400, detail="CameraIDS is not available")
        # Validate IDS camera index
        if request.camera_type == "cameraids":
            available_ids_cameras = jpeg_stream.check_cameraids_availability()
            if not any(device["index"] == request.index for device in available_ids_cameras):
                raise HTTPException(status_code=400, detail=f"Invalid CameraIDS index: {request.index}")
        await jpeg_stream.stop_storage_task()
        await jpeg_stream.stop_preview_task()
        jpeg_stream.set_camera(request.camera_type, request.index)
        await jpeg_stream.notify_clients()
        return {"message": f"Switched to {request.camera_type} camera (index: {request.index})"}
    except Exception as e:
        logger.error(f"Failed to switch to {request.camera_type} camera (index: {request.index}): {str(e)}")
        await jpeg_stream.notify_clients()
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint to list all cameras
@app.get("/cameras")
async def get_cameras():
    try:
        cameras = []
        # Add PiCam if available
        if jpeg_stream.camera_status["picamera"]:
            cameras.append({
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
                {**device, "type": "cameraids"}
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
async def start_storage():
    try:
        await jpeg_stream.start_storage_task()
        return {"message": "Image storage task started"}
    except Exception as e:
        logger.error(f"Failed to start storage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_storage")
async def stop_storage():
    try:
        await jpeg_stream.stop_storage_task()
        return {"message": "Image storage task stopped"}
    except Exception as e:
        logger.error(f"Failed to stop storage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_preview")
async def start_preview():
    try:
        await jpeg_stream.start_preview_task()
        return {"message": "Preview stream started"}
    except Exception as e:
        logger.error(f"Failed to start preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_preview")
async def stop_preview():
    try:
        await jpeg_stream.stop_preview_task()
        return {"message": "Preview stream stopped"}
    except Exception as e:
        logger.error(f"Failed to stop preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_interval")
async def set_interval(request: IntervalRequest):
    try:
        await jpeg_stream.set_interval(request.interval)
        await jpeg_stream.notify_clients()
        return {"message": f"Image save interval set to {request.interval} seconds"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
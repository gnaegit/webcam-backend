# Raspberry Pi Webcam Backend

The backend for a Raspberry Pi-based webcam system, built with FastAPI. It provides APIs and WebSocket streaming to control cameras (Raspberry Pi via picamera2 or CameraIDS), capture and store images, and manage stored files. Integrates with the frontend at [webcam-frontend](https://github.com/gnaegit/webcam-frontend).

## Features

- **Camera Control**: Supports Raspberry Pi (picamera2) and CameraIDS cameras with automatic detection.
- **WebSocket Streaming**: Streams camera feed via `/ws`.
- **Image Storage**: Saves images to `images/` at configurable intervals, with disk space monitoring (stops if <5GB).
- **File Management**: List, download (as ZIP), and delete images/folders.
- **API Endpoints**: Control preview, storage, camera selection, and status.

## Prerequisites

- **Hardware**: Raspberry Pi (tested on Raspberry Pi 5) with compatible cameras (picamera and/or CameraIDS).
- **Software**:
  - Python (v3.8 or later)
  - Systemd (for production)
  - IDS Peak software (for CameraIDS, see installation below)
- **Dependencies**: Listed in `requirements.txt` (e.g., `fastapi`, `uvicorn`, `picamera2`, `ids-peak`).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gnaegit/webcam-backend.git
   cd webcam-backend
   mv webcam-backend /home/pi/Projects/webcam-backend
   cd /home/pi/Projects/camera-app-fish
   ```
   - Note: The service expects the project in `/home/pi/Projects/webcam-backend`.

2. **Set Up Python Environment**:
   - Create and activate a virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install IDS Peak (for CameraIDS)**:
   - Download the IDS Peak `.deb` package for Linux (arm64 for Raspberry Pi) from https://en.ids-imaging.com/download-peak.html.
   - Install dependencies:
     ```bash
     sudo apt update
     sudo apt install -y libusb-1.0-0 libqt5core5a libqt5gui5 libqt5widgets5 libqt5quick5 \
       qml-module-qtquick-window2 qml-module-qtquick2 qml-module-qtquick-dialogs \
       qml-module-qtquick-controls qml-module-qtquick-layouts qml-module-qt-labs-settings \
       qml-module-qt-labs-folderlistmodel
     ```
   - Install IDS Peak:
     ```bash
     sudo dpkg -i ids-peak-<version>-<arch>.deb
     ```
     - Replace `<version>` and `<arch>` with the downloaded package details (e.g., `ids-peak-2.10.0.0.4-arm64.deb`).
   - Verify installation:
     ```bash
     ids_visioncockpit
     ```
     - Ensure CameraIDS is detected in IDS Vision Cockpit.
   - Set environment variable:
     ```bash
     export GENICAM_GENTL64_PATH=/usr/lib/ids/cti
     ```
     - This is included in `webcam-backend.service` for production.

4. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Installs `fastapi`, `uvicorn`, `picamera2`, `opencv-python`, `ids-peak`, `websockets`, etc.
   - Note: `ids-peak` is the Python API; IDS Peak software must be installed first.

5. **Configure Camera**:
   - Verify camera detection:
     ```bash
     libcamera-hello --list-cameras
     ```
   - For CameraIDS, use `ids_visioncockpit` to confirm detection and update firmware if needed:
     ```bash
     ids_deviceupdate -s <last-four-digits-serialnumber> -U --guf <path-to-guf-file>
     ```

6. **Set Up Systemd Service (Production)**:
   - Copy `webcam-backend.service` to `/etc/systemd/system/`:
     ```bash
     sudo cp webcam-backend.service /etc/systemd/system/
     ```
   - Service configuration:
     ```ini
     [Unit]
     Description=Fishapp FastAPI Server
     After=network.target

     [Service]
     User=pi
     WorkingDirectory=/home/pi/Projects/webcam-backend
     ExecStart=/home/pi/Projects/camera-app-fish/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
     Restart=always
     Environment="PYTHONUNBUFFERED=1"
     Environment="GENICAM_GENTL64_PATH=/usr/lib/ids/cti"
     Environment="PATH=/home/pi/Projects/webcam-backend/venv/bin:/usr/local/bin:/usr/bin:/bin"
     Environment="VIRTUAL_ENV=/home/pi/Projects/webcam-backend/venv"

     [Install]
     WantedBy=multi-user.target
     ```
   - Enable the service:
     ```bash
     sudo systemctl enable webcam-backend.service
     ```

7. **Verify Setup**:
   - Start the service:
     ```bash
     sudo systemctl start webcam-backend.service
     sudo systemctl status webcam-backend.service
     ```
   - Test endpoint:
     ```bash
     curl http://localhost:8000/get_stream_status
     ```

## Usage

1. **Run the Server**:
   - Development:
     ```bash
     source venv/bin/activate
     uvicorn main:app --host 0.0.0.0 --port 8000 --reload
     ```
   - Production:
     ```bash
     sudo systemctl start webcam-backend.service
     ```

2. **API Endpoints**:
   - **WebSocket**:
     - `/ws`: Streams camera feed and status updates.
   - **POST**:
     - `/select_camera`: Switch camera (`{"camera_type": "picamera"}` or `{"camera_type": "cameraids"}`).
     - `/start_storage`: Start image storage.
     - `/stop_storage`: Stop image storage.
     - `/start_preview`: Start preview stream.
     - `/stop_preview`: Stop preview stream.
     - `/set_interval`: Set storage interval (`{"interval": 5}`).
   - **GET**:
     - `/get_camera_status`: Camera availability.
     - `/get_stream_status`: Stream and storage status.
     - `/explorer?path=<path>`: List directory contents.
     - `/images/<filename>`: Retrieve image.
     - `/download_zip/<folder_path>`: Download folder as ZIP.
   - **DELETE**:
     - `/delete/<filename>`: Delete file or folder.
     - `/delete_folder/<folder_path>`: Delete folder.

3. **Integration**:
   - Frontend proxies `/py/:path*` to `http://0.0.0.0:8000/:path*`.
   - Images stored in `images/` are accessible via `/images/<filename>`.

## Project Structure

- `main.py`: FastAPI app with camera control, streaming, and file management.
- `requirements.txt`: Python dependencies.
- `webcam-backend.service`: Systemd service for production.
- `images/`: Directory for stored images (auto-created).

## Development

1. **Stop Production Service**:
   ```bash
   sudo systemctl stop webcam-backend.service
   ```
   - Check port:
     ```bash
     sudo lsof -i :8000
     ```

2. **Run the Server**:
   ```bash
   source venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Testing**:
   - Test endpoints:
     ```bash
     curl -X POST http://localhost:8000/start_preview
     curl http://localhost:8000/get_stream_status
     ```
   - Use a WebSocket client for `/ws`.
   - Verify camera functionality and image storage.

## Production

1. **Stop Development Server**:
   - Kill `uvicorn` process.

2. **Start Service**:
   ```bash
   sudo systemctl start webcam-backend.service
   ```

3. **Verify**:
   ```bash
   sudo systemctl status webcam-backend.service
   ```

## Troubleshooting

- **Service Fails**:
  - Check logs: `journalctl -u webcam-backend.service`.
  - Verify virtual environment: `/home/pi/Projects/webcam-backend/venv`.
  - Ensure IDS Peak and dependencies are installed.
- **Port Conflict**:
  - Check: `sudo lsof -i :8000`.
- **Camera Issues**:
  - For picamera2: Run `libcamera-hello --list-cameras`.
  - For CameraIDS: Run `ids_visioncockpit` and verify `GENICAM_GENTL64_PATH`.
- **WebSocket Errors**:
  - Ensure frontend proxies correctly.

## Contributing

Submit pull requests or issues to [webcam-backend](https://github.com/gnaegit/webcam-backend) or [webcam-frontend](https://github.com/gnaegit/webcam-frontend).

## License

MIT License.

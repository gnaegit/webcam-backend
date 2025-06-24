"""
    Camera interface was adopted from: https://github.com/bertan-karacora/nimbro_camera_ids
"""


import importlib.resources as resources
import threading
import os

import ids_peak.ids_peak as idsp
import ids_peak_ipl.ids_peak_ipl as idsp_ipl
import ids_peak.ids_peak_ipl_extension as idsp_extension

from pathlib import Path

from src.utils import *


TARGET_PIXELFORMAT = idsp_ipl.PixelFormatName_RGB8

class CameraConfigurationError(Exception):
    """Exception raised for camera configuration errors."""
    pass

class CameraIDS:
    def __init__(self, id_device=0, pixel_format=TARGET_PIXELFORMAT):
        self.acquiring = None
        self.capturing_thread = None
        self.capturing_threaded = None
        self.datastream = None
        self.device = None
        self.device_manager = None
        self.converter_image = None
        self.id_device = id_device
        self.killed = None
        self.nodemap = None
        self.pixel_format = pixel_format

        self._initialize()

    def __del__(self):
        self.close()

    def __repr__(self):
        r = f"{__package__}.{self.__class__.__name__}({self.device})"
        return r

    def __str__(self):
        name_model = self.device.ModelName()
        name_interface = self.device.ParentInterface().DisplayName()
        name_system = self.device.ParentInterface().ParentSystem().DisplayName()
        version_system = self.device.ParentInterface().ParentSystem().Version()

        s = f"{name_model} ({name_interface} ; {name_system} v.{version_system})"
        return s
    
    @staticmethod
    def list_devices():
        """
        Return a list of dictionaries containing device information for dropdown display.
        Each dictionary includes index, display name, model, and serial number.
        """
        idsp.Library.Initialize()
        device_manager = idsp.DeviceManager.Instance()
        device_manager.Update()
        device_descriptors = device_manager.Devices()

        devices = []
        for i, descriptor in enumerate(device_descriptors):
            device_info = {
                "index": i,
                "display_name": descriptor.DisplayName(),
                "model": descriptor.ModelName(),
                "serial": descriptor.SerialNumber(),
                "label": f"{descriptor.ModelName()} ({descriptor.SerialNumber()})"
            }
            devices.append(device_info)
        
        return devices

    def get_attributes(self):
        attr = {}
        for node in self.nodemap.Nodes():
            name = node.DisplayName()
            try:
                value = self.nodemap.FindNode(name).Value()
            except:
                value = None
            attr[name] = value

        attr = {k: attr[k] for k in sorted(attr.keys())}
        return attr

    def get_value(self, name):
        value = self.nodemap.FindNode(name).Value()
        return value

    def get_min(self, name):
        min = self.nodemap.FindNode(name).Minimum()
        return min

    def get_max(self, name):
        max = self.nodemap.FindNode(name).Maximum()
        return max

    def get_entry(self, name):
        entry = self.nodemap.FindNode(name).CurrentEntry().Value()
        return entry
    
    def get_nodemap(self):
        return self.nodemap
    
    def get_access_status(self, name):
        all_entries = self.nodemap.FindNode(name)
        if isinstance(all_entries, idsp.EnumerationNode):
            all_entries = all_entries.Entries()
        else:
            all_entries = [all_entries]
        access_status = []
        for entry in all_entries:
            access_status.append(parameter_access_status(entry.AccessStatus()))

        return access_status

    def has_attribute(self, name):
        has_it = self.nodemap.HasNode(name)
        return has_it

    def set_value(self, name, value):
        self.nodemap.FindNode(name).SetValue(value)

    def set_entry(self, name, value):
        all_entries = self.nodemap.FindNode(name)
        if isinstance(all_entries, idsp.EnumerationNode):
            all_entries = all_entries.Entries()
        else:
            all_entries = [all_entries]
        available_entries = []
        for entry in all_entries:
            if entry.AccessStatus() != idsp.NodeAccessStatus_NotAvailable and entry.AccessStatus() != idsp.NodeAccessStatus_NotImplemented:
                available_entries.append(entry.SymbolicValue())
        if value in available_entries:
            self.nodemap.FindNode(name).SetCurrentEntry(value)

    def execute(self, command):
        self.nodemap.FindNode(command).Execute()
        self.nodemap.FindNode(command).WaitUntilDone()

    def load_config(self, name_config):
        full_path = Path(os.getcwd()) / name_config
        self.nodemap.LoadFromFile(str(full_path))

    def save_config(self, name_config):
        self.nodemap.StoreToFile(name_config)

    def reset(self):
        self.execute("ResetToFactoryDefaults")

    def _open(self, device_manager, id_device):
        device_manager.Update()
        device_descriptors = device_manager.Devices()

        if device_descriptors.empty() or len(device_descriptors) <= id_device:
            raise RuntimeError("No device found")

        device_descriptor = device_descriptors[id_device]
        device = device_descriptor.OpenDevice(idsp.DeviceAccessType_Control)
        return device

    def _setup_buffers(self):
        # PayloadSize depends on the image size and the source pixel format
        payload_size = self.get_value("PayloadSize")
        num_buffers = self.datastream.NumBuffersAnnouncedMinRequired()
        for _ in range(num_buffers):
            buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
            self.datastream.QueueBuffer(buffer)

    def _revoke_buffers(self):
        for buffer in self.datastream.AnnouncedBuffers():
            self.datastream.RevokeBuffer(buffer)

    def _preallocate_conversion(self):
        """Pre-allocate conversion buffers to speed up first image conversion while the acquisition is running."""
        image_width = self.get_value("Width")
        image_height = self.get_value("Height")
        input_pixelformat = idsp_ipl.PixelFormat(self.get_entry("PixelFormat"))

        # NOTE: Re-create the image converter, so old conversion buffers get freed
        self.converter_image = idsp_ipl.ImageConverter()
        self.converter_image.PreAllocateConversion(input_pixelformat, self.pixel_format, image_width, image_height)

    def _initialize(self):
        idsp.Library.Initialize()

        self.device_manager = idsp.DeviceManager.Instance()
        self.device = self._open(self.device_manager, self.id_device)

        self.acquiring = False
        self.killed = False
        self.capturing_threaded = False
        self.nodemap = self.device.RemoteDevice().NodeMaps()[0]
        self.datastream = self.device.DataStreams()[0].OpenDataStream()
        self.converter_image = idsp_ipl.ImageConverter()
        self.capturing_thread = threading.Thread(target=self.capture_threaded)

    def close(self):
        try:
            self.stop_acquisition()
        except Exception as e:
            print(f"Close error: {e}")
        finally:
            self.device = None
            idsp.Library.Close()

    def start_acquisition(self):
        if self.acquiring:
            return
        
        self._setup_buffers()

        # Lock parameters that should not be accessed during acquisition
        self.set_value("TLParamsLocked", 1)

        self._preallocate_conversion()

        self.datastream.StartAcquisition()
        self.execute("AcquisitionStart")

        self.acquiring = True

    def stop_acquisition(self):
        if not self.acquiring:
            return

        self.nodemap.FindNode("AcquisitionStop").Execute()

        # Kill the datastream to exit out of pending `WaitForFinishedBuffer` calls
        #self.datastream.KillWait()
        self.datastream.StopAcquisition(idsp.AcquisitionStopMode_Default)
        # Discard all buffers from the acquisition engine. They remain in the announced buffer pool
        self.datastream.Flush(idsp.DataStreamFlushMode_DiscardAll)

        # Unlock parameters
        self.nodemap.FindNode("TLParamsLocked").SetValue(0)

        self._revoke_buffers()

        self.acquiring = False

    def capture(self):
        buffer = self.datastream.WaitForFinishedBuffer(10000)

        # NOTE: This still uses the buffer's underlying memory
        image = idsp_extension.BufferToImage(buffer)

        # This creates a copy the image, so the buffer is free to use again after queuing
        # NOTE: Use `ImageConverter`, since the `ConvertTo` function re-allocates the conversion buffers on every call
        image_converted = self.converter_image.Convert(image, self.pixel_format)

        self.datastream.QueueBuffer(buffer)

        return image_converted

    def start_capturing(self, on_capture_callback=lambda *args: None):
        if not self.acquiring:
            raise ValueError("Camera is not in acquisition mode.")

        if self.capturing_threaded:
            return

        self.capturing_thread = threading.Thread(target=self.capture_threaded, kwargs={"on_capture_callback": on_capture_callback})
        self.capturing_thread.start()
        self.capturing_threaded = True

    def stop_capturing(self):
        if not self.capturing_threaded:
            return

        self.kill_thread()
        self.killed = False

        self.capturing_threaded = False

    def kill_thread(self):
        self.killed = True
        self.capturing_thread.join()

    def set_roi(self, x, y, width, height):
        # Get the minimum ROI and set it. After that there are no size restrictions anymore

        x_min = self.get_min("OffsetX")
        y_min = self.get_min("OffsetY")
        w_min = self.get_min("Width")
        h_min = self.get_min("Height")

        self.set_value("OffsetX", x_min)
        self.set_value("OffsetY", y_min)
        self.set_value("Width", w_min)
        self.set_value("Height", h_min)

        # Get the maximum ROI values
        x_max = self.get_max("OffsetX")
        y_max = self.get_max("OffsetY")
        w_max = self.get_max("Width")
        h_max = self.get_max("Height")

        if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
            return False
        elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
            return False
        else:
            # Now, set final AOI
            self.set_value("OffsetX",x)
            self.set_value("OffsetY",y)
            self.set_value("Width",width)
            self.set_value("Height",height)

            print(x)
            print(y)
            print(width)
            print(height)

            self._revoke_buffers
            self._setup_buffers
            return True
        
    def set_roi_max(self):
        return self.set_roi(0, 0, self.get_max("Width"), self.get_max("Height"))
    
    def capture_threaded(self, on_capture_callback=lambda *args: None):
        while not self.killed:
            image = self.capture()
            on_capture_callback(image)

    def set_binning(self, horizontal_bin_factor, vertical_bin_factor, mode="Sum"):
        """
        Set binning for the camera.
        
        Args:
            horizontal_bin_factor (int): Horizontal binning factor (e.g., 2 for 2x binning).
            vertical_bin_factor (int): Vertical binning factor (e.g., 2 for 2x binning).
            mode (str): Binning mode, either "Sum" or "Average" (if supported).
        
        Returns:
            bool: True if binning was set successfully, False otherwise.
        
        Raises:
            CameraConfigurationError: If binning cannot be set due to invalid parameters or hardware limitations.
        """
        if self.acquiring:
            print("Warning: Cannot set binning while acquisition is active.")
            return False

        try:
            if self.has_attribute("BinningSelector"):
                self.set_entry("BinningSelector", "Sensor")
            else:
                raise CameraConfigurationError("BinningSelector not available.")

            if self.has_attribute("BinningHorizontal"):
                h_min = self.get_min("BinningHorizontal")
                h_max = self.get_max("BinningHorizontal")
                if h_min <= horizontal_bin_factor <= h_max:
                    self.set_value("BinningHorizontal", horizontal_bin_factor)
                    print(f"Horizontal binning set to {horizontal_bin_factor}x")
                else:
                    raise CameraConfigurationError(f"Horizontal binning factor {horizontal_bin_factor} out of range [{h_min}, {h_max}]")
            else:
                raise CameraConfigurationError("Horizontal binning not supported.")

            if self.has_attribute("BinningVertical"):
                v_min = self.get_min("BinningVertical")
                v_max = self.get_max("BinningVertical")
                if v_min <= vertical_bin_factor <= v_max:
                    self.set_value("BinningVertical", vertical_bin_factor)
                    print(f"Vertical binning set to {vertical_bin_factor}x")
                else:
                    raise CameraConfigurationError(f"Vertical binning factor {vertical_bin_factor} out of range [{v_min}, {v_max}]")
            else:
                raise CameraConfigurationError("Vertical binning not supported.")

            # if self.has_attribute("BinningHorizontalMode"):
            #     self.set_entry("BinningHorizontalMode", mode)
            #     print(f"Horizontal binning mode set to {mode}")
            # if self.has_attribute("BinningVerticalMode"):
            #     self.set_entry("BinningVerticalMode", mode)
            #     print(f"Vertical binning mode set to {mode}")

            self._revoke_buffers()
            self._setup_buffers()
            self._preallocate_conversion()

            return True

        except CameraConfigurationError as e:
            print(f"Configuration error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error setting binning: {e}")
            return False

    @property
    def status(self):
        self.nodemap.FindNode("DeviceSelector").SetValue(0)
        value = self.nodemap.FindNode("GevDeviceCurrentControlMode").CurrentEntry().SymbolicValue()
        return value

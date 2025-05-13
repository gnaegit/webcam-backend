import ids_peak.ids_peak as idsp
import ids_peak_afl.ids_peak_afl as idsp_afl
import ids_peak_ipl.ids_peak_ipl as idsp_ipl
import ids_peak.ids_peak_ipl_extension as idsp_extension
from typing import *
import traceback

from src.camera import CameraIDS

class AutoFeatureManager:

    def __init__(self, camera: CameraIDS):
        if not self.initialize_afl():
            return
        
        self._afm = idsp_afl.Manager(camera.get_nodemap())

        self._auto_exposure: Literal['on', 'off', 'once'] = 'off'
        self._auto_gain: Literal['on', 'off', 'once'] = 'off'
        self._auto_white_balance: Literal['on', 'off', 'once'] = 'off'

        self._initialize_controller()

    def __call__(self, image):
        self._afm.Process(image)

    def calibrate(self, exposure: bool=True, gain: bool=True, white_balance: bool=True):
        if exposure:
            self.auto_exposure = 'once'
        if gain:
            self.auto_gain = 'once'
        if white_balance:
            self.auto_white_balance = 'once'
        
    @property
    def auto_exposure(self):
        return self._auto_exposure
    
    @auto_exposure.setter
    def auto_exposure(self, value: Literal['on', 'off', 'once']):
        assert value in ['on', 'off', 'once'], f'Invalid auto mode: {value}\nChoose from: on, off, once'
        self._auto_exposure = value
        mode = self._get_auto_mode(value)
        if self._brightness_ctrl is not None:
            self._brightness_ctrl.BrightnessComponentSetMode(idsp_afl.PEAK_AFL_CONTROLLER_BRIGHTNESS_COMPONENT_EXPOSURE, mode)

    @property
    def auto_gain(self):
        return self._auto_gain
    
    @auto_gain.setter
    def auto_gain(self, value: Literal['on', 'off', 'once']):
        assert value in ['on', 'off', 'once'], f'Invalid auto mode: {value}\nChoose from: on, off, once'
        self._auto_gain = value
        mode = self._get_auto_mode(value)
        if self._brightness_ctrl is not None:
            self._brightness_ctrl.BrightnessComponentSetMode(idsp_afl.PEAK_AFL_CONTROLLER_BRIGHTNESS_COMPONENT_GAIN, mode)

    @property
    def auto_white_balance(self):
        return self._auto_white_balance

    @auto_white_balance.setter
    def auto_white_balance(self, value: Literal['on', 'off', 'once']):
        assert value in ['on', 'off', 'once'], f'Invalid auto mode: {value}\nChoose from: on, off, once'
        self._auto_white_balance = value
        mode = self._get_auto_mode(value)
        if self._white_balance_ctrl is not None:
            self._white_balance_ctrl.SetMode(mode)

    def initialize_afl(self):
        try:
            idsp_afl.Library.Init()
        except Exception as e:
            print("ERROR: Failed to initialize AFL library")
            traceback.print_exc()
            return False
        return True
    
    def _initialize_controller(self):
        try:
            self._brightness_ctrl = self._afm.CreateController(idsp_afl.PEAK_AFL_CONTROLLER_TYPE_BRIGHTNESS)
            self._afm.AddController(self._brightness_ctrl)
        except Exception as e:
            self._brightness_ctrl = None
            print("ERROR: Failed to create exposure controller")
            traceback.print_exc()
        try:
            self._white_balance_ctrl = self._afm.CreateController(idsp_afl.PEAK_AFL_CONTROLLER_TYPE_WHITE_BALANCE)
            self._afm.AddController(self._white_balance_ctrl)
        except Exception as e:
            self._white_balance_ctrl = None
            print("ERROR: Failed to create white balance controller")
            traceback.print_exc()
        
    def _get_auto_mode(self, value: Literal['on', 'off', 'once']):
        if value == 'on':
            return idsp_afl.PEAK_AFL_CONTROLLER_AUTOMODE_CONTINUOUS
        elif value == 'off':
            return idsp_afl.PEAK_AFL_CONTROLLER_AUTOMODE_OFF
        elif value == 'once':
            return idsp_afl.PEAK_AFL_CONTROLLER_AUTOMODE_ONCE
        else:
            raise ValueError(f'Invalid auto mode: {value}\nChoose from: on, off, once')
        
    def __str__(self) -> str:
        return f"Auto Feature Manager:\nAuto Exposure: {self.auto_exposure}\nAuto Gain: {self.auto_gain}\nAuto White Balance: {self.auto_white_balance}"

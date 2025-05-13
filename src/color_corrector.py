import importlib.resources as resources
import threading
import numpy as np

import ids_peak.ids_peak as idsp
import ids_peak_ipl.ids_peak_ipl as idsp_ipl
import ids_peak.ids_peak_ipl_extension as idsp_extension

from src.camera import CameraIDS
    
HQ_CORRECTION_MATRIX =  np.array(
    [
        [1.7813, -0.5898, -0.1875],
        [-0.4531, 1.8555, -0.3984],
        [0.0, -0.5625, 1.5664]
    ]
)

class ColorCorrector:

    def __init__(self, correction_matrix: np.ndarray=HQ_CORRECTION_MATRIX):
        self._color_corrector = idsp_ipl.ColorCorrector()

        self.set_correction_matrix(correction_matrix)
        self.correction_matrix = correction_matrix
        self.enabled = True

    def __call__(self, image) -> None:
        if self.enabled:
            self._color_corrector.ProcessInPlace(image)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_correction_matrix(self, correction_matrix: np.ndarray) -> None:

        gain_0_0 = correction_matrix[0, 0]
        gain_0_1 = correction_matrix[0, 1]
        gain_0_2 = correction_matrix[0, 2]
        gain_1_0 = correction_matrix[1, 0]
        gain_1_1 = correction_matrix[1, 1]
        gain_1_2 = correction_matrix[1, 2]
        gain_2_0 = correction_matrix[2, 0]
        gain_2_1 = correction_matrix[2, 1]
        gain_2_2 = correction_matrix[2, 2]
                   
        color_correction_factors = idsp_ipl.ColorCorrectionFactors(gain_0_0, gain_0_1, gain_0_2,
                                                                        gain_1_0, gain_1_1, gain_1_2,
                                                                        gain_2_0, gain_2_1, gain_2_2)
        
        self._color_corrector.SetColorCorrectionFactors(color_correction_factors)

    
    def __str__(self) -> str:
        return f"IDS Peak IPL Color Correction:\nEnabled: {self.enabled}\nCorrection Matrix:\n{self.correction_matrix}"
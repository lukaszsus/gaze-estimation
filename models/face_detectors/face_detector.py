import numpy as np
from abc import ABC, abstractmethod


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image) -> np.ndarray:
        pass
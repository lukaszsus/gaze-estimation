import numpy as np
from abc import ABC, abstractmethod


class LandmarksDetector(ABC):
    @abstractmethod
    def detect(self, image, faces) -> np.ndarray:
        pass
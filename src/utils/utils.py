"""Model loading utilities for Batch Plate Detection."""

import os

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.logger import LoggerManager


logger = LoggerManager()

drive_yolov8_weight = "@yolo_v8_plate.pt"
drive_yolov11_weight = "@yolo_v11_plate.pt"


class ModelLoader:
    """Load the selected plate detection model."""

    def __init__(self, config: dict):
        self.config = config
        self.application = Application()
        self.models = {}

    def download_weights(
        self,
        url,
        weight_name,
    ):
        """Download model weights when they are not in storage."""

        weight_path = f"/storage/{weight_name}"

        if not os.path.exists(weight_path):
            result = Download.download_from_drive(
                url,
                weight_path,
            )

            if result is None:
                logger.error(
                    "BatchPlateDetection - "
                    f"Model ({weight_name}) "
                    "download failed!"
                )

        return weight_path

    def load_models(self):
        """Load model and device from package configuration."""

        weight = self.application.get_param(
            config=self.config,
            name="Weights",
        )

        config_device = self.application.get_param(
            config=self.config,
            name="ConfigDevice",
        )

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

        if weight not in (
            "yolo_v8_plate.pt",
            "yolo_v11_plate.pt",
        ):
            raise ValueError(
                f"Unsupported weight: {weight}. "
                "Use yolo_v8_plate.pt or "
                "yolo_v11_plate.pt."
            )

        device = select_device(
            "cuda:0"
            if (
                config_device == "GPU"
                and torch.cuda.is_available()
            )
            else "cpu"
        )

        if weight == "yolo_v8_plate.pt":
            weight_url = drive_yolov8_weight
        else:
            weight_url = drive_yolov11_weight

        weight_path = self.download_weights(
            url=weight_url,
            weight_name=weight,
        )

        model = YOLO(weight_path)

        if (
            config_device == "GPU"
            and torch.cuda.is_available()
        ):
            half = self.application.get_param(
                config=self.config,
                name="Half",
            )

            if half:
                model.fuse()
                model = model.to(device).half()
            else:
                model = model.to(device)

        else:
            model = model.to(device)

        self.models["model"] = model
        self.models["device"] = device

        return self.models
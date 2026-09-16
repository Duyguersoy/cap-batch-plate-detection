"""Model loading utilities for Batch Plate Detection."""

import os
import sys

import torch
from ultralytics import YOLO

sys.path.append(
    "/opt/project/capsules/Yolov5/src/lib/yolov5"
)

from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.download import Download
from sdks.novavision.src.base.logger import LoggerManager


logger = LoggerManager()

drive_plate_weight = "@yolo_v5_plate.pt"
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

        if weight == "yolo_v5_plate.pt":
            from capsules.Yolov5.src.lib.yolov5.models.common import (
                DetectMultiBackend,
            )
            from capsules.Yolov5.src.lib.yolov5.utils.torch_utils import (
                select_device,
            )

            device = select_device(
                "cuda:0"
                if (
                    config_device == "GPU"
                    and torch.cuda.is_available()
                )
                else "cpu"
            )

            weight_path = self.download_weights(
                url=drive_plate_weight,
                weight_name=weight,
            )

            if (
                config_device == "GPU"
                and torch.cuda.is_available()
            ):
                half = self.application.get_param(
                    config=self.config,
                    name="Half",
                )

                model = DetectMultiBackend(
                    weight_path,
                    device=device,
                    fp16=bool(half),
                )
            else:
                model = DetectMultiBackend(
                    weight_path,
                    device=device,
                    fp16=False,
                )

        elif weight in (
            "yolo_v8_plate.pt",
            "yolo_v11_plate.pt",
        ):
            from ultralytics.utils.torch_utils import (
                select_device,
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

        else:
            raise ValueError(
                f"Unsupported weight: {weight}"
            )

        self.models["model"] = model
        self.models["device"] = device

        return self.models
"""Batch plate detection executor for NovaVision."""

import os
import sys
import uuid

import numpy as np

sys.path.append("/opt/project/capsules/Yolov5/src/lib/yolov5")
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../",
    )
)

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.helper.executor import Executor

from capsules.Yolov5.src.classes.yolov5_detect import Yolov5Detect

if __package__:
    from ..models.PackageModel import PackageModel, Detection
    from ..utils.response import build_response
    from ..utils.utils import ModelLoader
else:
    from capsules.BatchPlateDetection.src.models.PackageModel import (
        PackageModel,
        Detection,
    )
    from capsules.BatchPlateDetection.src.utils.response import (
        build_response,
    )
    from capsules.BatchPlateDetection.src.utils.utils import (
        ModelLoader,
    )


class BatchPlateDetection(Capsule):
    """Run plate detection independently for every input image."""

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

        self.request.model = PackageModel(**self.request.data)

        self.images = self.request.get_param("inputImage")

        self.conf_thres = self.request.get_param(
            "ConfidentThreshold"
        )

        self.iou_thres = self.request.get_param(
            "IOUThreshold"
        )

        self.backend = self.request.get_param("Weights")

        self.select_device = self.bootstrap.get("device")
        self.weight = self.bootstrap.get("model")

        self.detection_groups = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return ModelLoader(
            config=config
        ).load_models()

    def process_output(
        self,
        output,
        names,
        img_uid,
    ):
        detection_list = []

        if isinstance(output, np.ndarray):
            if (
                output.ndim == 1
                and output.shape[0] >= 6
            ):
                output = output.reshape(1, -1)

            elif (
                output.ndim == 2
                and output.shape[0] == 0
            ):
                return detection_list

            elif output.ndim != 2:
                raise ValueError(
                    "Unexpected output shape: "
                    f"{output.shape}"
                )

        elif (
            isinstance(output, list)
            and len(output) == 0
        ):
            return detection_list

        for plate in output:
            plate_bbox = BoundingBox(
                left=plate[0],
                top=plate[1],
                width=plate[2] - plate[0],
                height=plate[3] - plate[1],
            )

            detection = Detection(
                boundingBox=plate_bbox,
                confidence=plate[4],
                classLabel=names[int(plate[5])],
                classId=-int(plate[5]),
                imgUID=img_uid,
                UUID=str(uuid.uuid4()),
                source=img_uid,
            )

            detection_list.append(detection)

        return detection_list

    def plate_inference(self, image):
        if self.backend == "yolo_v5_plate.pt":
            output, names, _ = Yolov5Detect(
                model=self.weight,
                source=image.value,
                device=str(self.select_device),
                conf_thres=float(self.conf_thres),
                iou_thres=float(self.iou_thres),
            ).run()

            output = output[0].cpu().numpy()

        elif self.backend in (
            "yolo_v8_plate.pt",
            "yolo_v11_plate.pt",
        ):
            results = self.weight.predict(
                image.value,
                device=str(self.select_device),
                conf=float(self.conf_thres),
                iou=float(self.iou_thres),
            )

            output = []

            for box in results[0].boxes:
                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                conf = float(box.conf)
                cls_id = int(box.cls)

                output.append(
                    xyxy + [conf, cls_id]
                )

            names = self.weight.names

        else:
            raise ValueError(
                "Unsupported backend: "
                f"{self.backend}"
            )

        return self.process_output(
            output=output,
            names=names,
            img_uid=image.uID,
        )

    def run(self):
        if isinstance(self.images, list):
            image_items = self.images
        else:
            image_items = [self.images]

        self.detection_groups = []

        for image_item in image_items:
            image = Image.get_frame(
                img=image_item,
                redis_db=self.redis_db,
            )

            if image is None:
                self.detection_groups.append([])
                continue

            detections = self.plate_inference(
                image=image
            )

            self.detection_groups.append(
                detections
            )

        return build_response(
            context=self
        )


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
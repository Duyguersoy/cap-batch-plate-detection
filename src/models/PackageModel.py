from typing import List, Literal, Optional, Union

from pydantic import Field, validator

from sdks.novavision.src.base.model import (
    Config,
    Configs,
    Detection as BaseDetection,
    Image,
    Input,
    Inputs,
    Output,
    Outputs,
    Package,
    Request,
    Response,
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        input_value = values.get("value")

        if isinstance(input_value, list):
            return "list"

        return "object"

    class Config:
        title = "Image"


class BatchPlateDetectionInputs(Inputs):
    inputImage: InputImage


class Detection(BaseDetection):
    UUID: str
    source: str
    imgUID: str


class OutputDetections(Output):
    name: Literal["outputDetections"] = "outputDetections"
    value: List[List[Detection]]
    type: Literal["list"] = "list"

    class Config:
        title = "Grouped Detections"


class BatchPlateDetectionOutputs(Outputs):
    outputDetections: OutputDetections


class PlateYoloV5Weight(Config):
    name: Literal["PlateYoloV5Weight"] = "PlateYoloV5Weight"
    value: Literal["yolo_v5_plate.pt"] = "yolo_v5_plate.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Plate YOLOv5"


class PlateYoloV8Weight(Config):
    name: Literal["PlateYoloV8Weight"] = "PlateYoloV8Weight"
    value: Literal["yolo_v8_plate.pt"] = "yolo_v8_plate.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Plate YOLOv8"


class PlateYoloV11Weight(Config):
    name: Literal["PlateYoloV11Weight"] = "PlateYoloV11Weight"
    value: Literal["yolo_v11_plate.pt"] = "yolo_v11_plate.pt"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Plate YOLOv11"


class ConfigWeights(Config):
    name: Literal["Weights"] = "Weights"

    value: Union[
        PlateYoloV5Weight,
        PlateYoloV8Weight,
        PlateYoloV11Weight,
    ]

    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Weights"


class ConfigHalfTrue(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigHalfFalse(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigHalf(Config):
    name: Literal["Half"] = "Half"
    value: Union[ConfigHalfTrue, ConfigHalfFalse]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Half"


class ConfigDeviceGPU(Config):
    name: Literal["ConfigDeviceGPU"] = "ConfigDeviceGPU"
    configHalf: ConfigHalf
    value: Literal["GPU"] = "GPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "GPU"


class ConfigDeviceCPU(Config):
    name: Literal["ConfigDeviceCPU"] = "ConfigDeviceCPU"
    value: Literal["CPU"] = "CPU"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CPU"


class ConfigDevice(Config):
    name: Literal["ConfigDevice"] = "ConfigDevice"

    value: Union[
        ConfigDeviceCPU,
        ConfigDeviceGPU,
    ]

    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Device"


class ConfigConfidentThreshold(Config):
    name: Literal["ConfidentThreshold"] = "ConfidentThreshold"

    value: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
    )

    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Confidence Threshold"


class ConfigIOUThreshold(Config):
    name: Literal["IOUThreshold"] = "IOUThreshold"

    value: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
    )

    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "IoU Threshold"


class BatchPlateDetectionConfigs(Configs):
    configWeights: ConfigWeights
    configDevice: ConfigDevice
    configConfidentThreshold: ConfigConfidentThreshold
    configIOUThreshold: ConfigIOUThreshold


class BatchPlateDetectionRequest(Request):
    inputs: Optional[BatchPlateDetectionInputs]
    configs: BatchPlateDetectionConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class BatchPlateDetectionResponse(Response):
    outputs: BatchPlateDetectionOutputs


class BatchPlateDetectionExecutor(Config):
    name: Literal["BatchPlateDetection"] = "BatchPlateDetection"

    value: Union[
        BatchPlateDetectionRequest,
        BatchPlateDetectionResponse,
    ]

    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Batch Plate Detection"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: BatchPlateDetectionExecutor
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["BatchPlateDetection"] = "BatchPlateDetection"
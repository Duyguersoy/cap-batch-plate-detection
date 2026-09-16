"""Response builder for Batch Plate Detection."""

from sdks.novavision.src.helper.package import PackageHelper

if __package__:
    from ..models.PackageModel import (
        BatchPlateDetectionExecutor,
        BatchPlateDetectionOutputs,
        BatchPlateDetectionResponse,
        ConfigExecutor,
        OutputDetections,
        PackageConfigs,
        PackageModel,
    )
else:
    from capsules.BatchPlateDetection.src.models.PackageModel import (
        BatchPlateDetectionExecutor,
        BatchPlateDetectionOutputs,
        BatchPlateDetectionResponse,
        ConfigExecutor,
        OutputDetections,
        PackageConfigs,
        PackageModel,
    )


def build_response(context):
    """Build grouped detection response."""

    output_detections = OutputDetections(
        value=context.detection_groups
    )

    outputs = BatchPlateDetectionOutputs(
        outputDetections=output_detections
    )

    response = BatchPlateDetectionResponse(
        outputs=outputs
    )

    batch_executor = BatchPlateDetectionExecutor(
        value=response
    )

    executor = ConfigExecutor(
        value=batch_executor
    )

    package_configs = PackageConfigs(
        executor=executor
    )

    package = PackageHelper(
        packageModel=PackageModel,
        packageConfigs=package_configs,
    )

    return package.build_model(context)
import setuptools


setuptools.setup(
    name="batch-plate-detection",
    version="0.0.1",
    author="NovaVision AI",
    author_email="info@novavision.ai",
    description="Batch Plate Detection capsule for NovaVision",
    url="https://github.com/Duyguersoy/cap-batch-plate-detection",
    license="MIT",

    install_requires=[
        "numpy",
        "ultralytics",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=[
        "novavision.package",
        "novavision.package.executors",
        "novavision.package.models",
        "novavision.package.utils",
    ],

    package_dir={
        "novavision.package": "src",
    },

    python_requires=">=3.7",
)
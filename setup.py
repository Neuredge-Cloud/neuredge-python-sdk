from setuptools import setup, find_packages

setup(
    name="neuredge_sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "typing_extensions>=4.0.0",
    ],
    python_requires=">=3.8",
    description="Python SDK for the Neuredge AI Platform",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

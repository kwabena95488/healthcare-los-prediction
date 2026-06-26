"""
Setup configuration for the Healthcare Length of Stay Prediction package.

This setup.py file enables the project to be installed as a Python package,
making it easier to import modules and deploy the application.

@version 1.0.0
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="healthcare-length-of-stay",
    version="1.0.0",
    author="Healthcare Analytics Team",
    author_email="team@healthcareanalytics.com",
    description="Machine learning models to predict hospital length of stay",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/healthcare-analytics/length-of-stay",
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "healthcare-train=scripts.train_model:main",
            "healthcare-predict=scripts.make_predictions:main",
            "healthcare-evaluate=scripts.evaluate_model:main",
        ],
    },
    
    # Additional package data
    package_data={
        "": ["*.yaml", "*.yml", "*.txt", "*.md"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/healthcare-analytics/length-of-stay/issues",
        "Source": "https://github.com/healthcare-analytics/length-of-stay",
        "Documentation": "https://healthcare-analytics.github.io/length-of-stay/",
    },
    
    # Keywords for discovery
    keywords="healthcare, machine-learning, prediction, hospital, length-of-stay, medical",
) 
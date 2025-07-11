"""
Setup script for ML101 package
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml101-algorithms",
    version="0.1.0",
    author="hustcalm",
    author_email="hustcalm@gmail.com",
    description="Educational implementation of classical machine learning algorithms from scratch - CoPiloted with Claude Sonnet 4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hustcalm/ML101",
    project_urls={
        "Bug Reports": "https://github.com/hustcalm/ML101/issues",
        "Source": "https://github.com/hustcalm/ML101",
        "Documentation": "https://github.com/hustcalm/ML101/tree/main/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.20",
            "coverage>=6.0",
        ],
        "examples": [
            "matplotlib>=3.5",
            "scikit-learn>=1.0",
            "seaborn>=0.11",
        ],
    },
    include_package_data=True,
    package_data={
        "ml101": ["py.typed"],
    },
    keywords="machine learning, algorithms, education, data science, artificial intelligence",
    zip_safe=False,
)
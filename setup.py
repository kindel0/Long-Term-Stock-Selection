"""Setup script for stock-trading-system."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stock-trading-system",
    version="1.0.0",
    author="",
    description="ML-based stock selection with IBKR integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "yfinance>=0.2",
        "pandas-datareader>=0.10",
        "click>=8.1",
        "pyarrow>=12.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8",
        "joblib>=1.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
        "ibkr": [
            "ib_insync>=0.9.86",
        ],
        "reports": [
            "reportlab>=4.0",
            "plotly>=5.15",
        ],
        "dashboard": [
            "streamlit>=1.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-trading=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

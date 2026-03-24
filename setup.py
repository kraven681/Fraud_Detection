from setuptools import setup, find_packages

setup(
    name="fraud-detection",
    version="1.0.0",
    description="Production fraud detection pipeline using PaySim financial transaction data",
    author="Your Name",
    author_email="you@example.com",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.6.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "httpx>=0.26.0",
            "black>=24.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
        "torch": ["torch>=2.1.0"],
    },
    entry_points={
        "console_scripts": [
            "fraud-train=models.train:main",
            "fraud-eval=evaluation.metrics:main",
            "fraud-serve=api.app:serve",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

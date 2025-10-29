from setuptools import setup, find_packages

setup(
    name="knn-regression-advanced",
    version="1.0.0",
    description="Advanced KNN Regression with multiple algorithms comparison",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "optuna>=3.0.0",
        "mlflow>=2.0.0",
        "joblib>=1.1.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.8",
)
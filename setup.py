from setuptools import setup, find_packages

# Figure out requirements from requirements.txt. This was generated with pipreqs.
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="kpflow",
    version="1.0.0",
    description="K-PFlow: An Operator Perspective on Gradient Descent Learning in Recurrent Models.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="James Hazelden",
    url="https://github.com/meeree/kpflow",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    python_requires=">=3.2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)

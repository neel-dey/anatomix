from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="anatomix",
    version="0.1.0",
    description="A general-purpose feature extractor for 3D volumes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Neel Dey",
    author_email="dey@csail.mit.edu",
    url="https://github.com/neel-dey/anatomix",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)

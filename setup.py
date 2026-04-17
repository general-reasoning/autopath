import logging
import os
import setuptools


def readlines(fname: str) -> list:
    """Load a local filename."""
    lines = open(os.path.join(os.path.dirname(__file__), fname)).readlines()
    return [line.strip() for line in lines]

def get_requirements():
    lines = readlines('requirements.txt')
    requirements = [line for line in lines if line and not line.startswith(('-', '#'))]
    return requirements
    

setuptools.setup(
    name="autopath",
    version="0.1.2",
    author="Dmitry Karpeyev",
    author_email="dmitry.karpeyev@gmail.com",
    description="Automated Pathology inference models",
    packages=setuptools.find_packages(exclude=("test,")),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={'console_scripts': [
        'autopath_builddocs=autopath.builddocs:main',
    ]},
    python_requires='>=3.12',
    install_requires=get_requirements(),
)

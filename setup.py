from setuptools import find_packages, setup

setup(
    name='vbkf',
    version='0.1.0',  # always SemVer2
    packages=find_packages(),
    description="informative description",
    python_requires=">=3.6",
    install_requires=[
        'numpy',  # MIT
        'scipy',  # MIT
        'pytest',  # MIT
    ],
)

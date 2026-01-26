from setuptools import setup, find_packages

setup(
    name='bevy_zeroverse_dataloader',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch',
        'safetensors',
    ],
    entry_points={
        'console_scripts': [
            'run_tests = test:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

from setuptools import setup, find_packages

# Read the content of the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the content of the requirements.txt file
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="gift",
    version="0.0.1",
    description="GIFT: Generative Interpretable Fine-Tuning",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/savadikarc/gift",
    author="Chinmay Savadikar",
    author_email="csavadi@ncsu.edu",
    license="MIT License",
    packages=find_packages(include=['gift', 'gift.*']),
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)

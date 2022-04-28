from setuptools import setup, find_packages


def get_version():
    with open('version.txt', 'r') as f:
        version = f.readline().strip()
    return version


def get_readme():
    with open('README.md', 'r') as f:
        readme = f.read()
    return readme


def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f.readlines()]
    return requirements


setup(
    name='ddg',
    version=get_version(),
    description='DDG: This repo contains the code for our IJCAI 2022 paper: Dynamic Domain Generalization.',
    author='MetaVisionLab',
    license='GNU General Public License v3.0',
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/MetaVisionLab/DDG',
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=[
        'Domain Generalization'
    ]
)

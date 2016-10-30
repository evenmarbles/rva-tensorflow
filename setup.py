from setuptools import find_packages
from setuptools import setup


with open("requirements.txt", "r") as f:
    install_requires = [x.strip() for x in f.readlines() if not
                        x.startswith(("http", "git"))]

setup(
    name='trainer',
    version='1.0.0',
    description='Recurrent Visual Attention in TensorFlow',
    author='Astrid Jackson',
    author_email='ajackons@eecs.ucf.edu',
    packages=find_packages(),
    install_requires=install_requires,
    package_data={
        'trainer': ['configs/*.cfg'],
    }
)

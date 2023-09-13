#Ask TA what is the goal of this file

from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='DeepFakeDetection',
      version="0.0.1",
      description="Deepfake detection model (api_pred)",
      #install_requires=requirements, #comment to make the docker build faster
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)

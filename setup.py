from setuptools import setup, find_packages

setup(
  name = 'VisualNexus',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'VisualNexus - Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  url = 'https://github.com/kyegomez/VisualNexus',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
    'vision'
  ],
  install_requires=[
    'torch>=1.7.0',
    # Base-----------------------------------
    'matplotlib>=3.2.2',
    'opencv-python>=4.6.0',
    'Pillow>=7.1.2',
    'PyYAML>=5.3.1',
    'requests>=2.23.0',
    'scipy>=1.4.1',
    'torchvision>=0.8.1',
    'tqdm>=4.64.0',
    'pandas>=1.1.4',
    'seaborn>=0.11.0',
    'gradio==3.35.2',
    # Ultralytics-----------------------------------
    'ultralytics == 8.0.120',
    'metaseq'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
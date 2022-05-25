from setuptools import setup, find_packages

setup(
  name = 'video-diffusion-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.4.1',
  license='MIT',
  description = 'Video Diffusion - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/video-diffusion-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'denoising diffusion probabilistic models',
    'video generation'
  ],
  install_requires=[
    'einops>=0.4',
    'einops-exts',
    'rotary-embedding-torch',
    'torch>=1.10',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

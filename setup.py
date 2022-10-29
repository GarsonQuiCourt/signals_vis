from distutils.core import setup
import setuptools

def readme():
    with open(r'README.txt') as f:
        README = f.read()
    return README

setup(
    name = 'signals_vis', ###################################
    packages = setuptools.find_packages(),

    version = '3.2',
    url = 'https://github.com/GarsonQuiCourt/signals_vis', #Github link
    download_url = 'https://github.com/GarsonQuiCourt/signals_vis/archive/refs/tags/0.tar.gz',#No need to change
    keywords = ['get_url', 'download'],
    install_requires=[
          'beautifulsoup4',
          'requests',
          'selenium',
          
      ],
    include_package_data=True,
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    ],
)
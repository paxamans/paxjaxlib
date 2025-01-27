from setuptools import setup, find_packages

setup(
    name="paxjaxlib",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'numpy',
        'matplotlib'
    ],
    author="Igor Beketov",
    author_email="paxamans@gmail.com",
    description="A simple neural network implementation using JAX, made for educational purposes",
    keywords="neural network, deep learning, jax",
    url="https://github.com/paxamans/paxjaxlib",
)

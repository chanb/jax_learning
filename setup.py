from setuptools import setup, find_packages

setup(
    name="jax_learning",
    description="Learning algorithms using JAX",
    version="0.1",
    python_requires=">=3.10",
    install_requires=[
        "black==22.3.0",
        "equinox",
        "jax",
        "jaxlib",
        "jupyter",
        "optax",
        "wandb",
    ],
    extras_requires={
        "all": ["gym==0.26.0"]
    },
    packages=find_packages(),
    include_package_data=True,
)

# JAX Learning
This repository provides learning algorithms using JAX.

# Installation
## Prerequisites
- Python 3.10

You may simply install this package using pip.
```
cd ${JAX_LEARNING_PATH}
pip install -e .
```

Running `wandb` local instance:
1. Install [Docker](https://www.docker.com/)
2. Install [Weights & Biases Local](https://github.com/wandb/local) docker image: `docker pull wandb/local`
3. Start docker container `docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local`
4. You can now reach the dashboard via http://localhost:8080
5. If you are setting this up for the first time, create a license from [Deployer](https://deploy.wandb.ai/) and add it to the local instance. Create an account on the local instance and copy the API key.
6. Login to the local instance: `wandb login --host=http://localhost:8080`. Use the copied API key from step 5 if necessary.

### Reference
- Docker: https://www.docker.com/
- Weights & Biases Local: https://github.com/wandb/local

## Citation
Please consider citing this repository if you use/extend this codebase in your work:
```
@misc{jax_learning,
  author = {Chan, Bryan},
  title = {JAX Learning},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chanb/jax_learning}},
}
```


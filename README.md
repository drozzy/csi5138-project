# csi5138-project

## Setup

Create new environment (use "envrionment_cpu.yml" if you don't have a GPU):

	conda env create -f environment_gpu.yml
	conda activate csi5138-project

Update existing project environment:

	conda env update -f environment.yml  --prune

## Run

The project notebooks are meant to be run in jupyter lab:

	jupyter lab
    
## Other

To register an environment in the jupyter notebook, first switch to that env and then run:

    conda activate csi5138-project
    python -m ipykernel install --user --name=csi5138-project

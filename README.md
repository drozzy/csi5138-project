# csi5138-project

## Setup

Create new environment:

	conda env create -f environment.yml
	conda activate proj-5138

Update existing project environment:

	conda env update -f environment.yml  --prune

## Run

The project notebooks are meant to be run in jupyter lab:

	jupyter lab
    
## Other

To register an environment in the jupyter notebook, first switch to that env and then run:

    conda activate proj-5138
    python -m ipykernel install --user --name=proj-5138

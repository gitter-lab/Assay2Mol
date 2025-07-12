# Assay2Mol: Large Language Model-based Drug Design Using BioAssay Context

## Data preparation
Download the BioAssay embedding files `index.faiss` and `index.pkl` from [Zenodo](https://doi.org/10.5281/zenodo.15867214) and put both of them in the directory `BioAssay_vectorbase/`.

For the CrossDocked test set protein files, download `test_set.zip` from [Zenodo](https://doi.org/10.5281/zenodo.15867214) and put the extracted contents in the directory `docking/test_set/`.

## Running the code
First, install the dependencies from `requirements.txt`, preferably in a new virtual environment.

Then, fill in your API key for the LLM service (e.g. OpenAI, DeepSeek). If you want to run it on your local server, consider using [vLLM](https://github.com/vllm-project/vllm) to deploy a local LLM with the OpenAI API.

Then, run `python CrossDock/CrossDock_exp.py`. 

For the hERG experiment, run `python hERG/hERG_exp.py`. You can also change the hERG description to another target of interest to increase molecules' specificity. You need to change the prompt accordingly.

## Running docking
We use the same docking process as [TargetDiff](https://github.com/guanjq/targetdiff). Please check the TargetDiff repo for the environment requirement. The code in `docking/` is copied from TargetDiff under its [MIT License](https://github.com/guanjq/targetdiff/blob/142f1eb7178480d435fe0b8cb95a99beb48997c7/LICIENCE) and modified for Assay2Mol.

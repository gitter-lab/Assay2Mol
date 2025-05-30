# Assay2Mol: Large Language Model-based Drug Design Using BioAssay Context

## Data preparation
Download the BioAssay embedding we uploaded to [Zenodo]() and put it in BioAssay_vectorbase/.

For CrossDocked test set protein files, also download them from [Zenodo]() and put them in docking/test_set/.
## Running the code
First, install the dependencies from `requirements.txt`.

Then, fill in your API key for LLM service (e.g. GPT, DeepSeek). If you want to run it on your local server, consider using [vLLM](https://github.com/vllm-project/vllm) to deploy the local LLM with OpenAI API.

Then, run `python CrossDock/CrossDock_exp.py`. 

For hERG experiment, simply run `python hERG/hERG_exp.py`. You can also change the hERG description to other target of interest, to increase molecules' specificity. You need to change the prompt accordingly.

## Running docking
We use the same docking process as [TargetDiff](https://github.com/guanjq/targetdiff). Please check TargetDiff repo for environment requirement. The code in docking/ is copied from TargetDiff and modified for Assay2Mol.

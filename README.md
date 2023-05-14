# PaulGPT

**See [this presentation](https://docs.google.com/presentation/d/11ky3JXxKyHRsWvc_i89iektB2Y_NrnHz6gKzhZJwNXs/view#slide=id.p) where I walk through the iterations that led up to the latest version (as of May '23).**


## Overview

Right now, this project is a WIP decoder-only transformer language model build from scratch in pytorch, and trained on simple wikipedia. Written by [Paul Fornia](https://www.paulfornia.com).

In current form, closely follows [Andrei Karpathy's great youtube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), which is based on the architecture from the [*Attention is all you need* paper](https://arxiv.org/abs/1706.03762).

End-state goal: a demo-friendly prompt/response GPT chatbot, which is finetuned to always say great things about me (Paul).

## Code/Usage
* `./src/paul_gpt/attention_decoder.py` contains the pytorch class of the model architecture `AttentionModule`.
* `./src/paul_gpt/gpt_utils.py` contains several helper functions.
* See `./src/train_decoder.ipynb` for sample notebook on how to use all these functions.
* Install package from github w/ `pip install --force-reinstall 'https://github.com/pfornia/paul-gpt/blob/master/dist/paul_gpt-0.0.1-py3-none-any.whl?raw=true'` (Package not registered w/ pypi).
* To compile pip package w/ changes, `python3 -m build` (from this library, must be same location as `pyproject.toml`). This will create a new `whl` file. Then you can pip install locally, or push changes to github and pip install from there.

## High-level TODOs/Next Steps as I understand them:
* Add encoder stage to architecture
* Find prompt/response training data
* Fine tune with RLHF to be highly pro-Paul. I may cheat and use GPT-3.5 API to act as the "human" feedback.
* Host demo UI on my personal site


## Detailed, short-term next steps
* More easily experiment with hyperparams without building and pushing.
* Train my own tokenizer on simple wiki? This would reduce unhelpful tokens, e.g., those used in prgrogramming languages.
* Supervised fine tuning?? Good public Q&A data sets?
* RLHF
* Text corpus improvement ideas? Too many lists in wikipedia (how to fix??); Add dataset of news articles?
* Multi-head into a 4th dimension??



## Resources/Bibliography
* [Attention is all you need](https://arxiv.org/abs/1706.03762) (Transformers)
* [Deep Residual Learning...](https://arxiv.org/abs/1512.03385) (Skip Connections)
* [Dropout: A simple way...](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

* [Pip packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
  * To build package:
  * (`-m pip install --upgrade build`)
  * `python3 -m build` (from same dir as pyproject.toml)

  * Then install via
    * `pip install --force-reinstall 'https://github.com/pfornia/paul-gpt/blob/master/dist/paul_gpt-0.0.1-py3-none-any.whl?raw=true'`


# PaulGPT

## Overview

Right now, this project is a WIP decoder-only transformer build from scratch in pytorch, and trained on simple wikipedia. Written by [Paul Fornia](www.paulfornia.com).

In current form, closely follows [Andrei Karpathy's great youtube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), which is based on the architecture from the [*Attention is all you need* paper](https://arxiv.org/abs/1706.03762).

End-state goal: a demo-friendly prompt/response GPT chatbot, which is finetuned to always say great things about me (Paul).

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


# Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search

This repository contains the code to reproduce our results in our paper 'Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search'.

Preprint link: https://arxiv.org/abs/2411.17538

The paper studies the impact of isotropy on semantic code search performance and proposes a modified ZCA whitening post-processing technique to increase the isotropy of the embedding space and improve search performance.

### How to Run:

To create embeddings from code language models use the `create_embeddings.py` script. Required arguments:
* model <em>(codebert, codet5p, codellama)</em>
* lang <em>(ruby, javascript, go, java, python, php, r)</em>
* is_finetuned <em>(use for fine-tuned CodeBERT model)</em>

To fine-tune CodeBERT use the `fine-tune.py` script. Required arguments:
* lang <em>(ruby, javascript, go, java, python, php, r)</em>
* train_batch_size
* learning_rate
* num_train_epochs
* num_of_accumulation_steps

To evaluate the isotropy and MRR of the different model embeddings use the `eval_isotropy_mrr.py` script. The created embeddings should be stored in the `./embeddings` folder. Required arguments:

* model <em>(codebert, codet5p, codellama)</em>
* lang <em>(ruby, javascript, go, java, python, php, r)</em>
* epsilon <em>(eigenvalue regularizer for Soft-ZCA whitening, if set to 0 standard ZCA whitening)</em>
* is_finetuned <em>(use for fine-tuned CodeBERT model)</em>

### Check for package dependencies:

`numpy` `torch` `pandas` `transformers` `IsoScore` `datasets` `info_nce`

### Contributors:
Andor Diera, Lukas Galke, Ansgar Scherp

#!/bin/bash

python3 create_embeddings.py --model "codet5p" --lang "ruby"
python3 create_embeddings.py --model "codebert" --lang "ruby"
python3 create_embeddings.py --model "codellama" --lang "ruby"

python3 eval_isotropy_mrr.py --model "codet5p" --lang "ruby" # base codet5p
python3 eval_isotropy_mrr.py --model "codebert" --lang "ruby" # base codebert
#python3 eval_isotropy_mrr.py --model "codebert" --lang "ruby" --is_finetuned # finetuned codebert
python3 eval_isotropy_mrr.py --model "codellama" --lang "ruby" # base codellama

python3 eval_isotropy_mrr.py --model "codet5p" --lang "ruby" --epsilon 0.1 # codet5p with Soft ZCA whitening    

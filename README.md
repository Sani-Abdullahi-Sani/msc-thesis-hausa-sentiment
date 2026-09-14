# Investigating Cultural Grounding in Low-Resource African Language NLP: A Case Study of Hausa

Code accompanying the MSc thesis "Investigating Cultural Grounding in Low-Resource
African Language NLP" (University of the Witwatersrand, 2026), investigating whether
translation-based data augmentation helps or harms Hausa sentiment classification
compared to native-language training data, and whether transformer models trained on
native versus translated Hausa learn language-specific or universal semantic
representations.

## Repository contents

This repository contains the core pipeline used to produce the thesis's main results.
It is a curated subset of the full experimental codebase, kept intentionally minimal
for clarity and reproducibility -- exploratory analysis scripts, plotting utilities,
and intermediate result dumps generated during the research process are not included.

- data/matched/ -- Matched datasets (Native Hausa, Native English, Translated Hausa),
  each with identical train/dev/test splits (15,506 / 3,323 / 3,323 examples) produced
  via stratified sampling. See data/matched/README.md.
- tokenizers/ -- Pre-trained BPE and SentencePiece tokenizers used throughout the
  thesis, plus the Hausa training corpus.
- src/custom_tokenizers/ -- BPE / SentencePiece / morphological tokenizer wrappers
- src/data/ -- Dataset loading and preprocessing utilities
- src/models/ -- Custom transformer classifier and initialisation schemes
  (He / Generalized / Imbalanced, per Section 3.6 of the thesis)
- src/training/ -- Training loop and evaluation metrics
- src/utils/ -- Experiment tracking and visualisation helpers
- create_matched_datasets.py -- Builds the matched, stratified datasets from raw corpora
- translate_english_to_hausa.py -- English-to-Hausa translation pipeline
  (dammyogt/english-to-hausa-translation model)
- run_systematic_experiments.py -- Main experiment runner: 4 data sources x 2
  tokenisers x 3 initialisation strategies x 5 seeds (Section 3.1)
- finetune_afriberta.py -- Supplementary experiment fine-tuning a pretrained
  multilingual model (AfriBERTa) on Native vs. Translated Hausa, confirming the
  native-vs-translated degradation persists at higher model capacity (Section 5.4)
- requirements.txt -- Python dependencies

## Requirements

pip install -r requirements.txt

Developed and run with Python 3.8, PyTorch, and the HuggingFace transformers /
tokenizers libraries on a SLURM-managed GPU cluster.

## Reproducing the main results

1. Build the matched datasets (skip if using the provided data/matched/ directly):

   python create_matched_datasets.py

2. Run the systematic experiment suite (4 data sources x 2 tokenisers x 3
   initialisation strategies x 5 seeds = 120 runs):

   python run_systematic_experiments.py --dataset native --tokenizer sentencepiece --init imbalanced --seed 42

   (Repeat across the full grid of --dataset, --tokenizer, --init, and --seed
   values used in the thesis; see Section 3.1 and Table 4.1 for the full configuration.)

3. Run the supplementary AfriBERTa experiment:

   python finetune_afriberta.py --condition native --seed 42 --epochs 3
   python finetune_afriberta.py --condition translated --seed 42 --epochs 3

## Citation

If you use this code, please cite:

@mastersthesis{sani2026cultural,
  author  = {Sani, Sani Abdullahi},
  title   = {Investigating Cultural Grounding in Low-Resource African Language NLP},
  school  = {University of the Witwatersrand},
  year    = {2026}
}

## Acknowledgements

Supervised by Dr. Devon Jarvis, School of Computer Science and Applied Mathematics,
University of the Witwatersrand. This work was supported by a Google DeepMind study
grant.

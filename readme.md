# Mol-BERT
### Introduction
![17c812d2ff49c4ff06887395acdd9fa](https://user-images.githubusercontent.com/87004026/133225022-f9b6e2d2-1a23-4857-b4d8-d51e8d38b692.png)

The implementation of paper 'An Effective Molecular Representation with BERT for Molecular Property Prediction'.
Molecular property prediction is an essential task in drug discovery. Most computational approaches with deep learning
techniques either focus on designing novel molecular representation or combining with some advanced models together.
However, researchers pay fewer attention to the potential benefits in massive unlabeled molecular data (e.g., ZINC). 
This task becomes increasingly challenging owing to the limitation of the scale of labeled data. Motivated by the recent
advancements of pretrained models in natural language processing, the drug molecule can be naturally viewed as language
to some extent. In this paper, we investigate how to develop the pretrained model BERT to extract useful molecular
substructure information for molecular property prediction. We present a novel end-to-end deep learning framework,
named Mol-BERT, that combines an effective molecular representation with pretrained BERT model tailored for molecular
property prediction. Specifically, a large-scale prediction BERT model is pretrained to generate the embedding of molecular
substructures, by using four million unlabeled drug SMILES (i.e., ZINC 15 and ChEMBL 27). Then, the pretrained BERT
model can be fine-tuned on various molecular property prediction tasks. To examine the performance of our proposed
Mol-BERT, we conduct several experiments on 4 widely used molecular datasets. In comparison to the traditional and
state-of-the-art baselines, the results illustrate that our proposed Mol-BERT can outperform the current sequence-based
methods and achieve at least 2% improvement on ROC-AUC score on Tox21, SIDER, and ClinTox dataset.

### How to pre train the model
python3 BERT_training_seq100.py

### How to predict the molecular properties of drugs
```
python3 fine_tuning/prediction_Training_r0_r1_seq100.py 
        [--dataset_name DATASET_NAME] 
        [--cudaId CUDA_ID]
        [--batch_size BATCH_SIZE]
        [--model_Id MODEL_ID]
        [--num_hidden_layers NUM_HIDDEN_LAYERS]
        [--num_attention_heads NUM_ATTENTION_HEADS]
        [--hidden_dropout_prob HIDDEN_DROPOUT_PROB]
        [--attention_probs_dropout_prob ATTENTION_PROBS_DROPOUT_PROB]
        [--dr DR]

one example:
python3 fine_tuning/prediction_Training_r0_r1_seq100.py --dataset_name bbbp --cudaId 0 --batch_size 8 --model_Id 1 
--num_hidden_layers 6 --num_attention_heads 6 --hidden_dropout_prob 0.5 --attention_probs_dropout_prob 0.5 --dr 0.9

```
### Datasets
- Toxw21
- SIDER
- ClinTox
- BBBP

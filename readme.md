# Pretraining
python3 BERT_training_seq100.py

# Fine tuning
python3 fine_tuning/prediction_Training_r0_r1_seq100.py --dataset_name bbbp --cudaId 0 --batch_size 4 --model_Id 1 --num_hidden_layers 6 --num_attention_heads 6 --hidden_dropout_prob 0.5 --attention_probs_dropout_prob 0.5
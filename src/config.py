
MAX_LEN 					= 512
TRAIN_BATCH_SIZE 			= 4
VALID_BATCH_SIZE 			= 4
EPOCHS 						= 1

GPT2_MODEL_PATH  			= "/home/ubuntu/TAKD/smart-compose/outputs_distilgpt2_banking/checkpoint-500"
TRAINING_FILE 				= "../data/banking_data/only_messages_test.csv"
TESTING_FILE  				= "../data/banking_data/only_messages_test.csv"
LOG_FILE_NAME 				= "../logs/app.log"
TRAINING_LOG_FILE			= "../logs/train.log"



weight_decay				= 0.0
learning_rate				= 5e-5
adam_epsilon				= 1e-8
warmup_steps				= 0
gradient_accumulation_steps = 1
n_gpu						= 1
max_grad_norm               = 1.0
save_steps 					= 500
output_dir					= "../outputs_distilgpt2_banking/"
max_steps					= -1
temperature					= 1.0
k 							= 0
p 							= 0.9
repetition_penalty			= 1.0
num_return_sequences        = 1
stop_token 					= "\n"
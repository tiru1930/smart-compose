
from dataset import data_loders,LineByLineTextDataset
from model import GPT2BaseModel
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch
import logging
import config
import os 
import transformers

logging.basicConfig(filename=config.TRAINING_LOG_FILE,
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filemode='w',
                    )

logger = logging.getLogger()


class TrainGPT2onCustomData(object):
    """docstring for TrainGPT2onCustomData"""
    def __init__(self):
        super(TrainGPT2onCustomData, self).__init__()

        try:
            self.model      = GPT2BaseModel().gpt2
            self.tokenizer  = GPT2BaseModel().gpt2_tokenizer
            train_dataset   = LineByLineTextDataset(self.tokenizer,config.TRAINING_FILE)
            test_dataset    = LineByLineTextDataset(self.tokenizer,config.TESTING_FILE)
            loadData        = data_loders(self.tokenizer,train_dataset,test_dataset)
            self.train_dataloader = loadData.trainloder()
            self.test_dataloader  = loadData.testloder()
            self.device           = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

        except Exception as e:
            logger.error("Error in TrainGPT2onCustomData init {}".format(e),exc_info=True)
            exit(-1)
     

    def train(self):
        try:
            t_total = len(self.train_dataloader) // config.gradient_accumulation_steps * config.EPOCHS
            model   = self.model.module if hasattr(self.model, "module") else self.model
            model.resize_token_embeddings(len(self.tokenizer))

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": config.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total
            ) 
            

            global_step = 0
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            tr_loss, logging_loss = 0.0, 0.0


            model.zero_grad()
            train_iterator = trange(epochs_trained, int(config.EPOCHS), desc="Epoch", disable=True)

            for _ in train_iterator:
                epoch_iterator = tqdm(self.train_dataloader, desc="Iteration", disable=False)

                for step, batch in enumerate(epoch_iterator):

                    inputs, labels = (batch, batch)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    model.train()
                    outputs = model(inputs, labels=labels)
                    loss = outputs[0] 

                    if config.n_gpu > 1:
                        loss = loss.mean()  
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps

                    loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % config.gradient_accumulation_steps == 0:
                     
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                        scheduler.step() 
                        model.zero_grad()
                        global_step += 1


                        if  config.save_steps > 0 and global_step % config.save_steps == 0:
                            checkpoint_prefix = "checkpoint"
                            output_dir = os.path.join(config.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                            os.makedirs(output_dir, exist_ok=True)
                            model_to_save = (model.module if hasattr(model, "module") else model)  
                            model_to_save.save_pretrained(output_dir)
                            self.tokenizer.save_pretrained(output_dir)

                            logger.info("Saving model checkpoint to {}".format(output_dir))
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

                        if config.max_steps > 0 and global_step > config.max_steps:
                            epoch_iterator.close()
                            break
                if config.max_steps > 0 and global_step > config.max_steps:
                    train_iterator.close()
                    break

            return model,self.tokenizer

        except Exception as e:
            logger.error("Error in TrainGPT2onCustomData train {}".format(e),exc_info=True)
            exit(-1)

    def evaluate(self,model, tokenizer):

        try:
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = {}".format(len(self.test_dataloader)))
            logger.info("  Batch size = {}".format(config.VALID_BATCH_SIZE))

            eval_loss = 0.0
            nb_eval_steps = 0
            model.to(self.device)
            model.eval()

            for batch in tqdm(self.test_dataloader, desc="Evaluating"):
                inputs, labels = (batch, batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    outputs = model(inputs, labels=labels)
                    lm_loss = outputs[0]
                    eval_loss += lm_loss.mean().item()
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            perplexity = torch.exp(torch.tensor(eval_loss))
            result = {"perplexity": perplexity}
            logger.info("perplexity : {}".format(perplexity))

        except Exception as e:
            logger.error("Error in TrainGPT2onCustomData evaluate {}".format(e),exc_info=True)
            exit(-1)




def main():
    TGC = TrainGPT2onCustomData()
    model, tokenzier = TGC.train()

    # tokenzier      = transformers.AutoTokenizer.from_pretrained(config.FINE_TUNED_MODEL_PATH)
    # gpt2_config    = transformers.AutoConfig.from_pretrained(config.FINE_TUNED_MODEL_PATH)
    # model          = transformers.AutoModelWithLMHead.from_pretrained(config.FINE_TUNED_MODEL_PATH,
    #                                                                         config=gpt2_config)
    TGC.evaluate(model,tokenzier)
    
if __name__ == '__main__':
    main()
                                
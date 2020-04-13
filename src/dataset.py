from torch.utils.data import DataLoader, Dataset, RandomSampler,SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import config
import logging
import os 
import torch 

logger = logging.getLogger()

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer,file_path, block_size=512):

        try:
            assert os.path.isfile(file_path)
        except Exception as e:
            logger.error("Error in LineByLineTextDataset {}".format(e))

       
        logger.info("Creating features from dataset file at {}".format(file_path))

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

class data_loders(object):
    """docstring for data_loders"""
    def __init__(self, tokenizer,train_data,test_data):
        super(data_loders, self).__init__()
        self.tokenizer      = tokenizer
        self.train_data     = train_data
        self.test_data      = test_data 

    def trainloder(self):

        def collate(train_data):
            try:
                if self.tokenizer._pad_token is None:
                    return pad_sequence(train_data, batch_first=True)
                return pad_sequence(train_data, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            except Exception as e:
                logger.error("Error in data_loders {}".format(e))

        try:
            train_sampler = RandomSampler(self.train_data)
            train_dataloader = DataLoader(
                                        self.train_data, sampler=train_sampler, batch_size=config.TRAIN_BATCH_SIZE, \
                                        collate_fn=collate)
            return train_dataloader
        except Exception as e:
            logger.error("Error in data_loders train_dataloader {}".format(e),exc_info=True)

    def testloder(self):

        def collate(test_data):
            try:
                if self.tokenizer._pad_token is None:
                    return pad_sequence(test_data, batch_first=True)
                return pad_sequence(test_data, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            except Exception as e:
                logger.error("Error in data_loders {}".format(e))

        try:
            test_sampler    = SequentialSampler(self.test_data)
            test_dataloader = DataLoader(
                                        self.test_data, sampler=test_sampler, batch_size=config.VALID_BATCH_SIZE, \
                                        collate_fn=collate)
            return test_dataloader
        except Exception as e:
            logger.error("Error in data_loders testloder {}".format(e),exc_info=True)
            
        
        
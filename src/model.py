import config 
import transformers
import torch.nn as nn


class GPT2BaseModel(object):
	"""docstring for GPT2BaseModel"""
	def __init__(self):
		super(GPT2BaseModel, self).__init__()
		self.gpt2_tokenizer = transformers.AutoTokenizer.from_pretrained(config.GPT2_MODEL_PATH)
		self.gpt2_config	= transformers.AutoConfig.from_pretrained(config.GPT2_MODEL_PATH)
		self.gpt2 			= transformers.AutoModelWithLMHead.from_pretrained(config.GPT2_MODEL_PATH,
																				config=self.gpt2_config)

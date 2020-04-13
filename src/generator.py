import config
from model import GPT2BaseModel
import torch
import logging 

logger = logging.getLogger()

class GeneratText(object):
    """docstring for GeneratText"""
    def __init__(self):
        super(GeneratText, self).__init__()

        try:

            logger.info("Lazy loading of model and tokenizer")
            self.model          = GPT2BaseModel()
            self.tokenizer      = self.model.gpt2_tokenizer
            self.length         = config.MAX_LEN
            self.temperature    = config.temperature
            self.k              = config.k
            self.p              = config.p 
            self.repetition_penalty   = config.repetition_penalty 
            self.num_return_sequences = config.num_return_sequences
            self.stop_token           = config.stop_token
            self.device         = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
            self.model.gpt2.to(self.device)

        except Exception as e:
            logger.error("Exception in GeneratText init class {}".format(e),exc_info=True)
            exit(-1)
 

    def generateSequences(self,text):

        try:

            logger.info("encoding the input Text")
            encoded_text        =  self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            encoded_text        = encoded_text.to(self.device)
            logger.info("convert encoded text to {}".format(self.device))        
            output_sequences    =  self.model.gpt2.generate(
                                                        input_ids   =   encoded_text,
                                                        max_length  =   self.length + len(encoded_text[0]),
                                                        temperature =   self.temperature,
                                                        top_k       =   self.k,
                                                        top_p       =   self.p,
                                                        repetition_penalty  =   self.repetition_penalty,
                                                        do_sample           =   True,
                                                        num_return_sequences=   self.num_return_sequences,
                                                    )
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []
            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                logger.info("Generated sequenc {}".format(generated_sequence_idx+1))
                generated_sequence  = generated_sequence.tolist()
                decoded_text        = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                decoded_text        = decoded_text[: decoded_text.find(self.stop_token) if self.stop_token else None]
                total_sequence      = (
                                        decoded_text[len(self.tokenizer.decode(encoded_text[0],\
                                        clean_up_tokenization_spaces=True)) :]
                                      )
                generated_sequences.append(total_sequence)
            logger.info("sequence generation is completed and appended to original text")

            return generated_sequences,True

        except Exception as e:

            logger.error("Error in while generated_sequences {}".format(e),exc_info=True)
            return [],False

 
        

def main():
    
    GT          = GeneratText()
    prompt_text = input("Enter Text")
    print(GT.generateSequences(prompt_text))
        
if __name__ == '__main__':
    main()
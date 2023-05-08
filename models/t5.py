import torch
from transformers import T5Tokenizer, T5Model, T5EncoderModel

class T5:

    def __init__(self, model_name, device):
        self.model = T5EncoderModel.from_pretrained('google/flan-t5-xl')
        self.processor = T5Tokenizer.from_pretrained('google/flan-t5-xl')

        self.model.to(device)
        self.device = device
    
    def extract_features(self, sentences):
        sentences = sentences.replace("[MASK]", "<extra_id_0>")
        with torch.no_grad():
            inputs = self.processor(sentences, return_tensors="pt")
            inputs = inputs.to(self.device)
            inputs['output_hidden_states'] = True
            outputs = self.model(**inputs)            
        return outputs.last_hidden_state.squeeze(0)[-1].data.cpu().numpy()
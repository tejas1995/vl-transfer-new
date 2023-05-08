import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, BertModel, BertTokenizer, BertForMaskedLM

class Clip:

    def __init__(self, model_name, device):
        self.model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        self.model.to(device)
        self.device = device
    
    def extract_features(self, sentences):
        with torch.no_grad():
            inputs = self.processor(sentences, return_tensors="pt")
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
        return outputs.pooler_output.squeeze(0).data.cpu().numpy()
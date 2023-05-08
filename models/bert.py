import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, BertModel, BertTokenizer, BertForMaskedLM

class Bert:

    def __init__(self, model_name, device):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.processor = BertTokenizer.from_pretrained('bert-base-uncased')

        self.model.to(device)
        self.device = device

        if model_name == 'bert-lxmert':
            sd = torch.load('/home/woojeong2/vl-transfer/models/bert-lxmert-trained/mp_rank_00_model_states.pt', map_location=torch.device('cpu'))['module']
            for k in self.model.state_dict().keys():
                if 'bert.'+k in sd.keys():
                    self.model.state_dict()[k].copy_(sd['bert.'+k])
                elif 'pooler' in k:
                    self.model.state_dict()[k].copy_(sd[k.replace('pooler', 'cls.predictions.transform')])
                else:
                    print("Not initializing {}".format(k))

    def extract_features(self, sentences):
        with torch.no_grad():
            inputs = self.processor(sentences, return_tensors="pt")
            inputs = inputs.to(self.device)
            outputs = self.model(**inputs)
        return outputs.pooler_output.squeeze(0).data.cpu().numpy()
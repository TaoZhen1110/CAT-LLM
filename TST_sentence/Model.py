import torch.nn as nn
from transformers import BertModel

class ScatteredClassification(nn.Module):
    def __init__(self, checkpoint, freeze):
        super(ScatteredClassification, self).__init__()
        self.encoder = BertModel.from_pretrained(checkpoint)
        if freeze == "1":
            self.encoder.embeddings.word_embeddings.requires_grad_(False)
        if freeze == "2":
            self.encoder.embeddings.requires_grad_(False)
        if freeze == "3":
            self.encoder.embeddings.requires_grad_(False)
            self.encoder.encoder.requires_grad_(False)
        self.classifier = nn.Linear(self.encoder.config.hidden_size,2)

    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        outs = outs.pooler_output
        outs = self.classifier(outs)
        return outs


class RhetoricClassification(nn.Module):
    def __init__(self, checkpoint, freeze):
        super(RhetoricClassification, self).__init__()
        self.encoder = BertModel.from_pretrained(checkpoint)
        if freeze == "1":
            self.encoder.embeddings.word_embeddings.requires_grad_(False)
        if freeze == "2":
            self.encoder.embeddings.requires_grad_(False)
        if freeze == "3":
            self.encoder.embeddings.requires_grad_(False)
            self.encoder.encoder.requires_grad_(False)
        self.classifier = nn.Linear(self.encoder.config.hidden_size,10)

    def forward(self, enc_inputs, attention_mask, token_type_ids):
        outs = self.encoder(input_ids=enc_inputs, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        outs = outs.pooler_output
        outs = self.classifier(outs)
        return outs

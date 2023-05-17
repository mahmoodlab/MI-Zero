import numpy as np
import torch
import torch.nn.functional as F

def tokenize(tokenizer, texts):
    tokens = tokenizer.batch_encode_plus(texts, 
                                         max_length = 64,
                                         add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                         return_token_type_ids=False,
                                         truncation = True,
                                         padding = 'max_length',
                                         return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

def zero_shot_classifier(model, tokenizer, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.replace('CLASSNAME', classname) for template in templates]

            texts, attention_mask = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
            texts = torch.from_numpy(np.array(texts)).to(device)
            attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
            class_embeddings = model.encode_text(texts, attention_mask=attention_mask)
            
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def accuracy(logits, target, topk=(1,)):
    pred = logits.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
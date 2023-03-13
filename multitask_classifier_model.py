
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
                
        # self.bert2 = BertModel.from_pretrained('bert-base-uncased')
        # for param in self.bert2.parameters():
        #     if config.option == 'pretrain':
        #         param.requires_grad = False
        #     elif config.option == 'finetune':
        #         param.requires_grad = True
                
        ### TODO
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
    
        # sentiment
        self.sentiment_drop_out = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_output_proj = torch.nn.Linear(config.hidden_size, 5)

        # paraphrase
        self.paraphrase_drop_out = torch.nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_output_proj1 = torch.nn.Linear(2*config.hidden_size, config.hidden_size)
        self.paraphrase_nl = F.gelu
        self.paraphrase_output_proj2 = torch.nn.Linear(config.hidden_size, 1)
                
        # similarity, sts
        self.similarity_drop_out = torch.nn.Dropout(config.hidden_dropout_prob)
        self.similarity_output_proj1 = torch.nn.Linear(2*config.hidden_size, config.hidden_size)
        self.similarity_nl = F.gelu
        if config.sts_train_method == "regression":
            self.similarity_output_proj2 = torch.nn.Linear(config.hidden_size, 1)
        else:
            self.similarity_output_proj2 = torch.nn.Linear(config.hidden_size, 6)
        
    def call_backbone(self, input_ids, attention_mask):
        res = self.bert(input_ids, attention_mask)
        
        #sequence_output = self.bert2.encode(res['last_hidden_state'], attention_mask=attention_mask)
        sequence_output = res['last_hidden_state']        
        
        first_tk = sequence_output[:, 0]
        first_tk = self.bert.pooler_dense(first_tk)
    
        #first_tk = res['pooler_output']
        
        #first_tk = sequence_output[:, 0]
        #first_tk = self.pooler_dense(first_tk)
        #first_tk = self.pooler_af(first_tk)
    
        return first_tk, sequence_output
        
    def forward(self, input_ids, attention_mask, task_str):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        if task_str == 'sst':
            return self.predict_sentiment(input_ids, attention_mask)
        
        if task_str == 'sts':
            return self.predict_similarity(input_ids[0], attention_mask[0], input_ids[1], attention_mask[1])
        
        if task_str == 'para':
            return self.predict_paraphrase(input_ids[0], attention_mask[0], input_ids[1], attention_mask[1])

        raise "incorrect task str ..."

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooler_output, last_hidden_state = self.call_backbone(input_ids, attention_mask)
        logits = self.sentiment_output_proj(self.sentiment_drop_out(pooler_output))
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        pooler_output_1, _ = self.call_backbone(input_ids_1, attention_mask_1)       
        pooler_output_2, _ = self.call_backbone(input_ids_2, attention_mask_2)
        
        x = torch.concat((pooler_output_1, pooler_output_2), dim=1)
        x = self.paraphrase_output_proj1(self.paraphrase_drop_out(x))
        logits = self.paraphrase_output_proj2(self.paraphrase_nl(x))
        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        pooler_output_1, _ = self.call_backbone(input_ids_1, attention_mask_1)       
        pooler_output_2, _ = self.call_backbone(input_ids_2, attention_mask_2)
        
        x = torch.concat((pooler_output_1, pooler_output_2), dim=1)
        x = self.similarity_output_proj1(self.similarity_drop_out(x))
        logits = self.similarity_output_proj2(self.similarity_nl(x))
        return logits


if __name__ == "__main__":    
    pass
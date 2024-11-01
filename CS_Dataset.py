from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import h5py


class CSDataset(Dataset):
    def __init__(self, type, language_pair, mask_out_prob=0.15):
        self.texts = h5py.File(f"{language_pair}/{type}", 'r')
        self.inputs = self.texts['inputs']
        self.labels = self.texts['labels']
        self.attention_masks = self.texts['attention_masks']

        self.mask_out_prob = mask_out_prob

        mask_embeddings = h5py.File(f"{language_pair}/mask_vec", 'r')
        self.MASK_TOKEN_VEC_1 = mask_embeddings['mask_embed_1'][:]
        self.MASK_TOKEN_VEC_2 = mask_embeddings['mask_embed_2'][:]

        self.linear_layer = nn.Linear(300, 768)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sentence = self.inputs[idx]
        label = self.labels[idx]
        attention_mask = self.attention_masks[idx]

        # Randomly mask out words
        for i in range(len(sentence)):
            if np.random.rand() < self.mask_out_prob:
                if label[i] == '0':
                    sentence[i] = self.MASK_TOKEN_VEC_1
                else:
                    sentence[i] = self.MASK_TOKEN_VEC_2

        # Convert to tensor
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        input_tensor = torch.tensor(np.array(sentence), dtype=torch.float32)
        input_tensor = self.linear_layer(input_tensor).to(device)
        label_tensor = torch.tensor(label, dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(device)

        return {'inputs_embeds': input_tensor, 'labels': label_tensor, 'attention_mask': attention_mask_tensor}

    def __del__(self):
        self.texts.close()

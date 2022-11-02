from torch.utils import data
import  torch


class CSIC2010(data.Dataset):
    def __init__(self, token_text ,device):
        self.token_text = torch.Tensor(token_text).long()
        self.device = device
        if (self.device == "cuda"):
            self.token_text = self.token_text.cuda()
    def __getitem__(self, index):
        return self.token_text[index]
    def __len__(self):
        return len(self.token_text)


from torch.utils import data
from PIL import Image
import  torch
import  numpy as np
import  re
class MyDataset(data.Dataset):
    def __init__(self, token_text ,device):
        self.token_text = token_text
        self.device = device
    def __getitem__(self, index):
        tmp = torch.Tensor(self.token_text[index]).long()
        if(self.device=="cuda"):
            tmp = tmp.cuda()

        return [tmp, tmp]

    def __len__(self):
        return len(self.token_text)



def loaddata(path):
    data = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # get http://localhost:8080/tienda1/index.jsp
            line = line.strip();
            line = line.replace('/', ' ');
            line = line.replace(':', ' ');
            line = line.replace('.', ' ');
            line = line.replace('&', ' ');
            line = line.replace('+', ' ');
            line = line.replace('-', ' ');
            line = line.replace('?', ' ');
            line = line.replace('=', ' ');
            line = re.sub(' +', ' ', line)
            #print(line);
            line += ' [EOS]'
            line = '[BOS] ' + line
            line = line.split(' ')
            data.append(line)

    data = np.array(data, dtype=object)
    return data
def loaddataX(path):
    data = []
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        maxlen=0
        for line in lines:
            # get http://localhost:8080/tienda1/index.jsp
            line = line.strip();
            data.append(line)
            maxlen=max(maxlen,len(line))
        print(maxlen)  # 53  padding 到60
    data = np.array(data, dtype=object)
    return data
def getToken(text_data):
    text2tokensdic = {}
    text2tokensdic['PAD'] = 0

    tokens2textdic = {}
    tokens2textdic[0] = 'PAD'

    print(type(text2tokensdic))
    print(text2tokensdic.keys())
    i = 1
    for sentence in text_data:
        for voc in sentence:
            if (voc not in text2tokensdic.keys()):
                text2tokensdic[voc] = i
                tokens2textdic[i] = voc

                i += 1
    text2tokensdic['<UNK>'] = i
    tokens2textdic[i] = '<UNK>'

    print(text2tokensdic)
    print(tokens2textdic)

    return text2tokensdic,tokens2textdic
def getCharToken():
    CharType="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~!*'();:@&=+$,/?#[]"
    text2tokensdic = {}
    tokens2textdic = {}

    text2tokensdic['PAD'] = 0
    tokens2textdic[0] = 'PAD'

    text2tokensdic['[BOS]'] = 1
    tokens2textdic[1] = '[BOS]'

    text2tokensdic['[EOS]'] = 2
    tokens2textdic[2] = '[EOS]'

    print(type(text2tokensdic))
    print(text2tokensdic.keys())
    i = 3
    for chr in CharType:
        text2tokensdic[chr] = i
        tokens2textdic[i] = chr
        i = i + 1

    text2tokensdic['<UNK>'] = i
    tokens2textdic[i] = '<UNK>'

    print(text2tokensdic)
    print(tokens2textdic)

    return text2tokensdic,tokens2textdic
def tokenText(training_data,text2tokensdic) :
    Tokened_text = []
    maxlen = 0
    for sentence in training_data:
        tmpsentence = []
        for voc in sentence:
            if (voc not in text2tokensdic.keys()):
                tmpsentence.append(len(text2tokensdic))
            else:
                tmpsentence.append(text2tokensdic[voc])
        maxlen = max(maxlen, len(tmpsentence))
        while (len(tmpsentence) < 65):
            tmpsentence.append(0)
        Tokened_text.append(tmpsentence)

    print(maxlen)  # 53  padding 到60
    Tokened_text = np.array(Tokened_text)

    #print(Tokened_text)
    return  Tokened_text
def tokenTextX(training_data,text2tokensdic,padding_size) :
    Tokened_text = []
    maxlen = 0
    for sentence in training_data:
        tmpsentence = []
        tmpsentence.append(1)
        for chr in sentence:
            if (chr not in text2tokensdic.keys()):
                tmpsentence.append(len(text2tokensdic)-1)
            else:
                tmpsentence.append(text2tokensdic[chr])
        tmpsentence.append(2)
        maxlen = max(maxlen, len(tmpsentence))
        while (len(tmpsentence) < padding_size):
            tmpsentence.append(0)
        Tokened_text.append(tmpsentence)
    Tokened_text = np.array(Tokened_text)
    #print(Tokened_text)
    return  Tokened_text


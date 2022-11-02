
import  numpy as np
import  re

from datasets.CSIC2010 import CSIC2010


def read_data(path):
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
            # line += ' [EOS]'        #做多头注意力机制就不需要开头和结束标志了
            # line = '[BOS] ' + line
            line = line.split(' ')
            data.append(line)

    data = np.array(data, dtype=object)
    return data


#输入所有的训练集 得到 Token字典
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

    #print(text2tokensdic)
    #print(tokens2textdic)

    return text2tokensdic,tokens2textdic


#根据token字典把字符串词汇转成一个数字，并padding到指定长度
def tokenText(training_data,text2tokensdic,padding_len) :
    Tokened_text = []
    maxlen = 0
    for sentence in training_data:
        tmpsentence = []
        for voc in sentence:
            if (voc not in text2tokensdic.keys()):
                tmpsentence.append(len(text2tokensdic)-1)
            else:
                tmpsentence.append(text2tokensdic[voc])
        maxlen = max(maxlen, len(tmpsentence))
        while (len(tmpsentence) < padding_len):
            tmpsentence.append(0)
        Tokened_text.append(tmpsentence)

    print(maxlen)  # 53  padding 到60
    Tokened_text = np.array(Tokened_text)

    #print(Tokened_text)
    return  Tokened_text




def GetDataSet(data_path,padding_len,device,text2tokensdic=None):

    # 按照词汇分词
    raw_data = read_data(data_path)

    if(text2tokensdic is None):
        text2tokensdic, tokens2textdic = getToken(raw_data)

    Tokened_data= tokenText(raw_data, text2tokensdic, padding_len)
    DataSet = CSIC2010(Tokened_data, device)

    return DataSet,text2tokensdic
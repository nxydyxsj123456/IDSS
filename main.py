# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import *
from model import *
import torch.optim as optim
import binascii

normal_path = './data/normal.txt'

anomalous_path = './data/anomalous.txt'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Token_type="Char" #["Char","vocabulary"]

    text2tokensdic={}
    tokens2textdic={}
    if Token_type== "vocabulary":
        #按照词汇分词
        normal_data = loaddata(normal_path)

        anomalous_data = loaddata(anomalous_path)

        text2tokensdic,tokens2textdic=getToken(normal_data)

    elif Token_type == "Char":
        #按照符号分词
        text2tokensdic,tokens2textdic=getCharToken()

        normal_data = loaddataX(normal_path)

        anomalous_data = loaddataX(anomalous_path)

    Tokened_normal = tokenTextX(normal_data, text2tokensdic,700)

    Tokened_anomalous = tokenTextX(anomalous_data, text2tokensdic,700)

    INPUT_DIM = OUTPUT_DIM = len(text2tokensdic)

    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(device)
    print(device)
    #print(device=="cpu")
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    TRG_PAD_IDX = 0  # TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_valid_loss = float('inf')

    train_X, test_X, train_y, test_y = train_test_split(Tokened_normal, Tokened_normal, test_size=0.3, random_state=5)
    training_dataset = MyDataset(train_X,device)
    test_dataset = MyDataset(test_X,device)
    anomalous_dataset = MyDataset(anomalous_data,device)

    train_iterator = DataLoader(training_dataset, 1)
    valid_iterator = DataLoader(test_dataset, 1)
    test_iterator = DataLoader(anomalous_dataset, 1)


    Train_Loss=[]
    Valid_Loss=[]
    Test_Loss=[]
    for epoch in range(50):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_iterator, criterion)
        test_loss = evaluate(model, test_iterator, criterion)


        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            torch.save(model.state_dict(), str(epoch) + "model.pt")

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValid.Loss: {valid_loss:.3f} |  Valid. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tTest. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')
        Train_Loss.append(train_loss)
        Valid_Loss.append(valid_loss)
        Test_Loss.append(test_loss)


    print(Train_Loss)
    print(Valid_Loss)
    print(Test_Loss)

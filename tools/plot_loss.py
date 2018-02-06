from __future__ import print_function
import matplotlib.pyplot as plt
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file1',default='/home/lixiaojie/torch-srgan/checkpoints/densenet40/densenet40.json')
parser.add_argument('--file2',default='/home/lixiaojie/torch-srgan/checkpoints/resnet19/resnet18.json')
args = parser.parse_args()

if __name__ == '__main__':
    train_loss1 = json.load(open(args.file1))
    train_loss2 = json.load(open(args.file2))
    print(train_loss1.keys())
    print(train_loss2.keys())
    iter_num1 = len(train_loss1['train_loss_epoch_history'])
    iter_num2 = len(train_loss2['train_loss_history'])
    print(iter_num1)
    print(iter_num2)
    #iter_num2 = len(train_loss1['train_loss_epoch_history'])
    gap_length = 1
    iter_index1 = range(0,iter_num1)[::gap_length]
    iter_index2 = range(0,iter_num2)[::gap_length]
    #plt.plot(iter_index,train_loss1['train_loss_history'],'r')
    #plt.plot(iter_index,train_loss1['style_loss_history']['content-16'][::gap_length],'r',label = 'content loss')
    plt.plot(iter_index1, train_loss1['val_loss_history'][::gap_length], 'r', label='densenet40 loss')
    plt.plot(iter_index2, train_loss2['val_loss_history'][::gap_length],'b',label = 'resnet18 loss')
    ax = plt.gca()
    ax.set_xlabel('Parameters updates')
    ax.set_ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()

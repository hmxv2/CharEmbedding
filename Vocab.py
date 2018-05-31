import json
import sys
import random
import numpy as np
import pickle

class Vocab:
    def __init__(self, in_file):
        self.fi=in_file
        self.all_words_count=0
        self.all_lines_count=0
    def build_vocab(self):
        word2count={'unk':100,'sos':100,'eos':100,'padding':100}
        word2idx={'unk':0,'sos':1,'eos':2,'padding':3}
        idx2word=['unk','sos','eos','padding']
        lines_length=[]
        label_set=set()
        #
        fi=open(self.fi,'r')
        for line in fi:
            self.all_lines_count+=1
            #
            line=line.split(' | ')
            explain=line[0]
            label=line[1]
            label=label[:-1]
            #print(len(label))
            #
            lines_length.append(len(explain))
            #
            for x in explain:
                if x in word2idx:
                    word2count[x]+=1
                else:
                    word2count[x]=1
                    self.all_words_count+=1
                    #
                    word2idx[x]=len(idx2word)
                    idx2word.append(x)
            
            if label in word2idx:
                word2count[label]+=1
            else:
                word2count[label]=1
                self.all_words_count+=1
                #
                word2idx[label]=len(idx2word)
                idx2word.append(label)
            label_set.add(label)
            
        #
        self.word2count=word2count
        self.word2idx=word2idx
        self.idx2word=idx2word
        self.lines_length=lines_length
        self.label_set=label_set
        
        return self.word2idx, self.idx2word
    def print_vocab(self):
        print('all words count: %s' %self.all_words_count)
        print('all lines count: %s' %self.all_lines_count)
        print('labels count: %s' %len(self.label_set))
    #
    def save_vocab(self, fo):
        with open(fo,'wb') as f:
            pickle.dump(self,f)
        f.close()
        '''
        fo1=open(fo1,'w')
        json.dump(self.word2idx, fo1)
        fo1.close()
        fo2=open(fo2,'w')
        json.dump(self.idx2word, fo2)
        fo2.close()
        '''
    #
    def line_length_statistics(self,filter_rate=0.9, length_cut=25):
        cnt=0
        length_cut2=max(self.lines_length)*filter_rate
        for x in self.lines_length:
            if x<length_cut2 and x<=length_cut:
                cnt+=1
        print('max length of explain: %d' % max(self.lines_length))
        print('length cut result: %d' % cnt)
        
        
            
                
            
        
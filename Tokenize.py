import json
import sys
import random
import numpy as np
import pickle

class Tokenize:
    def __init__(self, fi, vocab, length_cut):
        self.fi=fi
        self.vocab=vocab
        self.max_length_cut=length_cut#self.vocab.lines_length
        self.sentence_lengths=[]
    #
    def word_tokenize(self):
        max_length_cut=self.max_length_cut
        #
        f=open(self.fi, 'r')
        explains_token=[]
        labels_token=[]
        for line in f:
            line=line.split(' | ')
            explain=line[0]
            label=line[1]
            label=label[:-1]
            #print(label)
            #
            explain_token=[]
            #explain_token=[self.vocab.word2idx['sos']]
            cnt=0
            for x in explain:
                cnt+=1
                if cnt<=max_length_cut:
                    if x in self.vocab.word2idx:
                        explain_token.append(self.vocab.word2idx[x])
                    else:
                        explain_token.append(self.vocab.word2idx['unk'])
                else:
                    break
            #explain_token.append(self.vocab.word2idx['eos'])
            while(len(explain_token)<max_length_cut):
                explain_token.append(self.vocab.word2idx['padding'])
            #
            explains_token.append(explain_token)
            if label in self.vocab.word2idx:
                labels_token.append(self.vocab.word2idx[label])
            else:
                labels_token.append(self.vocab.word2idx['unk'])
                
            self.sentence_lengths.append(min(len(explain), max_length_cut))#without 'sos' and 'eos'
        #
        self.explains_token=explains_token
        self.labels_token=labels_token
        #
        return explains_token, labels_token
    
    def save_tokenize(self, fo):
        with open(fo,'wb') as f:
            pickle.dump(self,f)
        f.close()

        
    
__author__ = 'zhangyin'

import re, string, timeit
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import FreqDist
import json
import os
import nltk
import numpy as np
stemmer = PorterStemmer()

###### data loading and parsing functions#########
def parse_to_sentence(content,stopwords):
    sent_word=[]
    sentences = nltk.sent_tokenize(content)
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        temp = [stemmer.stem(w.lower()) for w in words if w not in string.punctuation]
        temp2 = [v for v in temp if v not in stopwords]
        if len(temp2)>0:
            sent_word.append(temp2)
    return sent_word


def load_a_json_file(filename):
    with open(filename,encoding = "ISO-8859-1") as data_file:
        data = json.load(data_file)
    return data

def load_all_json_files(jsonfolder,suffix):
    data=[]
    def load_a_json_folder(folder,suffix):
        if not folder[-1]=="/":
            folder=folder+"/"
        fs = os.listdir(folder)    # list all the files and sub folders under the Path
        for f in fs:
            if not f.startswith("."):  # ignore files or folders start with period
                fpath=folder+f
                if not os.path.isdir(fpath):  # if this is not a folder, that is, this is a file
                    # add data loading code
                    if fpath.split(".")[-1]==suffix:
                        with open(fpath,encoding = "ISO-8859-1") as data_file:
                            data.append(json.load(data_file))
                else:
                    subfolder=fpath+"/"
                    load_a_json_folder(subfolder,suffix)  # else this is a folder
    load_a_json_folder(jsonfolder,suffix)
    return data


class CreateVocab:
    def create_stopwords(self):
        init_stopwords = [stemmer.stem(v) for v in stopwords.words('english')]
        additional_stopwords = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
        self.stopwords = additional_stopwords + init_stopwords

    def read_data(self,folder,suffix):
        # suffix="json"
        # folder="/Users/zhangyin/python projects/IR project/data/yelp mp1 data/"
        self.corpus=load_all_json_files(folder,suffix)

    def create_vocab(self):
        All_Contents = []
        i=0
        for hotel in self.corpus:
            print("loading file No."+str(i+1))
            for review in hotel.get("Reviews"):
                s= []
                for v in parse_to_sentence(review.get('Content'),self.stopwords):
                    s = v + s
                All_Contents = All_Contents + s
            i=i+1
        term_freq = FreqDist(All_Contents)
        Vocab = []
        Count = []
        VocabDict={}
        for k,v in term_freq.items():
            if v>5:
                Vocab.append(k)
                Count.append(v)
        self.Vocab = np.array(Vocab)[np.argsort(Vocab)].tolist()
        self.Count = np.array(Count)[np.argsort(Vocab)].tolist()
        self.VocabDict = dict(zip(self.Vocab,range(len(self.Vocab))))

    def save_to_file(self,savefilepath):
        np.save(savefilepath, (self.corpus,self.Vocab,self.Count,self.VocabDict))
        print("succeed saving to file "+savefilepath)

    def load_Vocab(self,loadfilepath):
        print("loading data from" + loadfilepath)
        return np.load(loadfilepath)

# savefilepath = "./output/yelp_mp1_corpus"
# loadfilepath = "./output/yelp_mp1_corpus.npy"

def get_top_p_tf(dict,p):
    temp = dict.copy()
    res=[]
    for i in range(p):
        key = temp.max()
        v = temp.get(key)
        temp.pop(key)
        # res.append((key,v))
        res.append(key)
    return res


# from  pprint import  pprint
# pprint(get_top_p_tf(term_freq,300))


# ########## Create the Vocab ##########
# All_Contents = []
# len(corpus)
# i=0
# for hotel in corpus:
#     for review in hotel.get("Reviews"):
#         s= []
#         for v in parse_to_sentence(review.get('Content')):
#             s = v + s
#         All_Contents = All_Contents + s
#     print(i)
#     i=i+1
#
# term_freq = FreqDist(All_Contents)



#
# Vocab = []
# Count = []
# VocabDict={}
# for k,v in term_freq.items():
#     if v>5 and k not in additional_stopwords:
#         Vocab.append(k)
#         Count.append(v)
#
# np.argsort(Vocab)
# Vocab = np.array(Vocab)[np.argsort(Vocab)].tolist()
# Count = np.array(Count)[np.argsort(Vocab)].tolist()
# ########################################
# VocabDict = dict(zip(Vocab,range(len(Vocab))))
#
# print(np.array(Vocab)[np.argsort(Vocab)][:1000])
# VocabDict.get("perfect")
#
# term_freq




# np.save("./output/yelp_mp1_corpus", (corpus,Vocab,Count,VocabDict))
# (corpus,Vocab,Count,VocabDict)=np.load("./output/yelp_mp1_corpus.npy")




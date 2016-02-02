__author__ = 'zhangyin'

import numpy as np
import os
import nltk
from nltk.stem.porter import *
from nltk import FreqDist
import math
from src.Structure import *

stemmer = PorterStemmer()



##### run code #######

# import numpy as np
# (corpus,Vocab,Count,VocabDict)=np.load("./output/yelp_mp1_corpus.npy")

# def run_bootstrapping():
analyzer = Bootstrapping()
loadfilepath = "/Users/zhangyin/python projects/IR project/init_aspect_word.txt"
load_Aspect_Terms(analyzer,loadfilepath,VocabDict)

(corpus,Vocab,Count,VocabDict)=np.load("./output/yelp_mp1_corpus.npy")
data = Corpus(corpus, Vocab, Count,VocabDict)
# 15 mins



Add_Aspect_Keywords(analyzer, 5, 5, data)

savefilepath = 'temp_aspect_words_yelp_mp1.txt'
save_Aspect_Keywords_to_file(analyzer,savefilepath,Vocab)

create_all_W(analyzer,data)
outputfolderpath = "/Users/zhangyin/python projects/IR project/output_data_for_rating/"
produce_data_for_rating(analyzer,data,outputfolderpath)




# # loadfilepath = "/Users/zhangyin/python projects/IR project/aspect words clean.txt"
# analyzer.Aspect_Terms
#
# # analyzer.sentence_label(data)
# # analyzer.calc_chi_sq(data)
#
# analyzer.Aspect_Terms[0]
#
# savefilepath = 'temp_aspect_words_yelp_mp1.txt'
# save_Aspect_Keywords_to_file(analyzer,savefilepath,Vocab)
#
#
# create_all_W(analyzer,data)
#
# outputfolderpath = "/Users/zhangyin/python projects/IR project/output_data_for_rating/"
# produce_data_for_rating(analyzer,data,outputfolderpath)
#
# ####### for
# # import numpy as np
# # (corpus,Vocab,Count,VocabDict)=np.load("./output/yelp_mp1_corpus.npy")
# data = Corpus(corpus, Vocab, Count,VocabDict)
# analyzer = Bootstrapping()
# loadfilepath = "/Users/zhangyin/python projects/IR project/init_aspect_word.txt"
# load_Aspect_Terms(analyzer,loadfilepath,VocabDict)
# # analyzer.calc_chi_sq(data)
# Add_Aspect_Keywords(analyzer, 5, 5, data)
# create_all_W(analyzer,data)
# outputfolderpath = "/Users/zhangyin/Google Drive/CS6501/data_for_rating/"
# produce_data_for_rating(analyzer,data,outputfolderpath)
# #######



# For tripadvisor data
# A1=[VocabDict.get(stemmer.stem(w.lower())) for w in ["value","price","quality","worth"]]
# A2=[VocabDict.get(stemmer.stem(w.lower())) for w in ["room","suite","view","bed"]]
# A3=[VocabDict.get(stemmer.stem(w.lower())) for w in ["location","traffic","minute","restaurant"]]
# A4=[VocabDict.get(stemmer.stem(w.lower())) for w in ["clean","dirty","maintain","smell"]]
# A5=[VocabDict.get(stemmer.stem(w.lower())) for w in ["stuff","check","help","reservation"]]
# A6=[VocabDict.get(stemmer.stem(w.lower())) for w in ["service","food","breakfast","buffet"]]
# A7=[VocabDict.get(stemmer.stem(w.lower())) for w in ["business","center","computer","internet"]]
# data.Aspect_Terms=[A1,A2,A3,A4,A5,A6,A7]
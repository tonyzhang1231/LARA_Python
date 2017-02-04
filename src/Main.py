__author__ = 'zhangyin'

from src.CreateVocab import *
from src.Structure import *

def run_CreateVocab():
    ##### step 1: creating a vocab from data
    cv_obj = CreateVocab()  ### create an instance
    cv_obj.create_stopwords()  ### create a list of stopwords
    suffix="json"
    folder="./data/yelp mp1 data/"
    cv_obj.read_data(folder,suffix)
    cv_obj.create_vocab()

    #### save it to a file
    savefilepath = "./data/yelp_mp1_corpus"
    cv_obj.save_to_file(savefilepath)


def run_bootstrap():
    ##### step 2. using bootstrapping method
    # if loading data from saved files
    cv_obj = CreateVocab()
    loadfilepath = "./data/yelp_mp1_corpus.npy"
    (cv_obj.corpus,cv_obj.Vocab,cv_obj.Count,cv_obj.VocabDict)=np.load(loadfilepath)
    # data = Corpus(cv_obj.corpus, cv_obj.Vocab, cv_obj.Count,cv_obj.VocabDict)
    # load_Aspect_Terms(BSanalyzer,loadfilepath,VocabDict)

    # otherwise
    data = Corpus(cv_obj.corpus, cv_obj.Vocab, cv_obj.Count,cv_obj.VocabDict)


    BSanalyzer = Bootstrapping()
    loadfilepath = "./init_aspect_word.txt"
    load_Aspect_Terms(BSanalyzer,loadfilepath,cv_obj.VocabDict)

    #### expand aspect keywords
    Add_Aspect_Keywords(BSanalyzer, 5, 5, data)

    savefilepath = './output/final_aspect_words.txt'
    save_Aspect_Keywords_to_file(BSanalyzer,savefilepath,cv_obj.Vocab)

    create_all_W(BSanalyzer,data)
    W_outputfolderpath = "./output/"
    produce_data_for_rating(BSanalyzer,data,W_outputfolderpath)

    print_summary_stats(data)

if __name__ == "__main__":
    run_bootstrap()

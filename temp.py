__author__ = 'zhangyin'

from src.CreateVocab import load_all_json_files

suffix="json"
folder="/Users/zhangyin/python projects/IR project/data/yelp mp1 data/"
corpus=load_all_json_files(folder,suffix)

for rest in corpus:
    print(rest.get('RestaurantInfo').get("Name"))

for rest in corpus:
    for review in rest.get("Reviews"):
        review.get("Overall")
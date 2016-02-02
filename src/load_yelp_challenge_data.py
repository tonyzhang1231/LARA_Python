__author__ = 'zhangyin'

from pprint import pprint
import json
from src.Structure import *

# read yelp challenge data
f = open("/Users/zhangyin/python projects/IR project/yelp_challenge_data/yelp_dataset_challenge_academic_dataset", 'r',encoding = "ISO-8859-1")
n = 0
type_list={}
for line in f:
    try:
        j = json.loads(line)
        try:
            if j.get("type") not in type_list:
                type_list[j.get("type")]=1
            else :
                type_list[j.get("type")]=type_list.get(j.get("type"))+1
        except AttributeError:
            pass
    except ValueError or UnicodeDecodeError:
        pass # invalid json
    n = n+1
f.close()
print(type_list)
print(n)

# {'user': 366714, 'review': 1569263, 'checkin': 45165, 'business': 61183, 'tip': 495106}
# 2540163

# f = open("/Users/zhangyin/python projects/IR project/yelp_challenge_data/yelp_dataset_challenge_academic_dataset", 'r',encoding = "ISO-8859-1")
# print(len(f.readlines()))
# f.close()
# 2540163 lines

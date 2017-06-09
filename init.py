import math
import os.path
from itertools import repeat
import glob
import re
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.datasets import fetch_20newsgroups


folder_list = []
main_dir = "20news-18828"
X_train = []
Y_train = []


def iterate_class_folders():
    all_folders = os.listdir(main_dir)
    return all_folders


def get_words(text):
    return re.compile('\w+').findall(text)


def extract_words(folder):
    for infile in glob.glob(os.path.join('{0}/{1}'.format(main_dir, folder), '*')):
        review_file = open(infile, 'r').read()
        words = get_words(review_file)
        X_train.append(words)
        Y_train.append(folder)
        # return


folder_list = iterate_class_folders()

# for folder in folder_list:
#     extract_words(folder)
extract_words(folder_list[0])
# stdSlr = StandardScaler().fit(X_train)
# X_train = stdSlr.transform(X_train)
index = 0
for item in X_train:
    result = chi2(item, Y_train[0])
    index += 1
end = 1

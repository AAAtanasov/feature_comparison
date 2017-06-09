import math
import os.path
from itertools import repeat
import glob
import re
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import  pickle


folder_list = []
main_dir = "20news-18828"
# X_train = []
# Y_train = []


# def iterate_class_folders():
#     all_folders = os.listdir(main_dir)
#     return all_folders
#
#
# def get_words(text):
#     return re.compile('\w+').findall(text)


# def extract_words(folder):
#     for infile in glob.glob(os.path.join('{0}/{1}'.format(main_dir, folder), '*')):
#         review_file = open(infile, 'r').read()
#         words = get_words(review_file)
#         X_train.append(words)
#         Y_train.append(folder)

#
# folder_list = iterate_class_folders()
# extract_words(folder_list[0])

categories = [
    # 'alt.atheism',
    # 'talk.religion.misc',
    # 'comp.graphics',
    'sci.space',
]


filtered = True
if filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names

y_train, y_test = data_train.target, data_test.target
useHashing = False
# if useHashing:
#     vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
#                                    n_features=2 ** 16)
#     X_train = vectorizer.transform(data_train.data)
# else:
#     vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                                  stop_words='english')
#     X_train = vectorizer.fit_transform(data_train.data)
cv = CountVectorizer()
hv = HashingVectorizer(stop_words='english', non_negative=True,
                               n_features=2 ** 16)
X_vector = hv.transform(data_train.data)
X_train = cv.fit_transform(data_train.data)
freqs = [(word, X_vector.getcol(idx).sum()) for word, idx in cv.vocabulary_.items()]

sorted_freq = sorted(freqs, key = lambda x: -x[1])

pickle.dump(sorted_freq, open("sorted.p", "wb"))

#sort from largest to smallest
# print sorted(freqs, key = lambda x: -x[1])
# result = list(cv.get_feature_names(), np.asarray(X_vector.sum(axis=0)).ravel())
# X_test = vectorizer.transform(data_test.data)

feature_names = cv.get_feature_names()
# result = zip(feature_names, np.asarray())

ch2 = SelectKBest(chi2, k=10)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
if feature_names:
    # keep selected feature names
    feature_names = [feature_names[i] for i
                     in ch2.get_support(indices=True)]

print(feature_names)
end = 1
import math
import os.path
from itertools import repeat
import glob
import re

folder_list = []
main_dir = "20news-18828"
files_list = []


def iterate_class_folders():
    all_folders = os.listdir(main_dir)
    return all_folders


def get_words(text):
    return re.compile('\w+').findall(text)


def extract_words(folder):
    for infile in glob.glob(os.path.join('{0}/{1}'.format(main_dir, folder), '*')):
        review_file = open(infile, 'r').read()
        words = get_words(review_file)
        files_list.append(words)


folder_list = iterate_class_folders()

for folder in folder_list:
    extract_words(folder)


end = 1

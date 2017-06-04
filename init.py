import math
import os.path
from itertools import repeat

"""
Array used to store all the images for 10-fold cross validation
"""
ten_fold_array = [[] for i in repeat(None, 10)]
sub_folders_list = []

def iterate_class_folders(number_of_classes):
    main_dir = "101_ObjectCategories"
    # sub_folders_list = []
    all_folder_names = os.listdir(main_dir)
    temp_folders = ['ant', 'accordion']
    # all_folder_names = temp_folders
    # all_folder_names.remove(".DS_Store")

    for tempIndex, _ in enumerate(range(number_of_classes)):
        choice_item_index = random.randrange(len(all_folder_names))
        choice_item = all_folder_names.pop(choice_item_index - 1)
        sub_folders_list.append(choice_item)

    for folder in sub_folders_list:
        retrieve_image_from_folder(folder)


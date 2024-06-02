import os

path_1 = "./your_output_folder/aorta"
path_2 = "./your_output_folder/inside_body"

list_1 = set(os.listdir(path_1))
list_2 = set(os.listdir(path_2))

print(list_1.intersection(list_2))

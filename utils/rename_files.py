import os
import sys

directory = sys.argv[1]
img_type = sys.argv[2]
filenames = os.listdir(directory)

counter = 1
for filename in filenames: 
	new_name = img_type + '_' + str(counter) + '.' + filename.split('.')[-1]
	os.rename(directory + '/' + filename, directory + '/' + new_name)
	counter += 1
import os
import sys

directory = sys.argv[0]
filenames = os.listdir(directory)

counter = 1
for filename in filenames: 
	new_name = directory + '_' + counter + filename.split('.')[-1]
	os.rename(filename, new_name)
	counter += 1
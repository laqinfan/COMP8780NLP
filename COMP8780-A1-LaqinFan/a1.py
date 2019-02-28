import os
import string
from collections import Counter

with open(input("Enter Filename: "),'r') as f:
	# define count to record the number of each word including punctuation 
	count = {}
	# A string without punctuation
	new_s = ""
	for line in f:
		line = line.replace("\n", " ")
		for word in line:
			if word in string.punctuation:
				if (word in count):
					count[word] = count[word] + 1
				else:
					count[word] = 1
			else:
				new_s = new_s + word

	#transform string into list
	words_list = new_s.lower().split(" ")

	for i in range(0, len(words_list)):
		if words_list[i] in count:
			count[words_list[i]] += 1
		else:
			count[words_list[i]] = 1



# remove empty string in dictionary
del count['']

# count the top 10 words
top10_count = Counter(count).most_common(10)

print ("Top 10 most frequent words: \n", top10_count)





 











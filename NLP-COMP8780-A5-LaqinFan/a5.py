import os
import string
import operator
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from collections import Counter, defaultdict
import re, sys
from nltk.util import ngrams

key_word = "(TOP ("
key = "TOP END_OF_TEXT_UNIT"

############################################################################
# 1. [10 points] From the SnapshotBROWN.pos.all.txt file extract all
# word types and their frequencies. Sort the list of word types in 
	# decreasing order based on their frequency. Draw a chart showing the
#relationship between the rank in the ordered list and the frequency
#(Zipf's Law). Do not stem but do ignore punctuation.
############################################################################

#read file SnapshotBROWN.pos.all.txt for later evaluation of tagger
with open("SnapshotBROWN.pos.all",'r') as f:
	# read file content
	content = "".join(line for line in f if not line.isspace())

#use key word(TOP) to split content into sentence list
sentence_list = content.split(key_word)

word_list = [] #store all the word in SnapshotBROWN.pos.all
value_list = [] # store all the tags in SnapshotBROWN.pos.all

for s in sentence_list:
	new_s = ""
	for w in s.split('('):
		if ')' in w:
			tag = w.split(')')[0]
			t = "".join(tag)
			if t[0] != '-' and t != key:
				new_s = new_s + t + " "
				t = tag.split(' ')
				if t[1] not in string.punctuation and t[1] != "``" and t[1] != "''" and t[1] != "--":
					word_list.append(t[1].lower())
					value_list.append(t[0])

# print(word_list)
# print(len(word_list))

##### Extract all word types and their frequencies
word_dict = {}
for w in word_list:
	count = word_list.count(w)
	word_dict[w] = count

##### Sort the list of word types in decreasing order based on their frequency.
new_worddict ={}
sorted_d = sorted(word_dict, key=word_dict.get, reverse=True)
for w in sorted_d:
	new_worddict[w] = word_dict[w]
# print(new_worddict)


##### Draw a chart showing the relationship between the rank in the ordered list and the frequency
# (Zipf's Law).
word = []
freq = []
i = 1
idx = []
for w, count in Counter(new_worddict).most_common(20):
	word.append(w)
	idx.append(i)
	i = i + 1
	freq.append(count)

# Draw chart(Zipf's Law)
# plt.xticks(idx, word)
# plt.plot(idx, freq)
# plt.show()


############################################################################
# 2. [20 points] Generate a Bigram Grammar from the above file. Perform
#    add-one smoothing. Show the grammar before and after smoothing for
#    the sentence "A similar resolution passed in the Senate".
############################################################################
def bigram_nlk(s):
	tokens = [token for token in s.split(" ") if token != ""]
	bigram = list(ngrams(tokens, 2))
	return bigram

###Without add-one smoothing
def without_sm(bigram):
	distinct_list = []
	for g in bigram:
		if g not in distinct_list:
			distinct_list.append(g)

	prob_dict = {}
	for w in distinct_list:
		count_bigram = bigram.count(w)
		word = w[0]
		if word in new_worddict.keys():
			prob = float(count_bigram/new_worddict[word])
		else:
			prob = float(count_bigram/1.0)
		prob = int(prob * 10000) / 10000.0
		prob_dict[w] = prob
	return prob_dict

###With add-one smoothing
def with_sm(unigram, bigram):
	distinct_list = []
	for g in bigram:
		if g not in distinct_list:
			distinct_list.append(g)
	prob_dict_smooth = {}
	for w in distinct_list:
		count_bigram = bigram.count(w)
		word = w[0]
		if word in new_worddict.keys():
			prob = float((count_bigram+1)/(new_worddict[word]+len(unigram)))
		else:
			prob = float((count_bigram+1)/(1+len(unigram)))

		prob = int(prob * 10000) / 10000.0
		prob_dict_smooth[w] = prob

	return prob_dict_smooth

s = ' '.join(word_list)
bigram = bigram_nlk(s)
prob_withoutsm = without_sm(bigram)
print(prob_withoutsm)

unigram = []
for w in word_list:
	if w not in unigram:
		unigram.append(w)
prob_withsm = with_sm(unigram, bigram)
print(prob_withsm)

###Example: "A similar resolution passed in the Senate".
sentence = "A similar resolution passed in the Senate"
new_s = sentence.lower()

bigram_output = bigram_nlk(new_s)
print(bigram_output)

###before smoothing
test_prob_without = without_sm(bigram_output)
print(test_prob_without)

###after smoothing
test_prob_with = with_sm(unigram, bigram_output)
print(test_prob_with)


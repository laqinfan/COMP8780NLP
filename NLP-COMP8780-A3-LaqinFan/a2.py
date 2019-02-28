import os
import string
from collections import Counter, defaultdict

key_word = "(TOP ("
key = "TOP END_OF_TEXT_UNIT"

with open(input("Enter  BROWN Corpus Filename: (BROWN.pos.all) "),'r') as f:
	# read file content
	content = "".join(line for line in f if not line.isspace())

#use key word(TOP) to split content into sentence list
sentence_list = content.split(key_word)

#write each sentence into ouput file
output = open("BROWN-clean.pos.txt", "w")

word_list = []
value_list = []

for s in sentence_list:
	new_s = ""
	for w in s.split('('):
		if ')' in w:
			tag = w.split(')')[0]
			t = "".join(tag)
			if t[0] != '-' and t != key:
				new_s = new_s + t + " "
				t = tag.split(' ')
				word_list.append(t[1].lower())
				value_list.append(t[0])
	output.write(new_s + "\n")
output.close()

# define word_tag to record tag and corresponding word
word_tag = {}
word_tag = {k:v for k, v in zip(word_list,value_list )}

#define a hash: key is word, value is hash
word_tag_hash = {}

i = iter(word_list)
j = iter(value_list)
m = list(zip(i,j))

# get hash of hashes with empty frequency
for k, v in word_tag.items():
	word_tag_hash[k] = {}
	for k1,v1 in m:
		if k1 == k:
			if v != v1:
				word_tag_hash[k][v1] = 0
			else:
				word_tag_hash[k][v] = 0


#Read file into string list
f = open("BROWN-clean.pos.txt", "r")
string_list = f.read().split()
# print(string_list)

#Count the frequency of each tag of corresponding word
count = 0
for k in word_tag_hash.keys():
	for k1, v1 in word_tag_hash[k].items():
		for i in range(len(string_list)):
			if k == string_list[i].lower() and k1 == string_list[i-1]:
				word_tag_hash[k][k1] += 1

# print(word_tag_hash)














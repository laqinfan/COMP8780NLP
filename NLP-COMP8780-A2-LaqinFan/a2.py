import os
import string
from collections import Counter, defaultdict

############################################################################
# [10 points] Write a Perl script that maps each parse tree in the
#    SnapshotBROWN.pos.all.txt file (see the website) into one-line
#    sentences as shown below. You should retain only the parts-of-speech
#    and the words from the parse trees. Each sentence should span a single
#    line in the outpute file.
############################################################################
key_word = "(TOP ("
key = "TOP END_OF_TEXT_UNIT"

with open(input("Enter Filename: "),'r') as f:
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
			#only retain POS tags and words 
			if t[0].isalpha() and t != key:
				new_s = new_s + t + " "
				t = tag.split(' ')
				word_list.append(t[1].lower())
				value_list.append(t[0])
	output.write(new_s + "\n")
output.close()


##############################################################
# [10 points] Generate the hash of hashes from the clean file 
# BROWN-clean.pos.txt .
##############################################################

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

##################################################################
# [10 points] In BROWN-clean.pos.txt detect the 20 most frequent
# tags. Report their frequency.
##################################################################

#define tag_fre to record tag and its frequency
tag_fre = {}
for k, v in word_tag_hash.items():
	for k1, v1 in v.items():
		if k1 in tag_fre:
			tag_fre[k1] += v1
		else:
			tag_fre[k1] = v1

# count the top 20 frequent tags in the BROWN-clean file
top20_count = Counter(tag_fre).most_common(20)
print ("Top 20 most frequent tags: \n", top20_count)


#############################################################################
# [10 points] take the most frequent tag and use it to
#    tag the words in all the sentences from the BROWN-clean.pos.txt file. 
#    Report the performance of this tagger. See the slides for details on 
#    how to measure the performance.
#############################################################################

############## The solution 1 #####################
# get the most frequent tag

# new_tags = []
# tag = Counter(tag_fre).most_common(1)[0][0]

# #use the most frequent tag to tag all the words in the file
# for w in word_list:
# 	new_tags.append(tag)

# count = 0
# #calculate the accuracy of this new tagger
# if len(new_tags) != len(value_list):
#     raise ValueError("Lists must have the same length.") 
# else:
# 	for mtag, tag in zip(new_tags,value_list):
# 		if mtag == tag:
# 			count += 1
# accuracy = float(count)/float(len(value_list))

# print("Solution 1 Performance of this tagger by accuracy: {0:.0%}".format(accuracy))




############## The solution 2 #####################

new_tags = []
#use the most frequent tag of each word to tag the word
for w in word_list:
	tag = max(word_tag_hash[w], key=lambda i: word_tag_hash[w][i])
	new_tags.append(tag)

count = 0
#calculate the accuracy of this new tagger
if len(new_tags) != len(value_list):
    raise ValueError("Lists must have the same length.") 
else:
	for tag1, tag2 in zip(new_tags,value_list):
		if tag1 == tag2:
			count += 1
accuracy = float(count)/float(len(value_list))

print("Performance of this tagger by accuracy: {0:.0%}".format(accuracy))










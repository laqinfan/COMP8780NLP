import os
import string
from collections import Counter, defaultdict
import re



############################################################################
# 1. [20 points] Extract from the BROWN file all grammar rules embedded in
#    parse trees. Do not consider punctuation as a nonterminal. Eliminate
#    numbers attached to non-terminals such as '-1', '-2', etc. Report 
#    how many distinct rules you found, what are the 10 most frequent
#    rules regardless of the non-terminal on the left-hand side, and
#    what is the non-terminal with the most alternate rules (i.e. the
#    non-terminal that can have most diverse structures). 
############################################################################

grammar_rules = []
distinct_rules = []
parsed_rules = []

# Find all grammar rules
def processGrammar(grammar, sentence):
	#start from the terminal rules, only keep the tag, remove the words
    for g in grammar:
        grammar = g.split(" ")
        tag = grammar[1]        
        sentence = sentence.replace(g, tag)

    inner_grammar = re.findall("\([^\(\)]*\)", sentence)
	
	#Parse the tree and find all (lhs rhs) rule
    while len(inner_grammar) > 0: 
	    for g in inner_grammar:
	        grammar = g.split(" ")
	        lhs = grammar[1]
	        rhs = grammar[2:-1]

	        grammar_rules.append((lhs, rhs))
	        sentence = sentence.replace(g, lhs)

	    inner_grammar = re.findall("\([^\(\)]*\)", sentence)

# Find distinct rules
# Use grammar_rules as input
def findDinstinct(grammar_rules):
	for g in grammar_rules:
		if g[0] not in string.punctuation and g[0] != "``" and g[0] != "''" and g[0] != "TOP":
			l = g[0] + " -> "
			r = ""
			for t in g[1]:
				# remove NONE and Punctuation
				if t != "``" and t != "''" and t != "-NONE-" and t is not string.punctuation:
					r += t + " "
			l = l + r
			parsed_rules.append(l)

key = "(TOP END_OF_TEXT_UNIT)"
with open(input("Enter Filename: "),'r') as f:
	# read file content
	content = "".join(line.strip() for line in f if not line.isspace())

sentence_str = content.replace(key,"").replace("("," ( ").replace(")"," ) ").replace("  ", " ")

# find inner parenthesis
inner_grammar = re.findall("\([^\(\)]*\)", sentence_str)

processGrammar(inner_grammar, sentence_str)
# print(grammar_rules)

findDinstinct(grammar_rules)


############################################################################
#    how many distinct rules you found
############################################################################
for r in parsed_rules:
	if r not in distinct_rules:
		distinct_rules.append(r)
dis_count = len(distinct_rules)
# print(distinct_rules)
print(dis_count)
print ("The number of distinct rules : ", dis_count)
# print(parsed_rules)


############################################################################
#    what are the 10 most frequent
#    rules regardless of the non-terminal on the left-hand side
############################################################################
d = {}

for g in parsed_rules:
	count = parsed_rules.count(g)
	d[g] = count
	
top10_count = Counter(d).most_common(10)
print ("Top 10 most frequent rules: \n", top10_count)


############################################################################
#    what is the non-terminal with the most alternate rules?
############################################################################

print("\nThe non-terminal with the most alternative rules: ", Counter(d).most_common(1)[0])









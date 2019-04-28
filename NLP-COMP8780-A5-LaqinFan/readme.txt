Input file: SnapshotBROWN.pos.all

P1.
1. Sort the list of word types in descending order:
	'the': 126, 'of': 53, 'to': 39, 'a': 35, 'and': 32, 'in': 29, 'that': 22, 'for': 20, 'jury': 18, 		'said': 18, "'s": 17, 'be': 17, 'fulton': 14, 'it': 14, 'was': 12, 'will': 12, ...

	Draw the zipf law figure.
	In current folder, please check the figure: zipf-law-of-word-distribution.png

2. Generate a Bigram Grammar from the above file. Perform add-one smoothing
### Before smoothing: (bigram and probability)
{('*start*', 'fulton'): 0.0015, ('fulton', 'county'): 0.4285, ('county', 'grand'): 0.1111, ('grand', 'jury'): 0.75, ('jury', 'said'): 0.4444, ('said', 'friday'): 0.0555, ('friday', 'an'): 0.3333, ('an', 'investigation'): 0.2, ('investigation', 'of'): 1.0, ('of', 'atlanta'): 0.0566, ('atlanta', "'s"): 0.2857, ("'s", 'recent'): 0.0588, ('recent', 'primary'): 1.0, ('primary', 'election'): 0.25,.....,}

### After smoothing: (bigram and probability)
{('*start*', 'fulton'): 0.0015, ('fulton', 'county'): 0.0108, ('county', 'grand'): 0.0031, ('grand', 'jury'): 0.0062, ('jury', 'said'): 0.0138, ('said', 'friday'): 0.003, ('friday', 'an'): 0.0031, ('an', 'investigation'): 0.0031, ('investigation', 'of'): 0.0031, ('of', 'atlanta'): 0.0058, ('atlanta', "'s"): 0.0046, ("'s", 'recent'): 0.003, ('recent', 'primary'): 0.0031,.......}


###Show the grammar before and after smoothing for the sentence "A similar resolution passed in the Senate".
[('*start*', 'similar'), ('similar', 'resolution'), ('resolution', 'passed'), ('passed', 'in'), ('in', 'the'), ('the', 'senate'), ('senate', '*end*')]
Total Probability before smoothing:  9.17e-13
Total Probability after smoothing:  2.53e-19




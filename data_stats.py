import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def get_frequency_dict(filename):
	freq_dict = {}
	with open(filename) as f:
		for line in f:
			num_words = len(line.split())
			if num_words in freq_dict:
				freq_dict[num_words] += 1
			else: 
				freq_dict[num_words] = 1 
	return freq_dict

def plot_histogram(freq_dict, xlabel=''):
	print xlabel, ': ', freq_dict 
	plt.bar(freq_dict.keys(), freq_dict.values())
	plt.xlabel(xlabel)
	plt.ylabel('Frequency')
	plt.show()


#plot_histogram(get_frequency_dict('../data/squad/train.question'), xlabel='Question length')
plot_histogram(get_frequency_dict('../data/squad/train.context'), xlabel='Context length')
#plot_histogram(get_frequency_dict('../data/squad/train.answer'), xlabel='Answer length')


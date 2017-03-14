import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def get_frequency_dict(filename):
	freq_dict = {}
	with open(filename) as f:
		num_lines = 0
		for line in f:
			num_lines += 1
			num_words = len(line.split())
			if num_words in freq_dict:
				freq_dict[num_words] += 1
			else: 
				freq_dict[num_words] = 1 
	print("Num lines: ", num_lines)
	return freq_dict 

def plot_histogram(freq_dict, xlabel=''):
	print xlabel, ': ', freq_dict 
	plt.bar(freq_dict.keys(), freq_dict.values())
	plt.xlabel(xlabel)
	plt.ylabel('Frequency')
	plt.show()


def find_max_id(filename):
	max_id = 0 
	with open(filename) as f:
		for line in f:
			for id_str in line.split():
				id_int = int(id_str)
				if id_int > max_id: 
					max_id = id_int 
	return max_id

plot_histogram(get_frequency_dict('../data/squad/val.ids.question'), xlabel='Question length')
#plot_histogram(get_frequency_dict('../data/squad/train.ids.context'), xlabel='Context length')
#plot_histogram(get_frequency_dict('../data/squad/train.span'), xlabel='Answer length')
#print(find_max_id('../data/squad/train.span'))

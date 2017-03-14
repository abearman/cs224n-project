from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random
import math

import numpy as np
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits	

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
		if opt == "adam":
				optfn = tf.train.AdamOptimizer
		elif opt == "sgd":
				optfn = tf.train.GradientDescentOptimizer
		else:
				assert (False)
		return optfn


class GRUAttnCell(rnn_cell.GRUCell):
		def __init__(self, num_units, encoder_output, scope=None):
			self.hs = encoder_output	# Source hidden state
			super(GRUAttnCell, self).__init__(num_units)	

		def __call__(self, inputs, state, scope=None):
			gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
			with vs.variable_scope(scope or type(self).__name__):
				with vs.variable_scope("Attn"):
					ht = rnn_cell._linear(gru_out, self._num_units, True, 1.0)
					ht = tf.expand_dims(ht, axis=1)
				scores = tf.reduce_sum(self.hs * ht, reduction_indices=2, keep_dims=True)
				context = tf.reduce_sum(self.hs * scores, reduction_indices=1)
				with vs.variable_scope("AttnConcat"):
					out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))

			return (out, out)


class Encoder(object):
		def __init__(self, state_size, embedding_size):	# vocab_dim == embed_size
				self.state_size = state_size
				self.embedding_size = embedding_size 

		def encode(self, question_embeddings, context_embeddings, question_mask, context_mask, 
							 encoder_state_input, dropout_keep_prob):
				"""
				In a generalized encode function, you pass in your inputs,
				masks, and an initial
				hidden state input into this function.

				:param inputs: Symbolic representations of your input
				:param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
											through masked steps
				:param encoder_state_input: (Optional) pass this as initial hidden state
																		to tf.nn.dynamic_rnn to build conditional representations
				:return: an encoded representation of your input.
								 It can be context-level representation, word-level representation,
								 or both.
				"""
				with vs.variable_scope("encoder", True):

					# Encode question
					#with vs.variable_scope("question", True):			
					lstm_cell = tf.nn.rnn_cell.LSTMCell(self.state_size)	# Should be 1 at first, then 200
					lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
					question_length = tf.reduce_sum(tf.cast(question_mask, tf.int32), reduction_indices=1)	
					print("Question length: ", question_length)
					(fw_out, bw_out), _ = bidirectional_dynamic_rnn(lstm_cell, lstm_cell, 
								question_embeddings, sequence_length=question_length,
								time_major=False, dtype=tf.float64, swap_memory=True)  #TODO: time_major=True was causing seg faults
					last_fw = fw_out[:,-1,:]
					print("last fw: ", last_fw)
					#h_q = tf.concat(1, [fw_out[:,-1,:], bw_out[:,-1,:]])
					h_q = fw_out[:,-1,:] + bw_out[:,-1,:]
					print("h_q after: ", h_q)
					#H_q = tf.concat(2, [fw_out, bw_out])
					H_q = fw_out + bw_out
					print("H_q: ", H_q)
	
					#with vs.variable_scope("context", True):
						# Encode context paragraph
					context_length = tf.reduce_sum(tf.cast(context_mask, tf.int32), reduction_indices=1)
					print("Context length: ", context_length)
					attn_cell = GRUAttnCell(self.state_size, H_q)  
					H_p, _ = dynamic_rnn(attn_cell, context_embeddings, sequence_length=context_length, dtype=tf.float64)
					h_p = H_p[:,-1,:]
					print("h_p after: ", h_p)

					return h_q, h_p

class Decoder(object):
		def __init__(self, output_size):
				self.output_size = output_size

		def decode(self, h_q, h_p):
				"""
				takes in a knowledge representation
				and output a probability estimation over
				all paragraph tokens on which token should be
				the start of the answer span, and which should be
				the end of the answer span.

				:param knowledge_rep: it is a representation of the paragraph and question,
															decided by how you choose to implement the encoder
				:return:
				"""
				with vs.variable_scope("answer_start"):
					a_s = rnn_cell._linear([h_q, h_p], self.output_size, True, 1.0)
				with vs.variable_scope("answer_end"):
					a_e = rnn_cell._linear([h_q, h_p], self.output_size, True, 1.0)

				return (a_s, a_e)

class QASystem(object):
		def __init__(self, encoder, decoder, **kwargs):
				"""
				Initializes your System

				:param encoder: an encoder that you constructed in train.py
				:param decoder: a decoder that you constructed in train.py
				:param args: pass in more arguments as needed
				"""
				# Dataset constants
				self.max_question_len = 60		# Longest question sequence to parse (in train or val set)
				self.max_context_len = 766		# Longest context sequence to parse (in train or val set): (766, truncated at 750)
				self.max_answer_len = 46			# Longest answer sequence to parse (in train or val set)
				self.n_classes = 2						# O or ANSWER

				# Model saver
				self.saver = None 

				# Encoder and decoder
				self.encoder = encoder
				self.decoder = decoder

				# ==== Set up handles to embeddings, preds, loss, and train_op ======
				# Create a handle to the pretrained embeddings
				self.question_embeddings = None
				self.context_embeddings = None

				# Create a handle to the prediction probabilities
				self.a_s_probs = None
				self.a_e_probs = None

				# Create a handle to loss and train_op
				self.loss = None	
				self.train_op = None
				self.grad_norm = None

				# kwargs passed in
				self.state_size = kwargs['state_size']
				self.embed_path = kwargs['embed_path']
				self.embedding_size = kwargs['embedding_size']
				self.output_size = kwargs['output_size']
				self.optimizer = kwargs['optimizer']	# 'adam' or 'sgd'

				self.initial_learning_rate = kwargs['learning_rate']
				self.global_step = tf.Variable(0, trainable=False)	

				self.epochs = kwargs['epochs']
				self.batch_size = kwargs['batch_size']
				self.max_gradient_norm = kwargs['max_gradient_norm']
				self.dropout_keep_prob = kwargs['dropout_keep_prob']
				self.train_dir = kwargs['train_dir']

				# ==== set up placeholder tokens ========
				self.question_input_placeholder = tf.placeholder(tf.int32, (None, self.max_question_len))
				self.context_input_placeholder = tf.placeholder(tf.int32, (None, self.max_context_len))

				self.question_mask_placeholder = tf.placeholder(tf.bool, (None, self.max_question_len))
				self.context_mask_placeholder = tf.placeholder(tf.bool, (None, self.max_context_len))

				#self.labels_placeholder = tf.placeholder(tf.int32, (None, self.n_classes))
				self.start_answer_placeholder = tf.placeholder(tf.int32, (None, self.max_context_len))
				self.end_answer_placeholder = tf.placeholder(tf.int32, (None, self.max_context_len))

				self.dropout_placeholder = tf.placeholder(tf.float32, ())


				# ==== assemble pieces ====
				with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
						self.setup_embeddings()
						self.setup_system()		# Equivalent to: self.add_prediction_op() 
						self.setup_loss()  # Equivalent to: self.add_loss_op(self.pred) and self.add_training_op(self.loss) 

				# ==== set up training/updating procedure ====
				self.setup_training_op()


		def setup_system(self):
				"""
				After your modularized implementation of encoder and decoder
				you should call various functions inside encoder, decoder here
				to assemble your reading comprehension system!
				:return:
				"""	
				# Set up prediction op	
				h_q, h_p = self.encoder.encode(self.question_embeddings, self.context_embeddings, 
																			 self.question_mask_placeholder, self.context_mask_placeholder, 
																			 None, self.dropout_keep_prob)
				self.a_s_probs, self.a_e_probs = self.decoder.decode(h_p, h_q)


		def setup_loss(self):
				"""
				Set up your loss computation here
				:return:
				"""
				# The label is a one-hot representation
				with vs.variable_scope("loss"):
					print("a_s_probs: ", self.a_s_probs)
					print("start answer: ", self.start_answer_placeholder)
					l1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.a_s_probs, labels=self.start_answer_placeholder)
					l2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.a_e_probs, labels=self.end_answer_placeholder)
					self.loss = l1 + l2 

		def setup_training_op(self):
				"""
				Sets up the training ops.

				Creates an optimizer and applies the gradients to all trainable variables.
				Clips the global norm of the gradients.
				"""
				# Update learning rate
				lr = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, 100, 0.96)

				opt = get_optimizer(self.optimizer)(learning_rate=lr)

				# Get the gradients using optimizer.compute_gradients
				gradients, params = zip(*opt.compute_gradients(self.loss))
				for param in params:
					print("Param: ", param)

				# Clip the gradients to self.max_gradient_norm
				gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
	
				# Re-zip the gradients and params
				grads_and_params = zip(gradients, params)

				# Compute the resultant global norm of the gradients and set self.grad_norm
				self.grad_norm = tf.global_norm(grads_and_params)

				# Create the training operation by calling optimizer.apply_gradients
				self.train_op = opt.apply_gradients(grads_and_params, global_step=self.global_step)
				#self.train_op = get_optimizer(self.optimizer)(learning_rate=lr).minimize(self.loss, global_step=self.global_step)


		def setup_embeddings(self):
				"""
				Loads distributed word representations based on placeholder tokens
				:return:
				"""
				with vs.variable_scope("embeddings"):
					# Step 1: Load the embeddings from the npz file
					pretrained_embeddings = np.load(self.embed_path)['glove']

					# Step 2: Assign the embeddings
					self.question_embeddings = tf.constant(pretrained_embeddings, name="question_embeddings")
					self.question_embeddings = tf.nn.embedding_lookup(self.question_embeddings, self.question_input_placeholder)
					self.question_embeddings = tf.reshape(self.question_embeddings, [-1, self.max_question_len, self.embedding_size])

					self.context_embeddings = tf.constant(pretrained_embeddings, name="context_embeddings")
					self.context_embeddings = tf.nn.embedding_lookup(self.context_embeddings, self.context_input_placeholder)
					self.context_embeddings = tf.reshape(self.context_embeddings, [-1, self.max_context_len, self.embedding_size])


		# This function is like "train_on_batch" in Assignment 3
		# train_x is like inputs_batch, and train_y is like labels_batch
		def optimize(self, session, train_batch, dropout_keep_prob=1):
				"""
				Takes in actual data to optimize your model
				This method is equivalent to a step() function
				:return:
				"""
				question_batch, context_batch, question_mask_batch, context_mask_batch, start_answer_batch, end_answer_batch = zip(*train_batch)

				feed = {}
				feed[self.question_input_placeholder] = question_batch
				feed[self.context_input_placeholder] = context_batch
				feed[self.question_mask_placeholder] = question_mask_batch
				feed[self.context_mask_placeholder] = context_mask_batch
				feed[self.start_answer_placeholder] = start_answer_batch
				feed[self.end_answer_placeholder] = end_answer_batch
				feed[self.dropout_placeholder] = dropout_keep_prob

				print("Question batch: ", self.question_input_placeholder)
				print("Context batch: ", self.context_input_placeholder)
				print("Start answer batch: ", self.start_answer_placeholder)
				print("End answer batch: ", self.end_answer_placeholder)
				print("Question mask batch: ", self.question_mask_placeholder)
				print("Context mask batch: ", self.context_mask_placeholder)

				_, loss, grad_norm = session.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
				#_, loss = session.run([self.train_op, self.loss], feed_dict=feed)
				print("a_s probs: ", self.a_s_probs.eval(feed_dict=feed, session=session))
				print("a_e probs: ", self.a_e_probs.eval(feed_dict=feed, session=session))

				# qa/answer_start/Linear/Matrix/
				for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
					print(param.name)
					print(np.unique(param.eval()))
					print("")

				#print("h_q: ", self.h_q.eval(feed_dict=feed, session=session))
				#print("h_p: ", self.h_p.eval(feed_dict=feed, session=session))
				#print("h_q unique: ", np.unique(self.h_q.eval(feed_dict=feed, session=session)))
				#print("h_p unique: ", np.unique(self.h_p.eval(feed_dict=feed, session=session)))
				return loss, grad_norm	# TODO
				#return loss


		def test(self, session, valid_x, valid_y):
				"""
				in here you should compute a cost for your validation set
				and tune your hyperparameters according to the validation set performance
				:return:
				"""
				input_feed = {}

				# fill in this feed_dictionary like:
				input_feed['valid_x'] = valid_x
				if valid_y is not None:
					input_feed['valid_y'] = valid_y

				output_feed = []

				outputs = session.run(output_feed, input_feed)

				return outputs

		def decode(self, session, test_x):
				"""
				Returns the probability distribution over different positions in the paragraph
				so that other methods like self.answer() will be able to work properly
				:return:
				"""
				input_feed = {}

				# fill in this feed_dictionary like:
				# input_feed['test_x'] = test_x

				output_feed = []

				outputs = session.run(output_feed, input_feed)

				return outputs

		def answer(self, session, test_x):

				yp, yp2 = self.decode(session, test_x)

				a_s = np.argmax(yp, axis=1)
				a_e = np.argmax(yp2, axis=1)

				return (a_s, a_e)

		def validate(self, sess, valid_dataset):
				"""
				Iterate through the validation dataset and determine what
				the validation cost is.

				This method calls self.test() which explicitly calculates validation cost.

				How you implement this function is dependent on how you design
				your data iteration function

				:return:
				"""
				valid_cost = 0

				for valid_x, valid_y in valid_dataset:
					valid_cost = self.test(sess, valid_x, valid_y)


				return valid_cost

		def evaluate_answer(self, session, dataset, sample=100, log=False):
				"""
				Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
				with the set of true answer labels

				This step actually takes quite some time. So we can only sample 100 examples
				from either training or testing set.

				:param session: session should always be centrally managed in train.py
				:param dataset: a representation of our data, in some implementations, you can
												pass in multiple components (arguments) of one dataset to this function
				:param sample: how many examples in dataset we look at
				:param log: whether we print to std out stream
				:return:
				"""
				random_indices = [random.random(0, len(dataset)) for _ in range(sample)]
				batch = dataset[random_indices]

				answer = p[a_s, a_e + 1]
				true_answer = p[true_s, true_e + 1]
				f1_score(answer, true_answer)
				exact_match_score(answer, true_answer)

				f1 = 0.
				em = 0.

				if log:
						logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

				return f1, em

		def train(self, session, dataset, train_dir):
				"""
				Implement main training loop

				TIPS:
				You should also implement learning rate annealing (look into tf.train.exponential_decay)
				Considering the long time to train, you should save your model per epoch.

				More ambitious appoarch can include implement early stopping, or reload
				previous models if they have higher performance than the current one

				As suggested in the document, you should evaluate your training progress by
				printing out information every fixed number of iterations.

				We recommend you evaluate your model performance on F1 and EM instead of just
				looking at the cost.

				:param session: it should be passed in from train.py
				:param dataset: a representation of our data, in some implementations, you can
												pass in multiple components (arguments) of one dataset to this function
				:param train_dir: path to the directory where you should save the model checkpoint
				:return:
				"""

				# some free code to print out number of parameters in your model
				# it's always good to check!
				# you will also want to save your model parameters in train_dir
				# so that you can use your trained model to make predictions, or
				# even continue training

				dataset = self.preprocess_dataset(dataset)

				init = tf.global_variables_initializer() 
				session.run(init)
				self.saver.save(session, self.train_dir + "/baseline_model_0") 

				for epoch in range(self.epochs):	
					logging.info("Epoch %d out of %d", epoch + 1, self.epochs)
					self.run_epoch(session, dataset['train'])

					# Save model
					self.saver.save(session, self.train_dir + "/baseline_model_" + str(epoch))

					tic = time.time()
					params = tf.trainable_variables()
					num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
					toc = time.time()
					logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

		
		# Some more preprocessing to make all the questions and context sequences the same length.
		# Also adds two 1-hot vectors for start_answer and end_answer
		def preprocess_dataset(self, dataset):
			for data_subset_name in ['train', 'val']:
				# Iterate over each triplet in the dataset (train or val)
				for i in range(len(dataset[data_subset_name])):  
					# Pad the question and context sequence to their respective max lengths
					# Do this in place in the dataset
					question = dataset[data_subset_name][i][0]
					context = dataset[data_subset_name][i][1]
					answer = dataset[data_subset_name][i][2]

					padded_question, question_mask = self.pad_sequence(question, self.max_question_len)
					padded_context, context_mask = self.pad_sequence(context, self.max_context_len)
					start_answer = [0] * self.max_context_len
					start_answer[answer[0]] = 1
					end_answer = [0] * self.max_context_len
					end_answer[answer[1]] = 1

					dataset[data_subset_name][i] = (padded_question, padded_context, 
								question_mask, context_mask, start_answer, end_answer)
 
			return dataset


		def pad_sequence(self, sequence, max_length):
			new_sequence = []
			mask = []

			if len(sequence) >= max_length:  # Truncate (or fill exactly)
				new_sequence = sequence[0:max_length]
				mask = [True] * max_length

			elif len(sequence) < max_length:	# Append 0's
				delta = max_length - len(sequence)
				new_sequence = sequence + ([0]*delta)
				mask = ([True] * len(sequence)) + ([False] * delta)

			return (new_sequence, mask)

	
		# A single training example is a triplet: (question, context, answer). Each entry 
		# of the triplet is a list of word IDs.
		# Each batch only has training examples (no val examples).
		def run_epoch(self, session, train_examples):
				#for i, batch in enumerate(self.get_tiny_batches(train_examples)):
				for i, batch in enumerate(self.minibatches(train_examples, self.batch_size, shuffle=True)):
						print("Global step: ", self.global_step.eval())
						loss, grad_norm = self.optimize(session, batch, self.dropout_keep_prob) #TODO	
						#loss = self.optimize(session, batch, self.dropout_keep_prob)	
						print("Loss: ", loss, " , grad norm: ", grad_norm)
						#print("Loss: ", loss)
		

		def get_tiny_batches(self, data):
				return [data[0:10], data[10:20]]
			

		# Partitioning code from: 
		# http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
		def minibatches(self, data, batch_size, shuffle=True):
				if shuffle:
					random.shuffle(data)
				num_batches = int(math.ceil(len(data) / batch_size))
				q, r = divmod(len(data), num_batches)
				indices = [q*i + min(i, r) for i in xrange(num_batches+1)]
				return [data[indices[i]:indices[i+1]] for i in xrange(num_batches)]




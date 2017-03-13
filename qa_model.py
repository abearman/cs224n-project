from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

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

		def encode(self, inputs, question_mask, context_mask, encoder_state_input):
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
				# Encode question
				question_inputs = inputs['question']
				print("question inputs: ", question_inputs)
				question_length = tf.reduce_sum(tf.cast(question_mask, tf.int32), reduction_indices=0)
				lstm_cell = tf.nn.rnn_cell.LSTMCell(self.state_size)	# Should be 1 at first, then 200
				(fw_out, bw_out), _ = bidirectional_dynamic_rnn(lstm_cell, lstm_cell, 
							question_inputs, sequence_length=question_length,
							time_major=True, dtype=tf.float64)
				last_fw = fw_out[:,-1,:]
				print("last fw: ", last_fw)
				#h_q = tf.concat(1, [fw_out[:,-1,:], bw_out[:,-1,:]])
				h_q = fw_out[:,-1,:] + bw_out[:,-1,:]
				print("h_q: ", h_q)
				#H_q = tf.concat(2, [fw_out, bw_out])
				H_q = fw_out + bw_out
				print("H_q: ", H_q)

				# Encode context paragraph
				context_inputs = inputs['context']
				context_length = tf.reduce_sum(tf.cast(context_mask, tf.int32), reduction_indices=0)
				attn_cell = GRUAttnCell(self.state_size, H_q)  
				with vs.variable_scope("encoder", True):	# reuse
					H_p, _ = dynamic_rnn(attn_cell, inputs['context'], sequence_length=context_length, dtype=tf.float64)
				h_p = H_p[:,-1,:]

				print("h_q: ", h_q)
				print("h_p: ", h_p)
				return h_q, h_p

class Decoder(object):
		def __init__(self, output_size):
				self.output_size = output_size

		def decode(self, knowledge_rep):
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
				h_q = knowledge_rep["h_q"]
				h_p = knowledge_rep["h_p"]
				with vs.variable_scope("answer_start"):
					a_s = rnn_cell._linear([h_q, h_p], self.output_size, True)
				with vs.variable_scope("answer_end"):
					a_e = rnn_cell._linear([h_q, h_p], self.output_size, True)

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
				self.max_context_len = 766		# Longest context sequence to parse (in train or val set)
				self.max_answer_len = 46			# Longest answer sequence to parse (in train or val set)
				self.n_classes = 2						# O or ANSWER

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

				# kwargs passed in
				self.embed_path = kwargs['embed_path']
				self.embedding_size = kwargs['embedding_size']
				self.output_size = kwargs['output_size']
				self.optimizer = kwargs['optimizer']	# 'adam' or 'sgd'
				self.learning_rate = kwargs['learning_rate']
				self.epochs = kwargs['epochs']
				self.batch_size = kwargs['batch_size']

				# ==== set up placeholder tokens ========
				self.question_input_placeholder = tf.placeholder(tf.int32, (None, self.max_question_len, 1))
				self.context_input_placeholder = tf.placeholder(tf.int32, (None, self.max_context_len, 1))

				self.question_mask_placeholder = tf.placeholder(tf.bool, (None, self.max_question_len))
				self.context_mask_placeholder = tf.placeholder(tf.bool, (None, self.max_context_len))

				self.labels_placeholder = tf.placeholder(tf.int32, (None, self.n_classes))
				self.dropout_placeholder = tf.placeholder(tf.float32, ())


				# ==== assemble pieces ====
				with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
						self.setup_embeddings()
						self.setup_system()		# Equivalent to: self.add_prediction_op() 
						self.setup_loss()  # Equivalent to: self.add_loss_op(self.pred) and self.add_training_op(self.loss) 

				# ==== set up training/updating procedure ====
				pass


		def setup_system(self):
				"""
				After your modularized implementation of encoder and decoder
				you should call various functions inside encoder, decoder here
				to assemble your reading comprehension system!
				:return:
				"""	
				# Set up prediction op	
				inputs = {}
				inputs['question'] = self.question_embeddings 
				inputs['context'] = self.context_embeddings
				h_q, h_p = self.encoder.encode(inputs, self.question_mask_placeholder, self.context_mask_placeholder, None)

				knowledge_rep = {}
				knowledge_rep['h_p'] = h_p
				knowledge_rep['h_q'] = h_q
				self.a_s_probs, self.a_e_probs = self.decoder.decode(knowledge_rep)


		def setup_loss(self):
				"""
				Set up your loss computation here
				:return:
				"""
				# The label is a one-hot representation
				with vs.variable_scope("loss"):
					start_answer = self.labels_placeholder[:,0]
					end_answer = self.labels_placeholder[:,1]
					l1 = sparse_softmax_cross_entropy_with_logits(self.a_s_probs, start_answer)
					l2 = sparse_softmax_cross_entropy_with_logits(self.a_e_probs, end_answer)
					self.loss = l1 + l2 

					# Set up training_op (using self.loss)
					self.train_op = get_optimizer(self.optimizer)(learning_rate=self.learning_rate).minimize(self.loss)

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
		def optimize(self, session, train_x, train_y):
				"""
				Takes in actual data to optimize your model
				This method is equivalent to a step() function
				:return:
				"""
				feed_dict = {}

				# fill in this feed_dictionary like:
				feed_dict['train_x'] = train_x
				if train_y is not None: 
					input_feed['train_y'] = train_y

				output_feed = self.decoder.decode(self.knowledge_rep) 

				outputs = session.run(output_feed, feed_dict)

				return outputs


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
				init = tf.global_variables_initializer() 
				session.run(init)

				for epoch in range(self.epochs):
					logging.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
					self.run_epoch(session, dataset['train'], dataset['val']) 

					#feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch) 

					tic = time.time()
					params = tf.trainable_variables()
					num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
					toc = time.time()
					logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

	
		# A single training example is a triplet: (question, context, answer). Each entry 
		# of the triplet is a list of word IDs.
		def run_epoch(self, session, train_examples, val_examples):
				for i, batch in enumerate(minibatches(train_examples, self.batch_size)):
						pass


		def minibatches(data, batch_size, shuffle=True):
				print(len(data))




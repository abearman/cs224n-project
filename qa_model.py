from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import random
import math

import numpy as np
from six.moves import xrange		# pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits	
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import datetime

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

def plot_losses(losses):
	plt.plot(losses)
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	plt.savefig('loss/Losses_' + st)


class GRUAttnCell(rnn_cell.GRUCell):
				def __init__(self, num_units, encoder_output, scope=None):
						self.hs = encoder_output		# Source hidden state
						super(GRUAttnCell, self).__init__(num_units)		

				def __call__(self, inputs, state, scope=None):
						gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
						with vs.variable_scope(scope or type(self).__name__):
								with vs.variable_scope("Attn"):
										ht = rnn_cell._linear(gru_out, self._num_units, True, 1.0)
										ht = tf.expand_dims(ht, axis=1)
								scores = tf.reduce_sum(self.hs * ht, reduction_indices=2, keep_dims=True)

								# New stuff
								scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
								scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))

								context = tf.reduce_sum(self.hs * scores, reduction_indices=1)
								with vs.variable_scope("AttnConcat"):
										out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))

						return out, out 


def matrix_multiply_with_batch(matrix=None, batch=None, matrixByBatch=True):
	ret = None
	if matrixByBatch:
		ret = tf.scan(lambda a, x: tf.matmul(matrix, x), batch)
	else:
		n = batch.get_shape().as_list()[1]
		m = batch.get_shape().as_list()[2]
		c = matrix.get_shape().as_list()[1]
		batch = tf.reshape(batch, [-1, m]) 
		ret = tf.matmul(batch, matrix)
		ret = tf.reshape(ret, [-1, n, c])

	return ret


class Encoder(object):
	def __init__(self, state_size, embedding_size, output_size):		# vocab_dim == embed_size
		self.state_size = state_size
		self.embedding_size = embedding_size 
		self.output_size= output_size
		self.h_q = None
		self.h_p = None
		self.H_q = None
		self.H_p = None


	def encode_v2(self, question_embeddings, document_embeddings, question_mask, context_mask,
																		encoderb_state_input, dropout_keep_prob, batch_size):
		""" encode_v2() 
		"""
		with vs.variable_scope("encoder"):
			# Question -> LSTM -> Q
			lstm_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size)
			question_length = tf.reduce_sum(tf.cast(question_mask, tf.int32), reduction_indices=1)
			print("Question length: ", question_length)
			Q_prime, _ = dynamic_rnn(lstm_cell, tf.transpose(question_embeddings, [0, 2, 1]), 
															 sequence_length=question_length, time_major=False, dtype=tf.float32)
			Q_prime = tf.transpose(Q_prime, [0, 2, 1])
			print("Q_prime: ", Q_prime)

			# Non-linear projection layer on top of the question encoding
			W_Q = tf.get_variable("W_Q", (self.embedding_size, self.embedding_size))
			b_Q = tf.get_variable("b_Q", (self.embedding_size, 1)) 
			Q = tf.tanh(matrix_multiply_with_batch(matrix=W_Q, batch=question_embeddings, matrixByBatch=True) + b_Q) 
			print("Q: ", Q)

			# Paragraph -> LSTM -> D
			tf.get_variable_scope().reuse_variables()	
			print("Context mask: ", context_mask)
			context_length = tf.reduce_sum(tf.cast(context_mask, tf.int32), reduction_indices=1)
			D, _ = dynamic_rnn(lstm_cell, tf.transpose(document_embeddings, [0, 2, 1]),
												 sequence_length=context_length, time_major=False, dtype=tf.float32)
			D = tf.transpose(D, [0, 2, 1])
			print("D: ", D)

			L = tf.matmul(tf.transpose(D, [0, 2, 1]), Q)
			A_Q = tf.nn.softmax(L)
			A_D = tf.nn.softmax(tf.transpose(L, [0, 2, 1]))
			print("A_Q: ", A_Q)
			print("A_D: ", A_D)

			C_Q = batch_matmul(D, A_Q)
			print("C_Q: ", C_Q)
			concat = tf.concat(1, [Q, C_Q])
			print("concat: ", concat)
			C_D = batch_matmul(tf.concat(1, [Q, C_Q]), A_D)
			print("C_D: ", C_D)

			final_D = tf.concat(1, [D, C_D])
			print("final D: ", final_D)
			return final_D
	

	def encode(self, question_embeddings, context_embeddings, question_mask, context_mask, 
											encoder_state_input, dropout_keep_prob, batch_size):
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
						with vs.variable_scope("question", True):					
								lstm_cell = tf.nn.rnn_cell.LSTMCell(self.state_size)		# Should be 1 at first, then 200
								lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
								question_length = tf.reduce_sum(tf.cast(question_mask, tf.int32), reduction_indices=1)		
								print("Question length: ", question_length)
								#(fw_out, bw_out), _ = bidirectional_dynamic_rnn(lstm_cell, lstm_cell, 
								#					question_embeddings, sequence_length=question_length,
								#					time_major=False, dtype=tf.float64, swap_memory=True)  #TODO: time_major=True was causing seg faults
								#self.H_q = tf.concat(2, [fw_out, bw_out])
								self.H_q, _ = dynamic_rnn(lstm_cell, question_embeddings, sequence_length=question_length,
																																		time_major=False, dtype=tf.float64, swap_memory=True)

								#last_h_q_indices = question_length - 1
								#last_h_q_indices = tf.stack([tf.range(batch_size), last_h_q_indices], axis=1)
								#self.h_q = tf.gather_nd(self.H_q, last_h_q_indices) 
								self.h_q = self.H_q[:,1,:]
								print("H_q: ", self.H_q)
								print("h_q: ", self.h_q)

						with vs.variable_scope("context", True):
								# Encode context paragraph
								context_length = tf.reduce_sum(tf.cast(context_mask, tf.int32), reduction_indices=1)
								print("Context length: ", context_length)
								#attn_cell = GRUAttnCell(2* self.state_size, self.H_q)		# TODO: 2* because fw_out and bw_out are concatenated 
								#self.H_p, _ = dynamic_rnn(attn_cell, context_embeddings, dtype=tf.float64)#, sequence_length=context_length, dtype=tf.float64)
								context_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
								context_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell, output_keep_prob=dropout_keep_prob)
								#(fw_out, bw_out), _ = bidirectional_dynamic_rnn(context_lstm_cell, context_lstm_cell,
								#					context_embeddings, sequence_length=context_length,
								#					time_major=False, dtype=tf.float64, swap_memory=True)  #TODO: time_major=True was causing seg faults
								#self.H_p = tf.concat(2, [fw_out, bw_out])
								self.H_p, _ = dynamic_rnn(context_lstm_cell, context_embeddings, sequence_length=context_length,
																																		time_major=False, dtype=tf.float64, swap_memory=True)

								#self.last_h_p_indices = context_length - 1
								#self.last_h_p_indices = tf.stack([tf.range(batch_size), self.last_h_p_indices], axis=1)
								#self.h_p = tf.gather_nd(self.H_p, self.last_h_p_indices) 
								self.h_p = self.H_p[:,1,:]
								print("H_p: ", self.H_p)
								print("h_p: ", self.h_p)

						return self.h_q, self.h_p

class Decoder(object):
				def __init__(self, output_size, state_size):
								self.state_size = state_size
								self.output_size = output_size

				def decode_v2(self, final_D, W, W_prime, context_mask, embed_size): 
					with vs.variable_scope("answer_start"):
						a_s = tf.squeeze(matrix_multiply_with_batch(matrix=W, batch=tf.transpose(final_D, [0, 2, 1]), matrixByBatch=False))		# a_s = final_D * W
						print("a_s: ", a_s)

					with vs.variable_scope("answer_end"):
						lstm_cell = tf.nn.rnn_cell.LSTMCell(self.output_size)
						context_length = tf.reduce_sum(tf.cast(context_mask, tf.int32), reduction_indices=1)
						print("Context length: ", context_length)
						final_D_prime, _ = dynamic_rnn(lstm_cell, final_D,
																					sequence_length=context_length, time_major=False, dtype=tf.float32)
						print("final D prime: ", final_D_prime)
						a_e = tf.squeeze(matrix_multiply_with_batch(matrix=W_prime, batch=tf.transpose(final_D_prime, [0, 2, 1]), matrixByBatch=False))
						print("a_e: ", a_e)

					return (a_s, a_e)


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
								self.max_question_len = 60				# Longest question sequence to parse (in train or val set)
								self.max_context_len = 301				# Longest context sequence to parse (in train or val set): (766, truncated at 750)
								self.max_answer_len = 46						# Longest answer sequence to parse (in train or val set)
								self.n_classes = 2												# O or ANSWER

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
								self.gradients = None
								self.clipped_gradients = None

								# kwargs passed in
								self.state_size = kwargs['state_size']
								self.embed_path = kwargs['embed_path']
								self.embedding_size = kwargs['embedding_size']
								self.output_size = kwargs['output_size']
								self.optimizer = kwargs['optimizer']		# 'adam' or 'sgd'

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
								self.batch_size_placeholder = tf.placeholder(tf.int32, ())

								# ==== assemble pieces ====
								with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
												self.setup_embeddings()
												self.setup_system()			# Equivalent to: self.add_prediction_op() 
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
					W = tf.get_variable("W", (3*self.embedding_size, 1)) 
					W_prime = tf.get_variable("W_prime", (3*self.embedding_size, 1)) 
					final_D = self.encoder.encode_v2(self.question_embeddings, self.context_embeddings, 
																					self.question_mask_placeholder, self.context_mask_placeholder, 
																					None, self.dropout_keep_prob, self.batch_size_placeholder)
					
					self.a_s_probs, self.a_e_probs = self.decoder.decode_v2(final_D, W, W_prime, self.context_mask_placeholder, self.embedding_size)


				def setup_loss(self):
								"""
								Set up your loss computation here
								:return:
								"""
								# The label is a one-hot representation
								with vs.variable_scope("loss"):
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
								lr = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, 1000, 0.96)
								opt = get_optimizer(self.optimizer)(learning_rate=lr)

								# Get the gradients using optimizer.compute_gradients
								self.gradients, params = zip(*opt.compute_gradients(self.loss))
								for param in params:
										print("Param: ", param)

								# Clip the gradients to self.max_gradient_norm
								self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)
		
								# Re-zip the gradients and params
								grads_and_params = zip(self.clipped_gradients, params)

								# Compute the resultant global norm of the gradients and set self.grad_norm
								self.grad_norm = tf.global_norm(self.clipped_gradients)

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
										self.question_embeddings = tf.constant(pretrained_embeddings, name="question_embeddings", dtype=tf.float32)
										self.question_embeddings = tf.nn.embedding_lookup(self.question_embeddings, self.question_input_placeholder)
										self.question_embeddings = tf.reshape(self.question_embeddings, [-1, self.embedding_size, self.max_question_len])
										print("Question embeddings: ", self.question_embeddings)

										self.context_embeddings = tf.constant(pretrained_embeddings, name="context_embeddings", dtype=tf.float32)
										self.context_embeddings = tf.nn.embedding_lookup(self.context_embeddings, self.context_input_placeholder)
										self.context_embeddings = tf.reshape(self.context_embeddings, [-1, self.embedding_size, self.max_context_len])
										print("Context embeddings: ", self.context_embeddings)


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

								self.batch_size = len(question_batch)
								feed[self.batch_size_placeholder] = self.batch_size 

								_, loss, grad_norm = session.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
								#_, loss = session.run([self.train_op, self.loss], feed_dict=feed)

								#for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
								#	print(param.name)
								#	print(np.unique(param.eval()))
								#	print("")

								#for grad in self.gradients:
								#	print(grad.name)
								#	print(np.unique(grad.eval(feed_dict=feed, session=session)))
								#	print("")
										
								#for clipped_grad in self.clipped_gradients:
								#	print(clipped_grad.name)
								#	print(np.unique(clipped_grad.eval(feed_dict=feed, session=session)))
								#	print("")

								#print("h_q: ", self.encoder.h_q.eval(feed_dict=feed, session=session))
								#print("h_p: ", self.encoder.h_p.eval(feed_dict=feed, session=session))
								#print("H_q: ", np.unique(self.encoder.H_q.eval(feed_dict=feed, session=session)))
								#print("H_p: ", self.encoder.H_p.eval(feed_dict=feed, session=session))

								#print("a_s probs: ", self.a_s_probs.eval(feed_dict=feed, session=session)) 
								#print("a_e probs: ", self.a_e_probs.eval(feed_dict=feed, session=session)) 

								#print("H_q: ", self.encoder.H_q.eval(feed_dict=feed, session=session))
								#print("H_p: ", self.encoder.H_p.eval(feed_dict=feed, session=session))
								#print("last_h_p_indices: ", self.encoder.last_h_p_indices.eval(feed_dict=feed, session=session))
								#print("h_q unique: ", np.unique(self.h_q.eval(feed_dict=feed, session=session)))
								#print("h_p unique: ", np.unique(self.h_p.eval(feed_dict=feed, session=session)))
								return loss, grad_norm		# TODO
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
								question_batch, context_batch, question_mask_batch, context_mask_batch, start_answer_batch, end_answer_batch = zip(*test_x)

								input_feed = {}
								input_feed[self.question_input_placeholder] = question_batch
								input_feed[self.context_input_placeholder] = context_batch
								input_feed[self.question_mask_placeholder] = question_mask_batch
								input_feed[self.context_mask_placeholder] = context_mask_batch
								input_feed[self.start_answer_placeholder] = start_answer_batch
								input_feed[self.end_answer_placeholder] = end_answer_batch
								# No dropout on test 
								input_feed[self.dropout_placeholder] = 1.0
								#input_feed[self.dropout_placeholder] = self.dropout_keep_prob

								self.batch_size = len(question_batch)
								input_feed[self.batch_size_placeholder] = self.batch_size 

								_, loss, grad_norm = session.run([self.train_op, self.loss, self.grad_norm], feed_dict=input_feed)
								a_s = self.a_s_probs.eval(feed_dict=input_feed, session=session)
								a_e = self.a_e_probs.eval(feed_dict=input_feed, session=session)
								return a_s, a_e # 2 arrays with batch size rows and output size columns


				def answer(self, session, test_x):

								yp, yp2 = self.decode(session, test_x)

								a_s = np.argmax(yp, axis=1)
								a_e = np.argmax(yp2, axis=1)

								return a_s, a_e 

				def validate(self, sess, valid_dataset):
								"""
								Iterate through the validation dataset and determine what
								the validation cost is.

								This method calls self.test() which explicitly calculates validation cost.

								How you implement this function is dependent on how you design
								your data iteration function

								:return:
								"""
								new_saver = tf.train.import_meta_graph(model_path + model_name)
                new_saver.restore(session, tf.train.latest_checkpoint(model_path))

                f1s = []
                ems = []
                step = 1000
                for start_idx in range(0, len(dataset['val']), step):
                  end_idx = min(start_idx + step, len(dataset['val']))
                  f1s_one_batch, ems_one_batch = self.evaluate_answer(session, dataset['val'][start_idx:end_idx], sample=None, log=True)
                  f1s += f1s_one_batch
                  ems += ems_one_batch
                f1_total = sum(f1s) / float(len(f1s))
                em_total = sum(ems) / float(len(ems))
                print("Total f1: ", f1_total)
                print("Total em: ", em_total)


				def evaluate_answer(self, session, dataset, sample=100, log=False, shuffle=True):
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
								indices = range(sample)
								if shuffle:
									indices = [random.randint(0, len(dataset)) for _ in range(sample)]
								batch = [dataset[idx] for idx in indices]
								question_batch, context_batch, question_mask_batch, context_mask_batch, start_answer_batch, end_answer_batch = zip(*batch)

								a_s, a_e = self.answer(session, batch)		# These are both arrays of length sample size
								true_a_s = np.argmax(start_answer_batch, axis=1)
								true_a_e = np.argmax(end_answer_batch, axis=1)
								print("predicted a_s: ", a_s) 
								print("predicted a_e: ", a_e) 
								print("true start answer: ", true_a_s)
								print("true end answer: ", true_a_e)
								answers = [context_batch[i][a_s[i]: a_e[i]+1] for i in range(len(a_s))]
								true_answers = [context_batch[i][true_a_s[i]: true_a_e[i]+1] for i in range(len(true_a_s))]

								f1s = []
								ems = []
								for i in range(len(true_answers)):
										answer = answers[i]
										true_answer = true_answers[i]
										f1_one_example = f1_score(answer, true_answer)
										f1s.append(f1_one_example)
										em_one_example = exact_match_score(answer, true_answer)
										ems.append(em_one_example)

								f1 = np.sum(f1s) / float(sample)
								em = np.sum(ems) / float(sample)

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
								self.saver.save(session, self.train_dir + "/baselinev2_model_0") 

								training_losses = []
								for epoch in range(100): #range(self.epochs):		
										logging.info("Epoch %d out of %d", epoch + 1, self.epochs)
										self.run_epoch(session, dataset['train'], epoch, training_losses)

										# Save model
										self.saver.save(session, self.train_dir + "/baselinev2_model_epoch_" + str(epoch))

										tic = time.time()
										params = tf.trainable_variables()
										num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
										toc = time.time()
										logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
										plot_losses(training_losses)

								plot_losses(training_losses)

								# Run on the validation set when done training
								self.evaluate_answer(session, dataset['val'], sample=None, log=True)

				
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
										if answer[0] < self.output_size:
												start_answer[answer[0]] = 1
										end_answer = [0] * self.max_context_len
										if answer[1] < self.output_size:
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

						elif len(sequence) < max_length:		# Append 0's
								delta = max_length - len(sequence)
								new_sequence = sequence + ([0]*delta)
								mask = ([True] * len(sequence)) + ([False] * delta)
						#print("old sequence: ", sequence)
						#print("new sequence: ", new_sequence)
						#print("mask: ", mask)
						#print("len: ", len(sequence), len(new_sequence), len(mask))

						return (new_sequence, mask)

		
				# A single training example is a triplet: (question, context, answer). Each entry 
				# of the triplet is a list of word IDs.
				# Each batch only has training examples (no val examples).
				def run_epoch(self, session, train_examples, epoch_no, training_losses):
								for i, batch in enumerate(self.get_tiny_batches(train_examples, sample=20)):
								#for i, batch in enumerate(self.minibatches(train_examples, self.batch_size, shuffle=True)):
												print("Global step: ", self.global_step.eval())
												loss, grad_norm = self.optimize(session, batch, self.dropout_keep_prob) #TODO	
												loss_for_batch = sum(loss) / float(len(loss))
												print("Loss: ", loss_for_batch, " , grad norm: ", grad_norm)
												training_losses.append(loss_for_batch)

												if (i % 100) == 0:
														self.evaluate_answer(session, train_examples, sample=20, log=True, shuffle=False)
				
												if (i % 1000) == 0:
														# Save model
														self.saver.save(session, self.train_dir + "/baselinev2_model_epoch_" + str(epoch_no) + "_iter_" + str(i))


				def get_tiny_batches(self, data, sample=20):
					ret = []
					step = self.batch_size
					for start_idx in range(0, sample, step):
						ret.append(data[start_idx: start_idx + step]) 
					return ret
						

				# Partitioning code from: 
				# http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
				def minibatches(self, data, batch_size, shuffle=True):
								if shuffle:
										random.shuffle(data)
								num_batches = int(math.ceil(len(data) / batch_size))
								q, r = divmod(len(data), num_batches)
								indices = [q*i + min(i, r) for i in xrange(num_batches+1)]
								return [data[indices[i]:indices[i+1]] for i in xrange(num_batches)]




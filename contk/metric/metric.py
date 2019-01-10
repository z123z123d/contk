r"""
``contk.metrics`` provides classes and functions evaluating results of models. It provides
a fair metric for every model.
"""
import random

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class MetricBase:
	'''Base class for metrics.
	'''
	def __init__(self):
		pass

class PerlplexityMetric(MetricBase):
	'''Metric for calcualting perplexity.

	Arguments:
		data_key (str): Reference sentences are passed to :func:`forward` by ``data[data_key]``.
			Default: ``resp``.
		data_len_key (str): Length of reference sentences are passed to :func:`forward`
			by ``data[data_len_key]``. Default: ``resp_length``.
		gen_prob_key (str): Sentence generations model outputs of **log softmax** probability
			are passed to :func:`forward` by ``data[gen_prob_key]``. Default: ``gen_prob``.
	'''
	def __init__(self, dataloader, data_key="resp", \
					   data_len_key="resp_length", \
					   gen_prob_key="gen_prob", \
			 full_check=False
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.data_key = data_key
		self.data_len_key = data_len_key
		self.gen_prob_key = gen_prob_key
		self.word_loss = 0
		self.length_sum = 0
		self.full_check = full_check

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[data_key] (list or :class:`numpy.array`): Reference sentences.
				Contains ``<go>`` and ``<eos>``. Size: `[batch_size, max_sentence_length]`
			data[data_len_key] (list): Length of Reference sentences. Contains ``<go>`` and ``<eos>``.
				Size: `[batch_size]`
			data[gen_prob_key] (list or :class:`numpy.array`): Setence generations model outputs of
				**log softmax** probability. Contains ``<eos>``, but without ``<go>``.
				Size: `[batch_size, gen_sentence_length, vocab_size]`.

		Warning:
			``data[gen_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		resp = data[self.data_key]
		resp_length = data[self.data_len_key]
		gen_prob = data[self.gen_prob_key]
		if resp.shape[0] != resp_length.shape[0]:
			raise ValueError("Batch num is not matched.")

		# perform random check to assert the probability is valid
		checkid = random.randint(0, len(resp_length)-1)
		checkrow = random.randint(0, resp_length[checkid]-2)
		if not np.isclose(np.sum(np.exp(gen_prob[checkid][checkrow])), 1):
			print("gen_prob[%d][%d] exp sum is equal to %f." % (checkid, checkrow, \
				np.sum(np.exp(gen_prob[checkid][checkrow]))))
			raise ValueError("data[gen_prob_key] must be processed after log_softmax.")

		for i, single_length in enumerate(resp_length):
			# perform full check to assert the probability is valid
			if self.full_check:
				expsum = np.sum(np.exp(gen_prob[i][:single_length]), -1)
				if not np.allclose(expsum, [1] * single_length):
					raise ValueError("data[gen_prob_key] must be processed after log_softmax.")

			self.word_loss += -np.sum(gen_prob[i][\
				list(range(single_length-1)), resp[i][1:single_length]])
			self.length_sum += single_length - 1

	def forward_pad(self, data):
		'''Processing a batch of data with padding in resp_length (resp_length[i] == 0).

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[data_key] (list or :class:`numpy.array`): Reference sentences.
				Contains ``<go>`` and ``<eos>``. Size: `[batch_size, max_sentence_length]`
			data[data_len_key] (list): Length of Reference sentences. Contains ``<go>`` and ``<eos>``.
				Size: `[batch_size]`
			data[gen_prob_key] (list or :class:`numpy.array`): Setence generations model outputs of
				**log softmax** probability. Contains ``<eos>``, but without ``<go>``.
				Size: `[batch_size, gen_sentence_length, vocab_size]`.

		Warning:
			``data[gen_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		resp = data[self.data_key]
		resp_length = data[self.data_len_key]
		gen_prob = data[self.gen_prob_key]
		if resp.shape[0] != resp_length.shape[0]:
			raise ValueError("Batch num is not matched.")
		valid_index = [idx for idx, single_length in enumerate(resp_length) \
			if single_length > 1]
		# perform random check to assert the probability is valid
		checkid = random.choice(valid_index)
		checkrow = random.randint(0, resp_length[checkid]-2)
		if not np.isclose(np.sum(np.exp(gen_prob[checkid][checkrow])), 1):
			print("gen_prob[%d][%d] exp sum is equal to %f." % (checkid, checkrow, \
				np.sum(np.exp(gen_prob[checkid][checkrow]))))
			raise ValueError("data[gen_prob_key] must be processed after log_softmax.")

		for i in valid_index:
			single_length = resp_length[i]
			# perform full check to assert the probability is valid
			if self.full_check:
				expsum = np.sum(np.exp(gen_prob[i][:single_length]), -1)
				if not np.allclose(expsum, [1] * single_length):
					raise ValueError("data[gen_prob_key] must be processed after log_softmax.")

			self.word_loss += -np.sum(gen_prob[i][\
				list(range(single_length-1)), resp[i][1:single_length]])
			self.length_sum += single_length - 1

	def close(self):
		'''Return a dict which contains:

			* **perplexity**: perplexity value
		'''
		return {"perplexity": np.exp(self.word_loss / self.length_sum)}

class BleuCorpusMetric(MetricBase):
	'''Metric for calcualting BLEU.

	Arguments:
		data_key (str): Reference sentences are passed to :func:.forward by ``data[data_key]``.
			Default: ``resp``.
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_prob_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, data_key="resp", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.data_key = data_key
		self.gen_key = gen_key
		self.bleu_value = 0
		self.sentence_num = 0

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[data_key] (list or :class:`numpy.array` of `int`): Reference sentences.
				Contains ``<go>`` and ``<eos>``. Size: `[batch_size, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Setences generated by model.
				Contains ``<eos>``, but without ``<go>``.
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]
		resp = data[self.data_key]
		if resp.shape[0] != gen.shape[0]:
			raise ValueError("Batch num is not matched.")

		for gen_sen, resp_sen in zip(gen, resp):
			gen_sen_processed = self.dataloader.trim_index(gen_sen)
			resp_sen_processed = self.dataloader.trim_index(resp_sen)
			self.bleu_value += sentence_bleu([resp_sen_processed[1:]], gen_sen_processed, \
				smoothing_function=SmoothingFunction().method7)
			self.sentence_num += 1

	def close(self):
		'''Return a dict which contains:

			* **bleu**: bleu value.
		'''
		return {"bleu": self.bleu_value / self.sentence_num}

class SingleDialogRecorder(MetricBase):
	'''A metric-like class for recorder BLEU.

	Arguments:
		dataloader (DataLoader): A dataloader for translating index to sentences.
		post_key (str): Dialog post are passed to :func:`forward` by ``data[data_key]``.
			Default: ``post``.
		resp_key (str): Dialog responses are passed to :func:`forward` by ``data[data_key]``.
			Default: ``resp``.
		gen_key (str): Sentence generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, post_key="post", resp_key="resp", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.post_key = post_key
		self.resp_key = resp_key
		self.gen_key = gen_key
		self.post_list = []
		self.resp_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[post_key] (list or :class:`numpy.array` of `int`): Dialog post.
				Contains ``<go>`` and ``<eos>``. Size: `[batch_size, max_sentence_length]`
			data[resp_key] (list or :class:`numpy.array` of `int`): Dialog responses.
				Contains ``<go>`` and ``<eos>``. Size: `[batch_size, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Setences generated by model.
				Contains ``<eos>``, but without ``<go>``.
				Size: `[batch_size, gen_sentence_length]`.
		'''
		post = data[self.post_key]
		resp = data[self.resp_key]
		gen = data[self.gen_key]
		if post.shape[0] != resp.shape[0] or resp.shape[0] != gen.shape[0]:
			raise ValueError("Batch num is not matched.")
		for i in range(post.shape[0]):
			self.post_list.append(self.dataloader.index_to_sen(post[i, 1:]))
			self.resp_list.append(self.dataloader.index_to_sen(resp[i, 1:]))
			self.gen_list.append(self.dataloader.index_to_sen(gen[i]))

	def close(self):
		'''Return a dict which contains:

			* **post**: a list of post sentences.
			* **resp**: a list of response sentences.
			* **gen**: a list of generated sentences.
		'''
		return {"post": self.post_list, "resp": self.resp_list, "gen": self.gen_list}

class LanguageGenerationRecorder(MetricBase):
	'''A metric-like class for recorder BLEU.

	Arguments:
		dataloader (DataLoader): A dataloader for translating index to sentences.
		gen_key (str): Sentences generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.gen_key = gen_key
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[gen_key] (list or :class:`numpy.array` of `int`): Setences generated by model.
				Contains ``<eos>``, but without ``<go>``.
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]
		for i in range(gen.shape[0]):
			self.gen_list.append(self.dataloader.index_to_sen(gen[i]))

	def close(self):
		'''Return a dict which contains:

			* **gen**: a list of generated sentences.
		'''
		return {"gen": self.gen_list}

class MetricChain(MetricBase):
	'''A metric-like class for stacked metric. You can use this class
	making multiples metric combination like one.

	Examples:
		>>> metric = MetricChain()
		>>> metric.add_metric(BleuCorpusMetric())
		>>> metric.add_metric(SingleDialogRecorder(dataloader))
	'''
	def __init__(self):
		super().__init__()
		self.metric_list = []

	def add_metric(self, metric):
		'''Add metric for processing.

		Arguments:
			metric (MetricBase): a metric class
		'''
		if not isinstance(metric, MetricBase):
			raise TypeError("Metric must be a subclass of MetricBase")
		self.metric_list.append(metric)

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains keys which all the
				metric components need.
		'''
		for metric in self.metric_list:
			metric.forward(data)

	def close(self):
		'''Return a dict which contains the items which all the
			meric components returned.
		'''
		ret_dict = {}
		for metric in self.metric_list:
			ret_dict.update(metric.close())
		return ret_dict

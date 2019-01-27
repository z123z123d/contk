r"""
``contk.metrics`` provides classes and functions evaluating results of models. It provides
a fair metric for every model.
"""
import random

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class MetricBase:
	'''Base class for metrics.
	'''
	def __init__(self):
		pass

class PerlplexityMetric(MetricBase):
	'''Metric for calcualting perplexity.

	Arguments:
		reference_allvocabs_key (str): Reference sentences with all vocabs
			are passed to :func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``resp_allvocabs``.
		reference_len_key (str): Length of reference sentences are passed to :func:`forward`
			by ``data[reference_len_key]``. Default: ``resp_length``.
		gen_log_prob_key (str): Sentence generations model outputs of **log softmax** probability
			are passed to :func:`forward` by ``data[gen_log_prob_key]``. Default: ``gen_log_prob``.
		invalid_vocab (bool): whether gen_log_prob contains invalid vocab. Default: False
		full_check (bool): whether perform full checks on `gen_log_prob` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			Default: False
	'''
	def __init__(self, dataloader, \
					   reference_allvocabs_key="resp_allvocabs", \
					   reference_len_key="resp_length", \
					   gen_log_prob_key="gen_log_prob", \
					   invalid_vocab=False, \
					   full_check=False \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_len_key = reference_len_key
		self.gen_log_prob_key = gen_log_prob_key
		self.word_loss = 0
		self.length_sum = 0
		self.invalid_vocab = invalid_vocab
		self.full_check = full_check

	def forward(self, data):
		'''Processing a batch of data. Smoothing will be performed for invalid vocabs.
		Unknowns vocabs will be ignored.

		TODO:
			Find a place to explain valid vocabs, invalid vocabs, and unknown vocabs.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array`): Reference sentences with all vocabs
				with all vocabs. Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[reference_len_key] (list): Length of Reference sentences. Contains start token (eg:``<go>``)
				and end token (eg:``<eos>``). Size: `[batch_size]`
			data[gen_log_prob_key] (list or :class:`numpy.array`): Setence generations model outputs of
				**log softmax** probability. Contains end token (eg:``<eos>``), but without start token
				(eg: ``<go>``).	The 2nd dimension can be jagged.
				Size: `[batch_size, gen_sentence_length, vocab_size]` for ``invalid_vocab = False``.
				`[batch_size, gen_sentence_length, all_vocab_size]` for ``invalid_vocab = True``.

		Warning:
			``data[gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_log_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		resp_allwords = data[self.reference_allvocabs_key]
		resp_length = data[self.reference_len_key]
		gen_log_prob = data[self.gen_log_prob_key]
		if len(resp_allwords) != len(resp_length) or len(resp_allwords) != len(gen_log_prob):
			raise ValueError("Batch num is not matched.")

		# perform random check to assert the probability is valid
		checkid = random.randint(0, len(resp_length)-1)
		if resp_length[checkid]-2 > 0:
			raise ValueError("resp_length must no less than 2, because <go> and <eos> are always included.")
		checkrow = random.randint(0, resp_length[checkid]-2)
		if not np.isclose(np.sum(np.exp(gen_log_prob[checkid][checkrow])), 1):
			print("gen_log_prob[%d][%d] exp sum is equal to %f." % (checkid, checkrow, \
				np.sum(np.exp(gen_log_prob[checkid][checkrow]))))
			raise ValueError("data[gen_log_prob_key] must be processed after log_softmax.")

		if not isinstance(resp_allwords, np.ndarray):
			resp_allwords = np.array(resp_allwords)
		if not isinstance(gen_log_prob, np.ndarray):
			gen_log_prob = np.array(gen_log_prob)

		invalid_vocab_num = self.dataloader.all_vocab_size - self.dataloader.vocab_size
		#resp = resp_allwords.copy()
		#resp[resp >= self.dataloader.vocab_size] = self.dataloader.unk_id

		for i, single_length in enumerate(resp_length):
			# perform full check to assert the probability is valid
			if self.full_check:
				expsum = np.sum(np.exp(gen_log_prob[i][:single_length]), -1)
				if not np.allclose(expsum, [1] * single_length):
					raise ValueError("data[gen_log_prob_key] must be processed after log_softmax.")

			resp_now = resp_allwords[i][1:single_length]

			if self.invalid_vocab:
				if resp_now.shape[1] != self.dataloader.vocab_size:
					raise ValueError("The third dimension gen_log_prob_key should be equals to vocab_size when \
						invalid_vocab = False, \
						but %d != %d" % (resp_now.shape[1], self.dataloader.vocab_size))
			else:
				if resp_now.shape[1] != self.dataloader.all_vocab_size:
					raise ValueError("The third dimension gen_log_prob_key should be equals to all_vocab_size \
						when invalid_vocab = True, \
						but %d != %d" % (resp_now.shape[1], self.dataloader.vocab_size))

			# calc normal vocab
			normal_idx = np.where(resp_now != self.dataloader.unk_id and \
									resp_now < self.dataloader.vocab_size)
			self.word_loss += -np.sum(gen_log_prob[i][normal_idx, resp_now[normal_idx]])
			self.length_sum += len(normal_idx)

			# calc invalid vocab
			invalid_idx = np.where(resp_now >= self.dataloader.vocab_size)
			invalid_log_prob = gen_log_prob[i][\
									invalid_idx, [self.dataloader.unk_id] * len(invalid_idx) \
								] - np.log(invalid_vocab_num)
			if self.invalid_vocab:
				extra_invalid_log_prob = gen_log_prob[i][invalid_idx, resp_now[invalid_idx]]
				self.word_loss += np.sum(np.log( \
						np.exp(invalid_log_prob) + np.exp(extra_invalid_log_prob) \
					))
			else:
				self.word_loss += np.sum(invalid_log_prob)
			self.length_sum += len(invalid_idx)

	def close(self):
		'''Return a dict which contains:

			* **perplexity**: perplexity value
		'''
		return {"perplexity": np.exp(self.word_loss / self.length_sum)}

class MultiTurnPerplexityMetric(MetricBase):
	'''Metric for calcualting multi-turn perplexity.

	Arguments:
		reference_allvocabs_key (str): Reference sentences with all vocabs
			are passed to :func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``sent_allvocabs``.
		reference_len_key (str): Length of reference sentences are passed to :func:`forward`
			by ``data[reference_len_key]``. Default: ``sent_length``.
		gen_log_prob_key (str): Sentence generations model outputs of **log softmax** probability
			are passed to :func:`forward` by ``data[gen_log_prob_key]``. Default: ``gen_log_prob``.
		invalid_vocab (bool): whether gen_log_prob contains invalid vocab. Default: False
		full_check (bool): whether perform full checks on `gen_log_prob` to make sure the sum
			of probability is 1. Otherwise, a random check will be performed for efficiency.
			Default: False
	'''
	def __init__(self, dataloader, reference_allvocabs_key="sent_allvocabs", \
					   reference_len_key="sent_length", \
					   gen_log_prob_key="gen_log_prob", \
					   invalid_vocab=False, \
					   full_check=False \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_len_key = reference_len_key
		self.gen_log_prob_key = gen_log_prob_key
		self.invalid_vocab = invalid_vocab
		self.sub_metric = PerlplexityMetric(dataloader, \
				reference_allvocabs_key="sent_allvocabs", \
				reference_len_key="sent_length", \
				gen_log_prob_key="gen_log_prob", \
				invalid_vocab=invalid_vocab, \
				full_check=full_check)

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array`): Reference sentences
				with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_turn_length, max_sentence_length]`
			data[reference_len_key] (list of list): Length of Reference sentences. Contains
				start token (eg:``<go>``) and end token (eg:``<eos>``). It must NOT be padded,
				which means the inner lists may have different length.
				Length of outer list: `batch_size`
			data[gen_log_prob_key] (list or :class:`numpy.array`): Setence generations model outputs of
				**log softmax** probability. Contains end token (eg:``<eos>``), but without start token
				(eg: ``<go>``).	The 2nd / 3rd dimension can be jagged.
				Size: `[batch_size, max_turn_length, gen_sentence_length, vocab_size]`.

		Warning:
			``data[gen_log_prob_key]`` must be processed after log_softmax. That means,
			``np.sum(np.exp(gen_log_prob), -1)`` equals ``np.ones((batch_size, gen_sentence_length))``
		'''
		reference_allvocabs = data[self.reference_allvocabs_key]
		length = data[self.reference_len_key]
		gen_log_prob = data[self.gen_log_prob_key]
		if len(length) != len(reference_allvocabs) or len(length) != len(gen_log_prob):
			raise ValueError("Batch num is not matched.")

		for i, sent_length in enumerate(length):
			self.sub_metric.forward({"sent_allvocabs": reference_allvocabs[i], \
					"sent_length": sent_length, \
					"gen_log_prob": gen_log_prob[i]})

	def close(self):
		'''Return a dict which contains:

			* **perplexity**: perplexity value
		'''
		return self.sub_metric.close()

class BleuCorpusMetric(MetricBase):
	'''Metric for calcualting BLEU.

	Arguments:
		reference_allvocabs_key (str): Reference sentences with all vocabs
			are passed to :func:.forward by ``data[reference_allvocabs_key]``.
			Default: ``resp``.
		gen_key (str): Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, reference_allvocabs_key="resp_allvocabs", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array` of `int`):
				reference_allvocabs sentences.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Setences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]
		resp = data[self.reference_allvocabs_key]
		if len(resp) != len(gen):
			raise ValueError("Batch num is not matched.")

		for gen_sen, resp_sen in zip(gen, resp):
			self.hyps.append(self.dataloader.trim_index(gen_sen))
			self.refs.append([self.dataloader.trim_index(resp_sen[1:])])

	def close(self):
		'''Return a dict which contains:

			* **bleu**: bleu value.
		'''
		return {"bleu": corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method7)}

class MultiTurnBleuCorpusMetric(MetricBase):
	'''Metric for calcualting multi-turn BLEU.

	Arguments:
		reference_allvocabs_key (str): reference sentences with all vocabs are passed to
			:func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``sent``.
		reference_allvocabs_len_key (str): Length of reference sentences are passed to :func:`forward`
			by ``data[reference_allvocabs_len_key]``. Default: ``sent_length``.
		gen_key (str):Sentences generated by model are passed to :func:.forward by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, reference_allvocabs_key="sent", \
					   gen_key="gen" \
			  ):
		super().__init__()
		self.dataloader = dataloader
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[reference_allvocabs_key] (list or :class:`numpy.array`):
				Reference sentences with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_turn_length, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array`): 3-d array of int.
				Setences generated by model.
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				The 2nd / 3rd dimension can be jagged.
				Size: `[batch_size, max_turn_length, gen_sentence_length]`.
		'''
		reference_allvocabs = data[self.reference_allvocabs_key]
		gen = data[self.gen_key]
		if len(gen) != len(reference_allvocabs):
			raise ValueError("Batch num is not matched.")

		for gen_session, ref_session in zip(gen, reference_allvocabs):
			gen_processed = self.dataloader.multi_turn_trim_index(gen_session)
			ref_processed = self.dataloader.multi_turn_trim_index(ref_session)
			if len(gen_processed) != len(ref_processed):
				raise ValueError("Turn num is not matched.")
			for gen_sent, ref_sent in zip(gen_processed, ref_processed):
				self.hyps.append(self.dataloader.trim_index(gen_sent))
				self.refs.append([self.dataloader.trim_index(ref_sent)[1:]])

	def close(self):
		'''Return a dict which contains:

			* **bleu**: bleu value.
		'''
		return {"bleu": corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method7)}

class SingleTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		dataloader (DataLoader): A dataloader for translating index to sentences.
		post_allvocabs_key (str): Dialog post are passed to :func:`forward`
			by ``data[post_allvocabs_key]``.
			Default: ``post``.
		resp_allvocabs_key (str): Dialog responses are passed to :func:`forward`
			by ``data[resp_allvocabs_key]``.
			Default: ``resp``.
		gen_key (str): Sentence generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, post_allvocabs_key="post_allvocabs", \
			resp_allvocabs_key="resp_allvocabs", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.post_allvocabs_key = post_allvocabs_key
		self.resp_allvocabs_key = resp_allvocabs_key
		self.gen_key = gen_key
		self.post_list = []
		self.resp_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[post_allvocabs_key] (list or :class:`numpy.array` of `int`):
				Dialog posts with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[resp_allvocabs_key] (list or :class:`numpy.array` of `int`):
				Dialog responses with all vocabs.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Setences generated by model.
				Contains end token (eg: ``<eos>``)`, but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		post_allvocabs = data[self.post_allvocabs_key]
		resp_allvocabs = data[self.resp_allvocabs_key]
		gen = data[self.gen_key]
		if len(post_allvocabs) != len(resp_allvocabs) or len(resp_allvocabs) != len(gen):
			raise ValueError("Batch num is not matched.")
		for i, post_sen in enumerate(post_allvocabs):
			self.post_list.append(self.dataloader.index_to_sen(post_sen[1:]))
			self.resp_list.append(self.dataloader.index_to_sen(resp_allvocabs[i][1:]))
			self.gen_list.append(self.dataloader.index_to_sen(gen[i]))

	def close(self):
		'''Return a dict which contains:

			* **post**: a list of post sentences.
			* **resp**: a list of response sentences.
			* **gen**: a list of generated sentences.
		'''
		return {"post": self.post_list, "resp": self.resp_list, "gen": self.gen_list}

class MultiTurnDialogRecorder(MetricBase):
	'''A metric-like class for recording generated sentences and references.

	Arguments:
		dataloader (DataLoader): A dataloader for translating index to sentences.
		context_key (str): Dialog context are passed to :func:`forward` by ``data[context_key]``.
			Default: ``post``.
		reference_allvocabs_key (str): Dialog references with all vocabs
			are passed to :func:`forward` by ``data[reference_allvocabs_key]``.
			Default: ``resp``.
		gen_key (str): Sentences generated by model are passed to :func:`forward` by
			``data[gen_key]``. Default: ``gen``.
	'''
	def __init__(self, dataloader, context_key="content", \
			reference_allvocabs_key="reference", gen_key="gen"):
		super().__init__()
		self.dataloader = dataloader
		self.context_key = context_key
		self.reference_allvocabs_key = reference_allvocabs_key
		self.gen_key = gen_key
		self.context_list = []
		self.reference_list = []
		self.gen_list = []

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys.
			data[context_key] (list or :class:`numpy.array` of `int`): Dialog post.
				A 3-d padded array containing id of words.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, _turn_length, max_sentence_length]`
			data[reference_allvocabs_key] (list or :class:`numpy.array` of `int`):
				Dialog responses with all vocabs. A 3-d padded array containing id of words.
				Contains start token (eg: ``<go>``) and end token (eg: ``<eos>``).
				Size: `[batch_size, max_turn_length, max_sentence_length]`
			data[gen_key] (list or :class:`numpy.array` of `int`): Setences generated by model.
				A 3-d padded array containing id of words.
				Contains  end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, max_turn_length, gen_sentence_length]`.
		'''
		context = data[self.context_key]
		reference_allvocabs = data[self.reference_allvocabs_key]
		gen = data[self.gen_key]
		if len(context) != len(reference_allvocabs) or len(context) != len(gen):
			raise ValueError("Batch num is not matched.")
		if not isinstance(context, np.ndarray):
			context = np.array(context)
		if not isinstance(reference_allvocabs, np.ndarray):
			reference_allvocabs = np.array(reference_allvocabs)
		if not isinstance(gen, np.ndarray):
			gen = np.array(gen)
		for i, context_sen in enumerate(context):
			self.context_list.append(self.dataloader.multi_turn_index_to_sen(context_sen[ :, 1:]))
			self.reference_list.append(\
				self.dataloader.multi_turn_index_to_sen(reference_allvocabs[i, :, 1:]))
			self.gen_list.append(self.dataloader.multi_turn_index_to_sen(gen[i, :]))

	def close(self):
		'''Return a dict which contains:

			* **context**: a list of post sentences.
			* **reference**: a list of response sentences.
			* **gen**: a list of generated sentences.
		'''
		return {"context": self.context_list, "reference": self.reference_list, "gen": self.gen_list}

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
				Contains end token (eg: ``<eos>``), but without start token (eg: ``<go>``).
				Size: `[batch_size, gen_sentence_length]`.
		'''
		gen = data[self.gen_key]
		for sen in gen:
			self.gen_list.append(self.dataloader.index_to_sen(sen))

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

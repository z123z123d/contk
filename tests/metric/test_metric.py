import copy
import itertools
import random

import numpy as np
import pytest

from contk.metric import MetricBase, \
	BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric, \
	PerplexityMetric, MultiTurnPerplexityMetric, BleuCorpusMetric, MultiTurnBleuCorpusMetric, \
	SingleTurnDialogRecorder, MultiTurnDialogRecorder, LanguageGenerationRecorder, HashValueRecorder, \
	MetricChain
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from contk.dataloader import BasicLanguageGeneration, MultiTurnDialog

def setup_module():
	random.seed(0)
	np.random.seed(0)

def test_bleu_bug():
	ref = [[[1, 3], [3], [4]]]
	gen = [[1]]
	with pytest.raises(ZeroDivisionError):
		corpus_bleu(ref, gen, smoothing_function=SmoothingFunction().method7)


class FakeDataLoader(BasicLanguageGeneration):
	def __init__(self):
		self.all_vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', \
							   'what', 'how', 'here', 'do', 'as', 'can', 'to']
		self.pad_id = 0
		self.unk_id = 1
		self.go_id = 2
		self.eos_id = 3
		self.end_token = self.eos_id
		self.valid_vocab_len = 8
		self.word2id = {x: i for i, x in enumerate(self.all_vocab_list)}
		self.key_name = ["train", "dev", "test"]

	def get_sen(self, max_len, len, gen=False, pad=True, all_vocab=False):
		sen = []
		for i in range(len):
			if all_vocab:
				vocab = random.randrange(self.word2id['<eos>'], self.all_vocab_size)
			else:
				vocab = random.randrange(self.word2id['<eos>'], self.vocab_size)
			if vocab == self.eos_id:
				vocab = self.unk_id
			# consider unk
			if vocab == self.word2id['<eos>']:
				vocab = self.unk_id
			sen.append(vocab)
		if not gen:
			sen[0] = self.word2id['<go>']
		sen[len - 1] = self.word2id['<eos>']
		if pad:
			for i in range(max_len - len):
				sen.append(self.word2id['<pad>'])
		return sen

	def get_data(self, reference_key=None, reference_len_key=None, gen_prob_key=None, gen_key=None, \
				 post_key=None, \
				 to_list=False, \
				 pad=True, gen_prob_check='no_check', \
				 gen_len='random', ref_len='random', \
				 ref_vocab='all_vocab', gen_vocab='all_vocab', gen_prob_vocab='all_vocab', \
				 resp_len='>=2', batch=5, max_len=10):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			post_key: [] \
		}

		for i in range(batch):
			if resp_len == '<2':
				ref_nowlen = 1
			elif ref_len == "random":
				ref_nowlen = random.randrange(2, 5)
			elif ref_len == "non-empty":
				ref_nowlen = 8
			elif ref_len == 'empty':
				ref_nowlen = 2
			data[reference_key].append(self.get_sen(max_len, ref_nowlen, pad=pad, \
													all_vocab=ref_vocab=='all_vocab'))
			data[reference_len_key].append(ref_nowlen)

			data[post_key].append(self.get_sen(max_len, ref_nowlen, pad=pad))

			if gen_len == "random":
				gen_nowlen = random.randrange(1, 4) if i > 2 else 3 # for BLEU not empty
			elif gen_len == "non-empty":
				gen_nowlen = 7
			elif gen_len == "empty":
				gen_nowlen = 1
			data[gen_key].append(self.get_sen(max_len, gen_nowlen, gen=True, pad=pad, \
											  all_vocab=gen_vocab=='all_vocab'))

			gen_prob = []
			for j in range(ref_nowlen - 1):
				vocab_prob = []
				if gen_prob_vocab == 'all_vocab':
					vocab_nowsize = self.all_vocab_size
				else:
					vocab_nowsize = self.vocab_size

				for k in range(vocab_nowsize):
					vocab_prob.append(random.random())
				vocab_prob /= np.sum(vocab_prob)
				if gen_prob_check != "random_check":
					vocab_prob = np.log(vocab_prob)
				gen_prob.append(list(vocab_prob))
			data[gen_prob_key].append(gen_prob)

		if gen_prob_check == "full_check":
			data[gen_prob_key][-1][0][0] -= 1

		if not to_list:
			for key in data:
				if key is not None:
					data[key] = np.array(data[key])
		return data

class FakeMultiDataloader(MultiTurnDialog):
	def __init__(self):
		self.all_vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', \
							   'what', 'how', 'here', 'do', 'as', 'can', 'to']
		self.pad_id = 0
		self.unk_id = 1
		self.go_id = 2
		self.eos_id = 3
		self.end_token = self.eos_id
		self.valid_vocab_len = 8
		self.word2id = {x: i for i, x in enumerate(self.all_vocab_list)}
		self.key_name = ["train", "dev", "test"]

	def get_sen(self, max_len, len, gen=False, pad=True, all_vocab=False):
		return FakeDataLoader.get_sen(self, max_len, len, gen, pad, all_vocab)

	def get_data(self, reference_key=None, reference_len_key=None, turn_len_key=None, gen_prob_key=None, gen_key=None, \
				 context_key=None, \
				 to_list=False, \
				 pad=True, gen_prob_check='no_check', \
				 gen_len='random', ref_len='random', \
				 ref_vocab='all_vocab', gen_vocab='all_vocab', gen_prob_vocab='all_vocab', \
				 resp_len='>=2', batch=5, max_len=10, max_turn=5):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			turn_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			context_key: [] \
		}

		for i in range(batch):
			turn_length = random.randrange(1, max_turn+1)
			turn_reference = []
			turn_reference_len = []
			turn_gen_prob = []
			turn_gen = []
			turn_context = []

			for j in range(turn_length):
				if resp_len == '<2':
					ref_nowlen = 1
				elif ref_len == "random":
					ref_nowlen = random.randrange(2, 5)
				elif ref_len == "non-empty":
					ref_nowlen = 8
				elif ref_len == 'empty':
					ref_nowlen = 2
				turn_reference.append(self.get_sen(max_len, ref_nowlen, pad=pad, all_vocab=ref_vocab=='all_vocab'))
				turn_reference_len.append(ref_nowlen)

				turn_context.append(self.get_sen(max_len, ref_nowlen, pad=pad, all_vocab=ref_vocab=='all_vocab'))
				if gen_len == "random":
					gen_nowlen = random.randrange(1, 4) if i != 0 else 3 # for BLEU not empty
				elif gen_len == "non-empty":
					gen_nowlen = 7
				elif gen_len == "empty":
					gen_nowlen = 1
				turn_gen.append(self.get_sen(max_len, gen_nowlen, gen=True, pad=pad, all_vocab=gen_vocab=='all_vocab'))

				gen_prob = []
				for k in range(max_len - 1 if pad else ref_nowlen - 1):
					vocab_prob = []
					if gen_prob_vocab == 'all_vocab':
						vocab_nowsize = self.all_vocab_size
					else:
						vocab_nowsize = self.vocab_size

					for l in range(vocab_nowsize):
						vocab_prob.append(random.random())
					vocab_prob /= np.sum(vocab_prob)
					if gen_prob_check != "random_check":
						vocab_prob = np.log(vocab_prob)
					gen_prob.append(list(vocab_prob))
				turn_gen_prob.append(gen_prob)

			data[reference_key].append(turn_reference)
			data[reference_len_key].append(turn_reference_len)
			data[turn_len_key].append(turn_length)
			data[gen_prob_key].append(turn_gen_prob)
			data[gen_key].append(turn_gen)
			data[context_key].append(turn_context)

		if gen_prob_check == "full_check":
			data[gen_prob_key][-1][-1][0][0] -= 1

		if not to_list:
			for key in data:
				if key is not None:
					data[key] = np.array(data[key])
		return data

test_argument =  [ 'default',   'custom']

test_shape =     [     'pad',      'jag',      'pad',      'jag']
test_type =      [   'array',    'array',     'list',     'list']

test_batch_len = [   'equal',  'unequal']
test_turn_len =  [   'equal',  'unequal']

test_check =     ['no_check', 'random_check', 'full_check']

test_gen_len =   [  'random', 'non-empty',   'empty']
test_ref_len =   [  'random', 'non-empty',   'empty']

test_ref_vocab =      ['valid_vocab', 'all_vocab']
test_gen_vocab =      ['valid_vocab', 'all_vocab']
test_gen_prob_vocab = ['valid_vocab', 'all_vocab']

test_resp_len = ['>=2', '<2']
test_include_invalid = [False, True]
test_ngram = [1, 2, 3, 4, 5, 6]

test_emb_mode = ['avg', 'extrema', 'sum']
test_emb_type = ['array', 'list']
test_emb_len =  [8, 11]

test_hash_data = ['has_key', 'no_key']

## test_batch_len: len(ref) == len(gen)?
## test_turn_len: len(single_batch(ref)) == len(single_batch(gen))?
## test_gen_len: 'empty' means all length are 1 (eos), 'non-empty' means all length are > 1, 'random' means length are random
## test_ref_len: 'empty' means all length are 2 (eos), 'non-empty' means all length are > 2, 'both' means length are random

def same_data(A, B):
	if type(A) != type(B):
		return False
	try:
		if len(A) != len(B):
			return False
	except TypeError:
		return A == B
	for i, x in enumerate(A):
		if not same_data(x, B[i]):
			return False
	return True

def same_dict(A, B):
	if A.keys() != B.keys():
		return False
	for x in A.keys():
		if not same_data(A[x], B[x]):
			return False
	return True

def generate_testcase(*args):
	args = [(list(p), mode) for p, mode in args]
	default = []
	for p, _ in args:
		default.extend(p[0])
	yield tuple(default)
	# add
	i = 0
	for p, mode in args:
		if mode == "add":
			for k in p[1:]:
				yield tuple(default[:i] + list(k) + default[i+len(p[0]):])
		i += len(p[0])

	# multi
	res = []
	for i, (p, mode) in enumerate(args):
		if mode == "add":
			res.append(p[:1])
		else:
			res.append(p)
	for p in itertools.product(*res):
		yield tuple(itertools.chain(*p))


bleu_precision_recall_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_gen_len), "multi"),
	(zip(test_ngram), "add")
)


class TestBleuPrecisionRecallMetric():
	def test_base_class(self):
		with pytest.raises(NotImplementedError):
			gen = []
			reference = []
			bprm = BleuPrecisionRecallMetric(ngram=1)
			super(BleuPrecisionRecallMetric, bprm).score(gen, reference)

	@pytest.mark.parametrize('argument, shape, type, batch_len, ref_len, gen_len, ngram',
		bleu_precision_recall_test_parameter)
	def test_close(self, argument, shape, type, batch_len, ref_len, gen_len, ngram):
		dataloader = FakeMultiDataloader()

		if ngram not in range(1, 5):
			with pytest.raises(ValueError, match="ngram should belong to \[1, 4\]"):
				bprm = BleuPrecisionRecallMetric(ngram)
			return

		if argument == 'default':
			reference_key, gen_key = ('resp', 'gen')
			bprm = BleuPrecisionRecallMetric(ngram)
		else:
			reference_key, gen_key = ('rk', 'gk')
			bprm = BleuPrecisionRecallMetric(ngram, reference_key, gen_key)

		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   ref_len=ref_len, gen_len=gen_len)
		_data = copy.deepcopy(data)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match="Batch num is not matched."):
				bprm.forward(data)
		else:
			bprm.forward(data)
			ans = bprm.close()
			prefix = 'BLEU-' + str(ngram)
			assert sorted(ans.keys()) == [prefix + ' precision', prefix + ' recall']

		assert same_dict(data, _data)



emb_similarity_precision_recall_test_parameter = generate_testcase( \
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_ref_len), "add"),
	(zip(test_gen_len), "add"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_vocab), "multi"),
	(zip(test_emb_mode), "add"),
	(zip(test_emb_type), "add"),
	(zip(test_emb_len), "multi")
)


class TestEmbSimilarityPrecisionRecallMetric():
	@pytest.mark.parametrize('argument, shape, type, batch_len, ref_len, gen_len, '
							 'ref_vocab, gen_vocab, emb_mode, emb_type, emb_len',
							 emb_similarity_precision_recall_test_parameter)
	def test_close(self, argument, shape, type, batch_len, ref_len, gen_len, \
							 ref_vocab, gen_vocab, emb_mode, emb_type, emb_len):
		dataloader = FakeMultiDataloader()

		emb = []
		for i in range(emb_len):
			vec = []
			for j in range(5):
				vec.append(random.random())
			emb.append(vec)
		print(emb_len, gen_vocab)
		if emb_type == 'array':
			emb = np.array(emb)
		if emb_type != 'array':
			with pytest.raises(ValueError, match="invalid type of shape of embed."):
				espr = EmbSimilarityPrecisionRecallMetric(emb, emb_mode)
			return
		if emb_mode not in ['avg', 'extrema']:
			with pytest.raises(ValueError, match="mode should be 'avg' or 'extrema'."):
				espr = EmbSimilarityPrecisionRecallMetric(emb, emb_mode)
			return

		if argument == 'default':
			reference_key, gen_key = ('resp', 'gen')
			espr = EmbSimilarityPrecisionRecallMetric(emb, emb_mode)
		else:
			reference_key, gen_key = ('rk', 'gk')
			espr = EmbSimilarityPrecisionRecallMetric(emb, emb_mode, \
													  reference_key, gen_key)

		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   ref_len=ref_len, gen_len=gen_len, \
								   ref_vocab=ref_vocab, gen_vocab=gen_vocab)

		_data = copy.deepcopy(data)
		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match="Batch num is not matched."):
				espr.forward(data)
		else:
			if emb_len < dataloader.all_vocab_size and \
				(ref_vocab == 'all_vocab' or gen_vocab == 'all_vocab'):
				with pytest.raises(ValueError, match="[a-z]* index out of range."):
					espr.forward(data)
			else:
				espr.forward(data)
				ans = espr.close()
				prefix = emb_mode + '-bow'
				assert sorted(ans.keys()) == [prefix + ' precision', prefix + ' recall']

		assert same_dict(data, _data)

perplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
	(zip(test_resp_len), "add"),
	(zip(test_include_invalid), "multi")
)

class TestPerplexityMetric():
	def get_perplexity(self, input, dataloader, invalid_vocab=False, \
					   reference_key='resp_allvocabs', reference_len_key='resp_length', \
					   gen_prob_key='gen_log_prob'):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			max_length = input[reference_len_key][i]

			for j in range(max_length - 1):
				vocab_now = input[reference_key][i][j + 1]
				if vocab_now == dataloader.unk_id:
					continue
				if vocab_now < dataloader.vocab_size:
					word_loss += -(input[gen_prob_key][i][j][vocab_now])
				else:
					invalid_log_prob = input[gen_prob_key][i][j][dataloader.unk_id] - \
									 np.log(dataloader.all_vocab_size - dataloader.vocab_size)
					if invalid_vocab:
						word_loss += -np.log(np.exp(invalid_log_prob) + \
											np.exp(input[gen_prob_key][i][j][vocab_now]))
					else:
						word_loss += -invalid_log_prob
				length_sum += 1
		# print('test_metric.word_loss: ', word_loss)
		# print('test_metric.length_sum: ', 	length_sum)
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, check, ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid', \
		perplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, \
				   ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, reference_len_key, gen_prob_key = ('resp_allvocabs', 'resp_length', 'gen_log_prob') \
			if argument == 'default' else ('ra', 'rl', 'glp')
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len, \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len=resp_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			pm = PerplexityMetric(dataloader, invalid_vocab=include_invalid, full_check=(check=='full_check'))
		else:
			pm = PerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
								   invalid_vocab=include_invalid,  full_check=(check=='full_check'))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				pm.forward(data)
		elif check == 'no_check':
			if resp_len == '<2':
				with pytest.raises(ValueError, match='resp_length must no less than 2,' \
													 ' because <go> and <eos> are always included.'):
					pm.forward(data)
			elif include_invalid != (gen_prob_vocab == 'all_vocab'):
				with pytest.raises(ValueError):
					pm.forward(data)
			else:
				pm.forward(data)
				assert np.isclose(pm.close()['perplexity'], \
								  self.get_perplexity(data, dataloader, include_invalid, \
													  reference_key, reference_len_key, gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match='data\[gen_log_prob_key\] must be processed after log_softmax.'):
				pm.forward(data)
		assert same_dict(data, _data)

multiperplexity_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_check), "add"),
	(zip(test_ref_len), "multi"),
	(zip(test_ref_vocab), "multi"),
	(zip(test_gen_prob_vocab), "multi"),
	(zip(test_resp_len), "multi"),
	(zip(test_include_invalid), "multi")
)


class TestMultiTurnPerplexityMetric:
	def get_perplexity(self, input, dataloader, invalid_vocab=False, \
					   reference_key='sent_allvocabs', reference_len_key='sent_length', \
					   gen_prob_key='gen_prob'):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			for turn in range(len(input[reference_key][i])):
				max_length = input[reference_len_key][i][turn]
				gen_prob_turn = input[gen_prob_key][i][turn]
				for j in range(max_length - 1):
					vocab_now = input[reference_key][i][turn][j + 1]
					if vocab_now == dataloader.unk_id:
						continue
					if vocab_now < dataloader.vocab_size:
						word_loss += -(gen_prob_turn[j][vocab_now])
					else:
						invalid_log_prob = gen_prob_turn[j][dataloader.unk_id] - \
										 np.log(dataloader.all_vocab_size - dataloader.vocab_size)
						if invalid_vocab:
							word_loss += -np.log(np.exp(invalid_log_prob) + \
												np.exp(gen_prob_turn[j][vocab_now]))
						else:
							word_loss += -invalid_log_prob
					length_sum += 1
		return np.exp(word_loss / length_sum)

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, check, ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid', \
		multiperplexity_test_parameter)
	def test_close(self, argument, shape, type, batch_len, check, \
				   ref_len, ref_vocab, gen_prob_vocab, resp_len, include_invalid):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random_check' or 'full_check' or 'no_check'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, reference_len_key, gen_prob_key = ('sent_allvocabs', 'sent_length', 'gen_log_prob') \
			if argument == 'default' else ('rk', 'rl', 'gp')
		data = dataloader.get_data(reference_key=reference_key, \
								   reference_len_key=reference_len_key, gen_prob_key=gen_prob_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_prob_check=check, ref_len=ref_len, \
								   ref_vocab=ref_vocab, gen_prob_vocab=gen_prob_vocab, \
								   resp_len = resp_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtpm = MultiTurnPerplexityMetric(dataloader, \
											 invalid_vocab=include_invalid, full_check=(check=='full_check'))
		else:
			mtpm = MultiTurnPerplexityMetric(dataloader, reference_key, reference_len_key, gen_prob_key, \
								   invalid_vocab=include_invalid,  full_check=(check=='full_check'))

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtpm.forward(data)
		elif check == 'no_check':
			if resp_len == '<2':
				with pytest.raises(ValueError, match='resp_length must no less than 2,' \
													 ' because <go> and <eos> are always included.'):
					mtpm.forward(data)
			elif include_invalid != (gen_prob_vocab == 'all_vocab'):
				with pytest.raises(ValueError):
					mtpm.forward(data)
			else:
				mtpm.forward(data)
				assert np.isclose(mtpm.close()['perplexity'], \
								  self.get_perplexity(data, dataloader, include_invalid, \
													  reference_key, reference_len_key, gen_prob_key))
		else:
			with pytest.raises(ValueError, \
							   match='data\[gen_log_prob_key\] must be processed after log_softmax.'):
				mtpm.forward(data)
		assert same_dict(data, _data)


bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)


class TestBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim_index(gen_sen)
			resp_sen_processed = dataloader.trim_index(resp_sen[1:])
			refs.append([resp_sen_processed])
			gens.append(gen_sen_processed)
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeDataLoader()
		reference_key, gen_key = ('resp_allvocabs', 'gen') \
			if argument == 'default' else ('rk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			bcm = BleuCorpusMetric(dataloader)
		else:
			bcm = BleuCorpusMetric(dataloader, reference_key, gen_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				bcm.forward(data)
		else:
				bcm.forward(data)
				assert np.isclose(bcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	def test_bleu_bug(self):
		dataloader = FakeDataLoader()
		ref = [[2, 1, 3]]
		gen = [[1]]
		data = {'resp_allvocabs': ref, 'gen': gen}
		bcm = BleuCorpusMetric(dataloader)

		with pytest.raises(ZeroDivisionError):
			bcm.forward(data)
			bcm.close()

multi_bleu_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)


class TestMultiTurnBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for i in range(len(input[reference_key])):
			for resp_sen, gen_sen in zip(input[reference_key][i], input[gen_key][i]):
				gen_sen_processed = dataloader.trim_index(gen_sen)
				resp_sen_processed = dataloader.trim_index(resp_sen)
				gens.append(gen_sen_processed)
				refs.append([resp_sen_processed[1:]])
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', multi_bleu_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		reference_key, turn_len_key, gen_key = ('reference_allvocabs', 'turn_length', 'gen') \
			if argument == 'default' else ('rk', 'tlk', 'gk')
		data = dataloader.get_data(reference_key=reference_key, turn_len_key=turn_len_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbcm = MultiTurnBleuCorpusMetric(dataloader)
		else:
			mtbcm = MultiTurnBleuCorpusMetric(dataloader, reference_key, gen_key, turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbcm.forward(data)
		else:
			mtbcm.forward(data)
			assert np.isclose(mtbcm.close()['bleu'], self.get_bleu(dataloader, data, reference_key, gen_key))
		assert same_dict(data, _data)

	def test_bleu(self):
		dataloader = FakeMultiDataloader()
		ref = [[[2, 1, 3]]]
		gen = [[[1]]]
		turn_len = [1]
		data = {'reference_allvocabs': ref, 'gen': gen, 'turn_length': turn_len}
		mtbcm = MultiTurnBleuCorpusMetric(dataloader)

		with pytest.raises(ZeroDivisionError):
			mtbcm.forward(data)
			mtbcm.close()


single_turn_dialog_recorder_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(test_gen_len), "multi"),
	(zip(test_ref_len), "multi")
)

class TestSingleTurnDialogRecorder():
	def get_sen_from_index(self, dataloader, data, post_key='post_allvocabs', reference_key='resp_allvocabs', gen_key='gen'):
		ans = { \
			'post': [], \
			'resp': [], \
			'gen': [], \
			}
		for sen in data[post_key]:
			ans['post'].append(dataloader.index_to_sen(sen[1:]))
		for sen in data[reference_key]:
			ans['resp'].append(dataloader.index_to_sen(sen[1:]))
		for sen in data[gen_key]:
			ans['gen'].append(dataloader.index_to_sen(sen))

		return ans

	@pytest.mark.parametrize('argument, shape, type, batch_len, gen_len, ref_len', single_turn_dialog_recorder_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		post_key, reference_key, gen_key = ('post_allvocabs', 'resp_allvocabs', 'gen') \
			if argument == 'default' else ('pk', 'rk', 'gk')
		data = dataloader.get_data(post_key=post_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			sr = SingleTurnDialogRecorder(dataloader)
		else:
			sr = SingleTurnDialogRecorder(dataloader, post_key, reference_key, gen_key)

		if batch_len == 'unequal':
			data[post_key] = data[post_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				sr.forward(data)
		else:
			sr.forward(data)
			assert sr.close() == self.get_sen_from_index(dataloader, data, post_key, reference_key, \
																			gen_key)
		assert same_dict(data, _data)


multi_turn_dialog_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_batch_len), "add"),
	(zip(['empty', 'non-empty']), "multi"),
	(zip(['empty', 'non-empty']), "multi"),
	(zip(test_turn_len), "add")
)

class TestMultiTurnDialogRecorder:
	def check(self, ans, dataloader, data, context_key='context_allvocabs', \
			  resp_key='reference_allvocabs', gen_key='gen', turn_length='turn_length'):
		_ans = {'context': [], 'reference': [], 'gen': []}
		for i, context_turn in enumerate(data[context_key]):
			context_now = []
			for j, context in enumerate(context_turn):
				t = dataloader.trim_index(context[1:])
				if len(t) == 0:
					break
				context_now.append(t)
			_ans['context'].append(context_now)

		for i, resp_turn in enumerate(data[resp_key]):
			resp_now = []
			for j, resp in enumerate(resp_turn):
				t = dataloader.trim_index(resp[1:])
				if data[turn_length] is None:
					if len(t) == 0:
						break
				elif j >= data[turn_length][i]:
					break
				resp_now.append(t)
			_ans['reference'].append(resp_now)

		for i, gen_turn in enumerate(data[gen_key]):
			gen_now = []
			for j, gen in enumerate(gen_turn):
				t = dataloader.trim_index(gen)
				if data[turn_length] is None:
					if len(t) == 0:
						break
				elif j >= data[turn_length][i]:
					break
				gen_now.append(t)
			_ans['gen'].append(gen_now)

		print('_ans[\'context\']: ', _ans['context'])
		print('ans[\'context\']: ', ans['context'])
		assert len(ans['context']) == len(_ans['context'])
		assert len(ans['reference']) == len(_ans['reference'])
		assert len(ans['gen']) == len(_ans['gen'])
		for i, turn in enumerate(ans['context']):
			assert len(_ans['context'][i]) == len(turn)
		for i, turn in enumerate(ans['reference']):
			assert len(_ans['reference'][i]) == len(turn)
		for i, turn in enumerate(ans['gen']):
			assert len(_ans['gen'][i]) == len(turn)

	@pytest.mark.parametrize( \
		'argument, shape, type, batch_len, gen_len, ref_len, turn_len', multi_turn_dialog_test_parameter)
	def test_close(self, argument, shape, type, batch_len, gen_len, ref_len, turn_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		# 'random', 'non-empty', 'empty'
		# 'random', 'non-empty', 'empty'
		dataloader = FakeMultiDataloader()
		context_key, reference_key, gen_key, turn_len_key = ('context_allvocabs', 'reference_allvocabs', 'gen', 'turn_length') \
			if argument == 'default' else ('ck', 'rk', 'gk', 'tk')
		data = dataloader.get_data(context_key=context_key, turn_len_key=turn_len_key, reference_key=reference_key, gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'), \
								   gen_len=gen_len, ref_len=ref_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			mtbr = MultiTurnDialogRecorder(dataloader)
		else:
			mtbr = MultiTurnDialogRecorder(dataloader, context_key, reference_key, gen_key,
										   turn_len_key)

		if batch_len == 'unequal':
			data[reference_key] = data[reference_key][1:]
			_data = copy.deepcopy(data)
			with pytest.raises(ValueError, match='Batch num is not matched.'):
				mtbr.forward(data)
		else:
			if turn_len == 'unequal':
				data[reference_key][0] = data[reference_key][0][1:]
				with pytest.raises(ValueError, match="Reference turn num \d* != gen turn num \d*."):
					mtbr.forward(data)
				return
			else:
				mtbr.forward(data)
				self.check(mtbr.close(), dataloader, \
					data, context_key, reference_key, gen_key, turn_len_key)

		assert same_dict(data, _data)


language_generation_test_parameter = generate_testcase(\
	(zip(test_argument), "add"),
	(zip(test_shape, test_type), "multi"),
	(zip(test_gen_len), "multi"),
)


class TestLanguageGenerationRecorder():
	def get_sen_from_index(self, dataloader, data, gen_key='gen'):
		ans = []
		for sen in data[gen_key]:
			ans.append(dataloader.index_to_sen(sen))
		return ans

	@pytest.mark.parametrize('argument, shape, type, gen_len', language_generation_test_parameter)
	def test_close(self, argument, shape, type, gen_len):
		# 'default' or 'custom'
		# 'pad' or 'jag'
		# 'list' or 'array'
		# 'equal' or 'unequal'
		dataloader = FakeDataLoader()
		gen_key = 'gen' \
			if argument == 'default' else 'gk'
		data = dataloader.get_data(gen_key=gen_key, \
								   to_list=(type == 'list'), pad=(shape == 'pad'),
								   gen_len=gen_len)
		_data = copy.deepcopy(data)
		if argument == 'default':
			lg = LanguageGenerationRecorder(dataloader)
		else:
			lg = LanguageGenerationRecorder(dataloader, gen_key)

		lg.forward(data)
		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, gen_key)
		assert same_dict(data, _data)


hash_value_recorder_test_parameter = generate_testcase(\
	(zip(test_argument), "multi"),
	(zip(test_hash_data), "multi")
)


class TestHashValueRecorder():
	@pytest.mark.parametrize('argument, hash_data', hash_value_recorder_test_parameter)
	def test_close(self, argument, hash_data):
		if argument == 'default':
			hash_key = 'hashvalue'
			hvr = HashValueRecorder()
		else:
			hash_key = 'hk'
			hvr = HashValueRecorder(hash_key)

		if hash_data == 'has_key':
			t = []
			for i in range(32):
				t.append(random.randrange(0, 100))
			data = {'hashvalue': bytes(t)}
			hvr.forward(data)
			ans = hvr.close()
			assert type(ans[hash_key]) == bytes
			assert len(ans[hash_key]) == 32
		else:
			data = {}
			hvr.forward(data)
			ans = hvr.close()
			assert ans == {}



class TestMetricChain():
	def test_init(self):
		mc = MetricChain()

	def test_add_metric(self):
		mc = MetricChain()
		with pytest.raises(TypeError):
			mc.add_metric([1, 2, 3])

	def test_close1(self):
		dataloader = FakeMultiDataloader()
		data = dataloader.get_data(reference_key='reference_key', reference_len_key='reference_len_key', \
								   turn_len_key='turn_len_key', gen_prob_key='gen_prob_key', \
								   gen_key='gen_key', context_key='context_key')
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key', \
									   invalid_vocab=True, full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity( \
			data, dataloader, True, 'reference_key', 'reference_len_key', 'gen_prob_key')

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'gen_key', 'turn_len_key')
		bleu = TestMultiTurnBleuCorpusMetric().get_bleu(dataloader, data, 'reference_key', 'gen_key')

		_data = copy.deepcopy(data)
		mc = MetricChain()
		mc.add_metric(pm)
		mc.add_metric(bcm)
		mc.forward(data)
		res = mc.close()

		assert np.isclose(res['perplexity'], perplexity)
		assert np.isclose(res['bleu'], bleu)
		assert same_dict(data, _data)

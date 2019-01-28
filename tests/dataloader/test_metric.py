import copy

import numpy as np
import pytest

from random import random, randrange
from contk.metric import MetricBase, PerlplexityMetric, MultiTurnPerplexityMetric, BleuCorpusMetric, \
	MultiTurnBleuCorpusMetric, SingleTurnDialogRecorder, MultiTurnDialogRecorder, LanguageGenerationRecorder, \
	MetricChain
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def test_bleu_bug():
	ref = [[[1, 3], [3], [4]]]
	gen = [[1]]
	with pytest.raises(ZeroDivisionError):
		corpus_bleu(ref, gen, smoothing_function=SmoothingFunction().method7)


class FakeDataLoader:
	def __init__(self):
		self.vocab_list = ['<pad>', '<unk>', '<go>', '<eos>', 'what', 'how', 'here', 'do']
		self.vocab_to_index = {'<pad>': 0, '<unk>': 1, '<go>': 2, '<eos>': 3, 'what': 4, 'how': 5, 'here': 6, 'do': 7}
		self.vocab_size = 8

	def trim_before_target(self, lists, target):
		try:
			lists = lists[:lists.index(target)]
		except ValueError:
			pass
		return lists

	def trim_index(self, index):
		print(index)
		index = self.trim_before_target(list(index), 3)
		idx = len(index)
		while idx > 0 and index[idx - 1] == 0:
			idx -= 1
		index = index[:idx]
		return index

	def multi_turn_sen_to_index(self, session):
		return list(map(lambda sent: list(map( \
			lambda word: self.word2id.get(word, self.unk_id), sent)), \
						session))

	def multi_turn_trim_index(self, index):
		res = []
		for turn_index in index:
			turn_trim = self.trim_index(turn_index)
			if turn_trim:
				res.append(turn_trim)
			else:
				break
		return res

	def multi_turn_index_to_sen(self, index, trim=True):
		if trim:
			index = self.multi_turn_trim_index(index)
		return list(map(lambda sent: \
							list(map(lambda word: self.vocab_list[word], sent)), \
						index))

	def index_to_sen(self, index, trim=True):
		if trim:
			index = self.trim_index(index)
		return list(map(lambda word: self.vocab_list[word], index))

	def get_sen(self, max_len, len, gen=False, pad=True):
		sen = []
		for i in range(len):
			sen.append(randrange(4, self.vocab_size))
		if not gen:
			sen[0] = self.vocab_to_index['<go>']
		sen[len - 1] = self.vocab_to_index['<eos>']
		if pad:
			for i in range(max_len - len):
				sen.append(self.vocab_to_index['<pad>'])
		return sen

	def get_data(self, reference_key=None, reference_len_key=None, gen_prob_key=None, gen_key=None, \
					   post_key=None, resp_key=None, context_key=None, multi_turn=False, to_list=False, \
				 pad=True, random_check=True, full_check=True, \
				 different_turn_len=False, \
				 ref_len_flag=2, gen_len_flag=2, \
				 batch=5, length=15):
		data = { \
			reference_key: [], \
			reference_len_key: [], \
			gen_prob_key: [], \
			gen_key: [], \
			post_key: [], \
			resp_key: [], \
			context_key: [], \
		}
		for i in range(batch):
			single_batch = { \
				reference_key: [], \
				reference_len_key: [], \
				gen_prob_key: [], \
				gen_key: [], \
				post_key: [], \
				resp_key: [], \
				context_key: [], \
			}
			turn_len = randrange(1, 5) if different_turn_len else 5
			for turn in range(turn_len if multi_turn else 1):
				ref = [[], [], [], []]
				gen = [[]]
				gen_prob = []
				ref_len = int()

				for j in range(4):
					tmp = randrange(2, length)
					ref[j] = self.get_sen(length, tmp, pad = True)
					if j == 0:
						ref_len = tmp

				gen[0] = self.get_sen(length, randrange(2, length), gen = True, pad = pad)

				if ref_len_flag < 2:
					ref[0] = self.get_sen(length, ref_len_flag + 2, gen = False, pad = True)
					ref_len = ref_len_flag
				if gen_len_flag < 2:
					gen[0] = self.get_sen(length, gen_len_flag + 1, gen = True, pad = True)

				for j in range(ref_len - 1 if not pad else length):
					vocab_prob = []
					for k in range(self.vocab_size):
						vocab_prob.append(random())
					vocab_prob /= np.sum(vocab_prob)
					if random_check == True:
						vocab_prob = np.log(vocab_prob)
					gen_prob.append(vocab_prob)

				# (self, reference_key, reference_len_key, gen_prob_key, gen_key, \
				#  post_key, resp_key, context_key, log_softmax=True, to_list=False, multi_turn=False):

				if not multi_turn:
					if reference_key:
						data[reference_key].append(ref[0])
					if reference_len_key:
						data[reference_len_key].append(ref_len)
					if gen_prob_key:
						data[gen_prob_key].append(gen_prob)
					if gen_key:
						data[gen_key].append(gen[0])
					if post_key:
						data[post_key].append(ref[1])
					if resp_key:
						data[resp_key].append(ref[2])
					if context_key:
						data[context_key].append(ref[3])
				else:
					if reference_key:
						single_batch[reference_key].append(ref[0])
					if reference_len_key:
						single_batch[reference_len_key].append(ref_len)
					if gen_prob_key:
						single_batch[gen_prob_key].append(gen_prob)
					if gen_key:
						single_batch[gen_key].append(gen[0])
					if post_key:
						single_batch[post_key].append(ref[1])
					if resp_key:
						single_batch[resp_key].append(ref[2])
					if context_key:
						single_batch[context_key].append(ref[3])

			if multi_turn:
				for key in data.keys():
					data[key].append(single_batch[key])
		if full_check == False:
			if multi_turn:
				data[gen_prob_key][0][0][0] -= 0.5
			else:
				data[gen_prob_key][0][0] -= 0.5
		if not to_list:
			for i in data.keys():
				data[i] = np.array(data[i])
		return data


class TestPerlplexityMetric():
	def get_perplexity(self, input, reference_key='resp', reference_len_key='resp_length', \
					   gen_prob_key='gen_prob'):
		length_sum = 0
		word_loss = 0
		for i in range(len(input[reference_key])):
			max_length = input[reference_len_key][i]

			length_sum += max_length - 1
			for j in range(max_length - 1):
				word_loss += -(input[gen_prob_key][i][j][input[reference_key][i][j + 1]])
		return np.exp(word_loss / length_sum)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='resp', reference_len_key='resp_length', gen_prob_key='gen_prob', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		pm = PerlplexityMetric(dataloader)
		_data = data
		pm.forward(data)

		assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data))
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		pm.forward(data)

		assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data, 'rfk', 'rlk', 'gpk'))

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=False, \
								   pad=False, random_check=True, full_check=True, \
								   different_turn_len=True)

		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		_data = data
		pm.forward(data)

		assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data, 'rfk', 'rlk', 'gpk'))
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=True, \
								   pad=True, random_check=True, full_check=True)

		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		_data = data
		pm.forward(data)

		assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data, 'rfk', 'rlk', 'gpk'))
		assert _data == data

	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=False, \
								   pad=False, random_check=True, full_check=True)

		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		_data = data
		pm.forward(data)

		assert np.isclose(pm.close()['perplexity'], self.get_perplexity(data, 'rfk', 'rlk', 'gpk'))
		assert _data == data

	def test_close6(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=True, \
								   pad=False, random_check=True, full_check=True)
		data['rfk'] = np.delete(data['rfk'], 1, 0)

		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		_data = data
		with pytest.raises(ValueError):
			pm.forward(data)
		assert _data == data

	def test_close7(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=True, \
								   pad=False, random_check=True, full_check=True)
		data['rfk'] = np.delete(data['rlk'], 1, 0)

		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		_data = data
		with pytest.raises(ValueError):
			pm.forward(data)
		assert _data == data

	def test_close8(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=True, \
								   pad=False, random_check=True, full_check=True)
		data['rfk'] = np.delete(data['gpk'], 1, 0)

		_data = data
		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		with pytest.raises(ValueError):
			pm.forward(data)
		assert _data == data

	def test_close9(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=True, \
								   pad=False, random_check=False, full_check=True)

		_data = data
		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		with pytest.raises(ValueError):
			pm.forward(data)
		assert _data == data

	def test_close10(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=False, to_list=True, \
								   pad=False, random_check=True, full_check=False)

		_data = data
		pm = PerlplexityMetric(dataloader, 'rfk', 'rlk', 'gpk', full_check=True)
		with pytest.raises(ValueError):
			pm.forward(data)
		assert _data == data


class TestMultiTurnPerplexityMetric:
	def get_perplexity(self, input, reference_key='sent', reference_len_key='sent_length', gen_prob_key='gen_prob'):
		length_sum = 0
		word_loss = 0
		for turn in range(len(input[reference_key])):
			for i in range(len(input[reference_key][turn])):
				max_length = input[reference_len_key][turn][i]

				length_sum += max_length - 1
				for j in range(max_length - 1):
					print(turn, i, j)
					word_loss += -(input[gen_prob_key][turn][i][j][input[reference_key][turn][i][j + 1]])
		return np.exp(word_loss / length_sum)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='sent', reference_len_key='sent_length', gen_prob_key='gen_prob', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader)
		mtpm.forward(data)

		assert np.isclose(mtpm.close()['perplexity'], self.get_perplexity(data))
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rk', 'rlk', 'gpk')
		mtpm.forward(data)

		assert np.isclose(mtpm.close()['perplexity'], self.get_perplexity(data, 'rk', 'rlk', 'gpk'))
		assert _data == data

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=False, \
								   pad=False, random_check=True, full_check=True, different_turn_len=True)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rk', 'rlk', 'gpk')
		mtpm.forward(data)

		assert np.isclose(mtpm.close()['perplexity'], self.get_perplexity(data, 'rk', 'rlk', 'gpk'))
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=True, \
								   pad=False, random_check=True, full_check=True, different_turn_len=True)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rk', 'rlk', 'gpk')
		mtpm.forward(data)

		assert np.isclose(mtpm.close()['perplexity'], self.get_perplexity(data, 'rk', 'rlk', 'gpk'))
		assert _data == data

	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=True, \
								   pad=False, random_check=False, full_check=True)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		with pytest.raises(ValueError):
			mtpm.forward(data)
		assert _data == data

	def test_close6(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=True, \
								   pad=False, random_check=True, full_check=False)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rfk', 'rlk', 'gpk', full_check=True)
		with pytest.raises(ValueError):
			mtpm.forward(data)
		assert _data == data

	def test_close7(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=True, \
								   pad=False, random_check=True, full_check=False)
		data['rfk'] = np.delete(data['rfk'], 1, 0)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		with pytest.raises(ValueError):
			mtpm.forward(data)
		assert _data == data

	def test_close8(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=True, \
								   pad=False, random_check=True, full_check=False)
		data['rlk'] = np.delete(data['rlk'], 1, 0)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		with pytest.raises(ValueError):
			mtpm.forward(data)
		assert _data == data

	def test_close9(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', reference_len_key='rlk', gen_prob_key='gpk', \
								   multi_turn=True, to_list=True, \
								   pad=False, random_check=True, full_check=False)
		data['gpk'] = np.delete(data['gpk'], 1, 0)

		_data = data
		mtpm = MultiTurnPerplexityMetric(dataloader, 'rfk', 'rlk', 'gpk')
		with pytest.raises(ValueError):
			mtpm.forward(data)
		assert _data == data

class TestBleuCorpusMetric:
	def get_bleu(self, dataloader, input, reference_key, gen_key):
		refs = []
		gens = []
		for gen_sen, resp_sen in zip(input[gen_key], input[reference_key]):
			gen_sen_processed = dataloader.trim_index(gen_sen)
			resp_sen_processed = dataloader.trim_index(resp_sen[1:])
			refs.append([resp_sen_processed])
			gens.append(gen_sen_processed)
		print('refs:', refs)
		print('gens:', gens)
		print('bleu:', corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7))
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='resp', gen_key='gen', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		bm = BleuCorpusMetric(dataloader)
		bm.forward(data)

		print(self.get_bleu(dataloader, data, 'resp', 'gen'))
		assert np.isclose(bm.close()['bleu'], self.get_bleu(dataloader, data, 'resp', 'gen'))
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		bm = BleuCorpusMetric(dataloader, 'rfk', 'gk')
		bm.forward(data)

		print(self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert np.isclose(bm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=False, to_list=True, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		bm = BleuCorpusMetric(dataloader, 'rfk', 'gk')
		bm.forward(data)

		print(self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert np.isclose(bm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True, \
								   gen_len_flag=0)

		_data = data
		bm = BleuCorpusMetric(dataloader, 'rfk', 'gk')
		bm.forward(data)

		assert self.get_bleu(dataloader, data, 'rfk', 'gk') == 0
		assert np.isclose(bm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True, \
								   ref_len_flag=0)

		_data = data
		bm = BleuCorpusMetric(dataloader, 'rfk', 'gk')
		bm.forward(data)

		print(self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert self.get_bleu(dataloader, data, 'rfk', 'gk') == 0
		assert np.isclose(bm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close6(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, full_check=True, \
								   ref_len_flag=0)
		data['rfk'] = np.delete(data['rfk'], -1, 0)
		_data = data
		bm = BleuCorpusMetric(dataloader, 'rfk', 'gk')
		with pytest.raises(ValueError):
			bm.forward(data)
		assert _data == data


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
		print('refs:', refs)
		print('gens:', gens)
		print('bleu:', corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7))
		return corpus_bleu(refs, gens, smoothing_function=SmoothingFunction().method7)

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='sent', gen_key='gen', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader)
		mtbm.forward(data)
		assert np.isclose(mtbm.close()['bleu'], self.get_bleu(dataloader, data, 'sent', 'gen'))
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader, 'rfk', 'gk')
		mtbm.forward(data)
		assert np.isclose(mtbm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=True, to_list=True, \
								   pad=True, random_check=True, full_check=True)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader, 'rfk', 'gk')
		mtbm.forward(data)
		assert np.isclose(mtbm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True, full_check=True, \
								   ref_len_flag=0)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader, 'rfk', 'gk')
		mtbm.forward(data)
		assert self.get_bleu(dataloader, data, 'rfk', 'gk') == 0
		assert np.isclose(mtbm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data

	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True, full_check=True, \
								   gen_len_flag=0)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader, 'rfk', 'gk')
		with pytest.raises(ValueError):
			mtbm.forward(data)
		assert _data == data

	def test_close6(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=True, to_list=False, \
								   pad=True, random_check=True)
		data['rfk'] = np.delete(data['rfk'], 1, 0)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader, 'rfk', 'gk')
		with pytest.raises(ValueError):
			mtbm.forward(data)
		assert _data == data

	def test_close7(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='rfk', gen_key='gk', \
								   multi_turn=True, to_list=False, \
								   pad=False, random_check=True, \
								   different_turn_len=True)

		_data = data
		mtbm = MultiTurnBleuCorpusMetric(dataloader, 'rfk', 'gk')
		mtbm.forward(data)
		assert np.isclose(mtbm.close()['bleu'], self.get_bleu(dataloader, data, 'rfk', 'gk'))
		assert _data == data


class TestSingleTurnDialogRecorder():
	def get_sen_from_index(self, dataloader, data, post_key='post', resp_key='resp', gen_key='gen'):
		ans = { \
			'post': [], \
			'resp': [], \
			'gen': [], \
			}
		for sen in data[post_key]:
			ans['post'].append(dataloader.index_to_sen(sen[1:]))
			print(ans['post'][-1])
		for sen in data[resp_key]:
			ans['resp'].append(dataloader.index_to_sen(sen[1:]))
			print(ans['resp'][-1])
		for sen in data[gen_key]:
			ans['gen'].append(dataloader.index_to_sen(sen))

		return ans

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(post_key='post', resp_key='resp', gen_key='gen', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True)

		_data = data
		sdr = SingleTurnDialogRecorder(dataloader)
		sdr.forward(data)

		assert sdr.close() == self.get_sen_from_index(dataloader, data)
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(post_key='pk', resp_key='rk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True)

		_data = data
		sdr = SingleTurnDialogRecorder(dataloader, 'pk', 'rk', 'gk')
		sdr.forward(data)

		assert sdr.close() == self.get_sen_from_index(dataloader, data, 'pk', 'rk', 'gk')
		assert _data == data

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(post_key='pk', resp_key='rk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, \
								   gen_len_flag=0)

		_data = data
		sdr = SingleTurnDialogRecorder(dataloader, 'pk', 'rk', 'gk')
		sdr.forward(data)

		assert sdr.close() == self.get_sen_from_index(dataloader, data, 'pk', 'rk', 'gk')
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(post_key='pk', resp_key='rk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=False, random_check=True)

		_data = data
		sdr = SingleTurnDialogRecorder(dataloader, 'pk', 'rk', 'gk')
		sdr.forward(data)

		assert sdr.close() == self.get_sen_from_index(dataloader, data, 'pk', 'rk', 'gk')
		assert _data == data

	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(post_key='pk', resp_key='rk', gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=False, random_check=True)
		data['pk'] = np.delete(data['pk'], 2, 0)
		_data = data
		sdr = SingleTurnDialogRecorder(dataloader, 'pk', 'rk', 'gk')
		with pytest.raises(ValueError):
			sdr.forward(data)
		assert _data == data


class TestMultiTurnDialogRecorder:
	def get_sen_from_index(self, dataloader, data, post_key='context', resp_key='reference', gen_key='gen'):
		ans = { \
			'context': [], \
			'reference': [], \
			'gen': [], \
			}
		for turn in data[post_key]:
			ans['context'].append(dataloader.multi_turn_index_to_sen(np.array(turn)[ :, 1 :]))
		for turn in data[resp_key]:
			ans['reference'].append(dataloader.multi_turn_index_to_sen(np.array(turn)[ :, 1 :]))
		for turn in data[gen_key]:
			ans['gen'].append(dataloader.multi_turn_index_to_sen(turn))

		return ans

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(context_key='post', reference_key='resp', gen_key='gen', \
								   multi_turn=True, pad=True)
		_data = data
		mdr = MultiTurnDialogRecorder(dataloader)
		mdr.forward(data)

		assert mdr.close() == self.get_sen_from_index(dataloader, data)
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(context_key='ck', reference_key='rk', gen_key='gk', \
								   multi_turn=True, pad=True)
		_data = data
		mdr = MultiTurnDialogRecorder(dataloader, 'ck', 'rk', 'gk')
		mdr.forward(data)

		assert mdr.close() == self.get_sen_from_index(dataloader, data, 'ck', 'rk', 'gk')
		assert _data == data

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(context_key='ck', reference_key='rk', gen_key='gk', \
								   multi_turn=True, pad=True, to_list=True)
		_data = data
		mdr = MultiTurnDialogRecorder(dataloader, 'ck', 'rk', 'gk')
		mdr.forward(data)

		assert mdr.close() == self.get_sen_from_index(dataloader, data, 'ck', 'rk', 'gk')
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(context_key='ck', reference_key='rk', gen_key='gk', \
								   multi_turn=True, pad=True, to_list=True)
		mdr = MultiTurnDialogRecorder(dataloader, 'ck', 'rk', 'gk')
		data['ck'] = np.delete(data['ck'], 1, 0)
		_data = data

		with pytest.raises(ValueError):
			mdr.forward(data)
		assert _data == data

	def test_close5(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(context_key='ck', reference_key='rk', gen_key='gk', \
								   multi_turn=True, pad=True, to_list=True)
		mdr = MultiTurnDialogRecorder(dataloader, 'ck', 'rk', 'gk')
		data['gk'] = np.delete(data['gk'], 1, 0)
		_data = data

		with pytest.raises(ValueError):
			mdr.forward(data)
		assert _data == data


class TestLanguageGenerationRecorder():
	def get_sen_from_index(self, dataloader, data, gen_key = 'gen'):
		ans = []
		for sen in data[gen_key]:
			ans.append(dataloader.index_to_sen(sen))
			print(ans[-1])
		return ans

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='sent', gen_key='gen', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True)

		_data = data
		lg = LanguageGenerationRecorder(dataloader)
		lg.forward(data)

		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data)
		assert _data == data

	def test_close2(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True)

		_data = data
		lg = LanguageGenerationRecorder(dataloader, 'gk')
		lg.forward(data)

		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, 'gk')
		assert _data == data

	def test_close3(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, \
								   gen_len_flag=0)

		_data = data
		lg = LanguageGenerationRecorder(dataloader, 'gk')
		lg.forward(data)

		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, 'gk')
		assert _data == data

	def test_close4(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(gen_key='gk', \
								   multi_turn=False, to_list=False, \
								   pad=True, random_check=True, \
								   gen_len_flag=1)

		_data = data
		lg = LanguageGenerationRecorder(dataloader, 'gk')
		lg.forward(data)

		assert lg.close()['gen'] == self.get_sen_from_index(dataloader, data, 'gk')
		assert _data == data


class TestMetricChain():
	def test_init(self):
		mc = MetricChain()

	def test_add_metric(self):
		mc = MetricChain()
		with pytest.raises(TypeError):
			mc.add_metric([1, 2, 3])

	def test_close1(self):
		dataloader = FakeDataLoader()
		data = dataloader.get_data(reference_key='reference_key', reference_len_key='reference_len_key', gen_prob_key='gen_prob_key', \
								   gen_key='gen_key', post_key='post_key', resp_key='resp_key', \
								   multi_turn=True)
		pm = MultiTurnPerplexityMetric(dataloader, 'reference_key', 'reference_len_key', 'gen_prob_key',
									   full_check=True)
		perplexity = TestMultiTurnPerplexityMetric().get_perplexity(data, 'reference_key', 'reference_len_key', 'gen_prob_key')

		bcm = MultiTurnBleuCorpusMetric(dataloader, 'reference_key', 'gen_key')
		bleu = TestMultiTurnBleuCorpusMetric().get_bleu(dataloader, data, 'reference_key', 'gen_key')

		_data = data
		mc = MetricChain()
		mc.add_metric(pm)
		mc.add_metric(bcm)
		mc.forward(data)
		res = mc.close()

		assert np.isclose(res['perplexity'], perplexity)
		assert np.isclose(res['bleu'], bleu)
		assert _data == data

import copy
import pytest

from contk.dataloader import MultiTurnDialog, UbuntuCorpus
from contk.metric import MetricBase

class TestMultiTurnDialog():
	def base_test_init(self, dl):
		assert isinstance(dl, MultiTurnDialog)
		assert isinstance(dl.ext_vocab, list)
		assert dl.ext_vocab[:5] == ["<pad>", "<unk>", "<go>", "<eos>", "<eot>"]
		assert [dl.pad_id, dl.unk_id, dl.go_id, dl.eos_id, dl.eot_id] == [0, 1, 2, 3, 4]
		assert isinstance(dl.key_name, list)
		assert dl.key_name
		for word in dl.key_name:
			assert isinstance(word, str)
		assert isinstance(dl.vocab_list, list)
		assert dl.vocab_list[:len(dl.ext_vocab)] == dl.ext_vocab
		assert isinstance(dl.word2id, dict)
		print(dl.word2id)
		print(dl.vocab_list)
		assert len(dl.word2id) == len(dl.vocab_list)
		for i, word in enumerate(dl.vocab_list):
			assert isinstance(word, str)
			assert dl.word2id[word] == i
		assert dl.vocab_size == len(dl.vocab_list)
		assert isinstance(dl.data, dict)
		assert len(dl.data) == 3
		for key in dl.key_name:
			assert isinstance(dl.data[key], dict)
			assert isinstance(dl.data[key]['session'], list)
			assert isinstance(dl.data[key]['session'][0], list)
			content = dl.data[key]['session'][0][0]
			assert content[0] == dl.go_id
			assert content[-1] == dl.eot_id

	def base_test_restart(self, dl):
		with pytest.raises(ValueError):
			dl.restart("unknown set")
		for key in dl.key_name:
			with pytest.raises(ValueError):
				dl.restart(key)
			record_index = copy.copy(dl.index[key])
			dl.restart(key, batch_size=3, shuffle=False)
			assert record_index == dl.index[key]
			assert dl.batch_id[key] == 0
			assert dl.batch_size[key] == 3
			dl.restart(key, shuffle=True)
			assert dl.batch_id[key] == 0
			record_index = copy.copy(dl.index[key])
			dl.restart(key, shuffle=False)
			assert record_index == dl.index[key]
			assert dl.batch_id[key] == 0

	def base_test_get_batch(self, dl):
		for key in dl.key_name:
			batch = dl.get_batch(key, [0, 1])
			assert len(dl.index[key]) >= 2
			assert all([len(turn['length']) == 2 for turn in batch])
			assert all([len(turn['content']) == 2 for turn in batch])
			assert all([turn['length'].shape[0] == 2 for turn in batch])
			assert all([turn['content'].shape[0] == 2 for turn in batch])

	def base_test_get_next_batch(self, dl):
		with pytest.raises(ValueError):
			dl.get_next_batch("unknown set")
		for key in dl.key_name:
			with pytest.raises(RuntimeError):
				dl.get_next_batch(key)

			dl.restart(key, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(key, ignore_left_samples=True)
				if not batch:
					break
				assert all([turn['content'].shape[0] == 7 for turn in batch])
				assert all([turn['length'].shape[0] == 7 for turn in batch])
				sample_num += batch[0]['content'].shape[0]
			assert sample_num + 7 >= len(dl.data[key]['session'])

			dl.restart(key, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(key)
				if not batch:
					break
				sample_num += batch[0]['content'].shape[0]
			assert sample_num == len(dl.data[key]['session'])

	def base_test_convert(self, dl):
		sent_id = [0, 1, 2]
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.index_to_sen(sent_id)
		assert sent_id == dl.sen_to_index(sent)

		sent = ["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"]
		sent_id = [1, 2, 0, 1, 0, 2]
		assert sent_id == dl.sen_to_index(sent)

		sent_id = [0, 1, 2, 3, 0, 4, 1, 0, 0]
		sent = ["<pad>", "<unk>", "<go>", "<eos>", "<pad>", "<eot>", "<unk>", "<pad>", "<pad>"]
		assert sent == dl.index_to_sen(sent_id, trim=False)
		sent = ["<pad>", "<unk>", "<go>", "<eos>"]
		assert sent == dl.index_to_sen(sent_id)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.vocab_list == dl_list[0].vocab_list for x in dl_list)

@pytest.fixture
def load_ubuntucorpus():
	def _load_ubuntucorpus():
		return UbuntuCorpus("./tests/dataloader/dummy_ubuntucorpus")
	return _load_ubuntucorpus

class TestUbuntuCorpus(TestMultiTurnDialog):
	def test_init(self, load_ubuntucorpus):
		super().base_test_init(load_ubuntucorpus())

	def test_restart(self, load_ubuntucorpus):
		super().base_test_restart(load_ubuntucorpus())

	def test_get_batch(self, load_ubuntucorpus):
		super().base_test_get_batch(load_ubuntucorpus())

	def test_get_next_batch(self, load_ubuntucorpus):
		super().base_test_get_next_batch(load_ubuntucorpus())

	def test_convert(self, load_ubuntucorpus):
		super().base_test_convert(load_ubuntucorpus())

	def test_init_multi_runs(self, load_ubuntucorpus):
		super().base_test_multi_runs([load_ubuntucorpus() for i in range(3)])

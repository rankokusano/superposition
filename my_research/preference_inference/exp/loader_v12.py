"""
v12 proposal C (docs/v12_experiment_log.md Sec.45): data loader whose network INPUT view comes from another key
(e.g. 'self_vision_in', A-1's view with A-2 removed on hidden steps) while the prediction TARGET (y) and everything
else stay exactly as in exp/loader.py (subclasses; exp/loader.py is not changed).
"""

import torch

import util
from exp import loader


class SequenceLoaderInput(loader.SequenceLoader):
    def __init__(self, data, batch_begin, n_steps, input_key):
        super().__init__(data, batch_begin, n_steps)
        self.input_key = input_key

    def __next__(self):
        x, y, t = super().__next__()
        x['self_vision'] = self.data[self.input_key][:, t]
        return x, y, t


class SeqLoaderInput(loader.SeqLoader):
    def __init__(self, data, batch_size, n_data, n_steps, shuffle, device, input_key):
        super().__init__(data, batch_size, n_data, n_steps, shuffle, device)
        self.input_key = input_key

    def __next__(self):
        seq, batch_index = super().__next__()
        batch = seq.data
        batch[self.input_key] = torch.from_numpy(util.scale_vision(
            util.transpose_vision(self.data[self.input_key][batch_index]))).to(self.device)
        return SequenceLoaderInput(batch, seq.batch_begin, seq.n_steps, self.input_key), batch_index


class DataLoaderInput(loader.DataLoader):
    def __init__(self, data, batch_size, shuffle, device, input_key):
        super().__init__(data, batch_size, shuffle, device)
        self.input_key = input_key

    def load_sequence(self):
        if not hasattr(self, '_seq'):
            self._seq = SeqLoaderInput(self.data, self.batch_size, self.n_data, self.n_steps, self.shuffle,
                                       self.device, self.input_key)
        return self._seq


def gen_data_loader(data_name, batch_size, data_load_memory, device, input_key):
    data = util.load_data(util.gen_data_filename(data_name), data_load_memory)
    if 'test' not in data:
        data['test'] = data['train']
    return (DataLoaderInput(data['train'], batch_size, True, device, input_key),
            DataLoaderInput(data['train'], batch_size, False, device, input_key),
            DataLoaderInput(data['test'], batch_size, False, device, input_key))

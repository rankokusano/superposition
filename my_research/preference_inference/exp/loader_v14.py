"""
v14-C (docs/v14_experiment_log.md Sec.5): data loader that also passes A-1's per-episode liked colour
x['a1_liked_t'] (1 Red / 0 Green, data key 'a1_liked_t', shape (N, 101)) to the model, so that
model_v14.SuperpositionNetworkMotionPleasureHoldSigned can set the pleasure sign. Everything else is exactly
exp/loader.py (subclasses; exp/loader.py is not changed). Datasets without the key behave as exp/loader.py.
"""

import torch

import util
from exp import loader

KEY = 'a1_liked_t'


class SequenceLoaderLiked(loader.SequenceLoader):
    def __next__(self):
        x, y, t = super().__next__()
        if KEY in self.data:
            x[KEY] = self.data[KEY][:, t]
        return x, y, t


class SeqLoaderLiked(loader.SeqLoader):
    def __next__(self):
        seq, batch_index = super().__next__()
        batch = seq.data
        if KEY in self.data:
            batch[KEY] = torch.from_numpy(self.data[KEY][batch_index]).to(self.device)
        return SequenceLoaderLiked(batch, seq.batch_begin, seq.n_steps), batch_index


class DataLoaderLiked(loader.DataLoader):
    def load_sequence(self):
        if not hasattr(self, '_seq'):
            self._seq = SeqLoaderLiked(self.data, self.batch_size, self.n_data, self.n_steps, self.shuffle,
                                       self.device)
        return self._seq


def gen_data_loader(data_name, batch_size, data_load_memory, device):
    data = util.load_data(util.gen_data_filename(data_name), data_load_memory)
    if 'test' not in data:
        data['test'] = data['train']
    return (DataLoaderLiked(data['train'], batch_size, True, device),
            DataLoaderLiked(data['train'], batch_size, False, device),
            DataLoaderLiked(data['test'], batch_size, False, device))

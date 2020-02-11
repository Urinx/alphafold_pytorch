import json
import pickle
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NUM_RES = None
FEATURES = {
    'aatype': ('float32', [NUM_RES, 21]),
    'alpha_mask': ('int64', [NUM_RES, 1]),
    'alpha_positions': ('float32', [NUM_RES, 3]),
    'beta_mask': ('int64', [NUM_RES, 1]),
    'beta_positions': ('float32', [NUM_RES, 3]),
    'between_segment_residues': ('int64', [NUM_RES, 1]),
    'chain_name': ('string', [1]),
    'deletion_probability': ('float32', [NUM_RES, 1]),
    'domain_name': ('string', [1]),
    'gap_matrix': ('float32', [NUM_RES, NUM_RES, 1]),
    'hhblits_profile': ('float32', [NUM_RES, 22]),
    'hmm_profile': ('float32', [NUM_RES, 30]),
    'key': ('string', [1]),
    'mutual_information': ('float32', [NUM_RES, NUM_RES, 1]),
    'non_gapped_profile': ('float32', [NUM_RES, 21]),
    'num_alignments': ('int64', [NUM_RES, 1]),
    'num_effective_alignments': ('float32', [1]),
    'phi_angles': ('float32', [NUM_RES, 1]),
    'phi_mask': ('int64', [NUM_RES, 1]),
    'profile': ('float32', [NUM_RES, 21]),
    'profile_with_prior': ('float32', [NUM_RES, 22]),
    'profile_with_prior_without_gaps': ('float32', [NUM_RES, 21]),
    'pseudo_bias': ('float32', [NUM_RES, 22]),
    'pseudo_frob': ('float32', [NUM_RES, NUM_RES, 1]),
    'pseudolikelihood': ('float32', [NUM_RES, NUM_RES, 484]),
    'psi_angles': ('float32', [NUM_RES, 1]),
    'psi_mask': ('int64', [NUM_RES, 1]),
    'residue_index': ('int64', [NUM_RES, 1]),
    'resolution': ('float32', [1]),
    'reweighted_profile': ('float32', [NUM_RES, 22]),
    'sec_structure': ('int64', [NUM_RES, 8]),
    'sec_structure_mask': ('int64', [NUM_RES, 1]),
    'seq_length': ('int64', [NUM_RES, 1]),
    'sequence': ('string', [1]),
    'solv_surf': ('float32', [NUM_RES, 1]),
    'solv_surf_mask': ('int64', [NUM_RES, 1]),
    'superfamily': ('string', [1]),
}
Protein = collections.namedtuple('Protein', ['len', 'seq', 'inputs_1d', 'inputs_2d', 'inputs_2d_diagonal', 'scalars', 'targets'])

def tfrec_read(tfrec_file):
    import tensorflow as tf
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    features = [
        'aatype',
        'beta_mask',
        'beta_positions',
        'between_segment_residues',
        'chain_name',
        'deletion_probability',
        'domain_name',
        'gap_matrix',
        'hhblits_profile',
        'hmm_profile',
        'non_gapped_profile',
        'num_alignments',
        'num_effective_alignments',
        'profile',
        'profile_with_prior',
        'profile_with_prior_without_gaps',
        'pseudo_bias',
        'pseudo_frob',
        'pseudolikelihood',
        'residue_index',
        'resolution',
        'reweighted_profile',
        'sec_structure',
        'sec_structure_mask',
        'seq_length',
        'sequence',
        'solv_surf',
        'solv_surf_mask',
        'superfamily'
    ]
    features = {name: FEATURES[name] for name in features}

    def parse_tfexample(raw_data, features):
        feature_map = {k: tf.io.FixedLenSequenceFeature(shape=(), dtype=eval(f'tf.{v[0]}'), allow_missing=True) for k, v in features.items()}
        parsed_features = tf.io.parse_single_example(raw_data, feature_map)
        num_residues = tf.cast(parsed_features['seq_length'][0], dtype=tf.int32)

        for k, v in parsed_features.items():
            new_shape = [num_residues if s is None else s for s in FEATURES[k][1]]

            assert_non_empty = tf.assert_greater(tf.size(v), 0, name=f'assert_{k}_non_empty',
                message=f'The feature {k} is not set in the tf.Example. Either do not '
                'request the feature or use a tf.Example that has the feature set.')
            with tf.control_dependencies([assert_non_empty]):
                parsed_features[k] = tf.reshape(v, new_shape, name=f'reshape_{k}')

        return parsed_features

    raw_dataset = tf.data.TFRecordDataset([tfrec_file])
    raw_dataset = raw_dataset.map(lambda raw: parse_tfexample(raw, features))
    return raw_dataset

def tfrec2pkl(dataset, pkl_file):
    datalist = []
    dataset = dataset.batch(1)
    for x in dataset:
        data = {}
        for k, v in x.items():
            if k in ['sequence', 'domain_name', 'chain_name', 'resolution', 'superfamily', 'num_effective_alignments']:
                # print(f"{k}: {v.numpy()[0,0].decode('utf-8')}")
                if v.numpy().dtype == 'O':
                    data[k] = v.numpy()[0,0].decode('utf-8')
                else:
                    data[k] = v.numpy()[0,0]
            else:
                # print(k, v.numpy().shape)
                data[k] = v.numpy()[0]
        datalist.append(data)

    with open(pkl_file, 'wb') as f:
        pickle.dump(datalist, f)

    return datalist

def load_data(data_file, config):
    if data_file.endswith('.tfrec'):
        raw_dataset = tfrec_read(data_file)
        raw_dataset = tfrec2pkl(raw_dataset, data_file[:-5]+'pkl')
    else:
        raw_dataset = np.load(data_file, allow_pickle=True)

    def normalize(data):
        feature_normalization = {k: config.feature_normalization for k in config.network_config.features if k not in config.normalization_exclusion}
        copy_unnormalized = list(set(config.network_config.features) & set(config.network_config.targets))
        
        for k in copy_unnormalized:
            if k in data: data[f'{k}_unnormalized'] = data[k]
        
        range_epsilon = 1e-12
        for k, v in data.items():
            if k not in feature_normalization or feature_normalization[k] == 'none': pass
            elif feature_normalization[k] == 'std':
                train_range = np.sqrt(np.float32(config.norm_stats.var[k]))
                v = v - np.float32(config.norm_stats.mean[k])
                v = v / train_range if train_range > range_epsilon else v
                data[k] = v
            else:
                raise ValueError(f'Unknown normalization mode {feature_normalization[k]} for feature {k}.')

        return data

    dataset = [normalize(data) for data in raw_dataset]

    def convert_to_input(data):
        tensors_1d = []
        tensors_2d = []
        tensors_2d_diagonal = []
        L = len(data['sequence'])

        desired_features = config.network_config.features
        desired_scalars = config.network_config.scalars
        desired_targets = config.network_config.targets

        for k in desired_features:
            dim = len(FEATURES[k][1]) - 1
            if dim == 1:
                tensors_1d.append(np.float32(data[k]))
            elif dim == 2:
                if k not in data:
                    if not(f'{k}_cropped' in data and f'{k}_diagonal' in data):
                      raise ValueError(
                          f'The 2D feature {k} is not in the features dictionary and neither are its cropped and diagonal versions.')
                    else:
                      tensors_2d.append(np.float32(data[f'{k}_cropped']))
                      tensors_2d_diagonal.append(np.float32(data[f'{k}_diagonal']))
                else:
                    tensors_2d.append(np.float32(data[k]))

        inputs_1d = np.concatenate(tensors_1d, -1)
        if config.network_config.is_ca_feature:
            # The background model is not conditioned on the sequence
            # a binary feature δαβ to indicate whether the residue is a glycine (Cα atom) or not (Cβ)
            inputs_1d = inputs_1d[:, 7:8]
        inputs_2d = np.concatenate(tensors_2d, -1) if tensors_2d else np.zeros((L, L, 0), dtype=np.float32)
        
        if tensors_2d_diagonal:
            diagonal_crops1 = [t[:, :, :(t.shape[2] // 2)] for t in tensors_2d_diagonal]
            diagonal_crops2 = [t[:, :, (t.shape[2] // 2):] for t in tensors_2d_diagonal]
            inputs_2d_diagonal = np.concatenate(diagonal_crops1 + diagonal_crops2, 2)
        else:
            inputs_2d_diagonal = inputs_2d

        scalars = collections.namedtuple('ScalarClass', desired_scalars)(*[data.get(f'{k}_unnormalized', data[k]) for k in desired_scalars])
        targets = collections.namedtuple('TargetClass', desired_targets)(*[data.get(f'{k}_unnormalized', data[k]) for k in desired_targets])

        p = Protein(
            len=len(data['sequence']),
            seq=data['sequence'],
            inputs_1d=inputs_1d,
            inputs_2d=inputs_2d,
            inputs_2d_diagonal=inputs_2d_diagonal,
            scalars=scalars,
            targets=targets
        )
        return p

    dataset = [convert_to_input(data) for data in dataset]
    return dataset

class ProteinDataset(Dataset):
    def __init__(self, fname, config):
        super().__init__()
        self.dataset = load_data(fname, config)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def feature_1d_to_2d(x_1d, res_idx, L, crop_x, crop_y, crop_size_x, crop_size_y, binary_code_bits):
    res_idx = np.int32(res_idx)
    n_x, n_y = crop_size_x, crop_size_y
    range_scale = 100.0
    x_1d_y = np.pad(
        x_1d[max(0, crop_y[0]):crop_y[1]],
        [[max(0, -crop_y[0]), max(0, n_y - (L - crop_y[0]))],
        [0, 0]]
    ) # LxD
    range_n_y = np.pad(
        res_idx[max(0, crop_y[0]):crop_y[1]],
        [max(0, -crop_y[0]), max(0, n_y - (L - crop_y[0]))]
    ) # L
    x_1d_x = np.pad(
        x_1d[max(0, crop_x[0]):crop_x[1]], 
        [[max(0, -crop_x[0]), max(0, n_x - (L - crop_x[0]))],
        [0, 0]]
    ) # LxD
    range_n_x = np.pad(
        res_idx[max(0, crop_x[0]):crop_x[1]],
        [max(0, -crop_x[0]), max(0, n_x - (L - crop_x[0]))]
    ) # L

    offset = np.float32(np.expand_dims(range_n_x, 0) - np.expand_dims(range_n_y, 1)) / range_scale # LxL
    position_features = [
        np.tile(
            np.reshape((np.float32(range_n_y) - range_scale) / range_scale, [n_y, 1, 1]),
            [1, n_x, 1]
        ),
        np.reshape(offset, [n_y, n_x, 1])
    ]

    if binary_code_bits:
        exp_range_n_y = np.expand_dims(range_n_y, 1)
        bin_y = np.concatenate([exp_range_n_y // (1 << i) % 2 for i in range(binary_code_bits)], 1)
        exp_range_n_x = np.expand_dims(range_n_x, 1)
        bin_x = np.concatenate([exp_range_n_y // (1 << i) % 2 for i in range(binary_code_bits)], 1)
        position_features += [
            np.tile(
                np.expand_dims(np.float32(bin_y), 1),
                [1, n_x, 1],
            ),
            np.tile(
                np.expand_dims(np.float32(bin_x), 0),
                [n_y, 1, 1],
            )
        ]

    augmentation_features = position_features + [
        np.tile(
            np.expand_dims(x_1d_x, 0),
            [n_y, 1, 1]
        ),
        np.tile(
            np.expand_dims(x_1d_y, 1),
            [1, n_x, 1]
        )
    ]
    augmentation_features = np.concatenate(augmentation_features, -1)
    return augmentation_features

def make_crops(inputs_1d, inputs_2d, L, res_idx, crop_size_x, crop_step_x, crop_size_y, crop_step_y, binary_code_bits):
    for i in range(-crop_size_x // 2, L - crop_size_x // 2, crop_step_x):
        for j in range(-crop_size_y // 2, L - crop_size_y // 2, crop_step_y):
            '''
            start                                                   end
            |                                                       |
            i              crop_size_x               end_x          i                                        end_x
            |----------------------------------------|              |----------------------------------------|
            ....................KVEPVGNAYGHWTKHGKEFPEYQNAKQYVDAAHNFMTNPPLTNPPPGTLTKTRPNGD.....................
            |___________________|________________________________________|              |____________________|
                  prepad_x      0              crop_size_x                              L      postpad_x
                                ic
            '''
            end_x = i + crop_size_x
            end_y = j + crop_size_y
            crop_x = np.array([i, end_x], dtype=np.int32)
            crop_y = np.array([j, end_y], dtype=np.int32)
            ic = max(0, i)
            jc = max(0, j)
            end_x_cropped = min(L, end_x)
            end_y_cropped = min(L, end_y)
            prepad_x = max(0, -i)
            prepad_y = max(0, -j)
            postpad_x = end_x - end_x_cropped
            postpad_y = end_y - end_y_cropped

            cyx = np.pad(
                inputs_2d[jc:end_y, ic:end_x, :],
                [[prepad_y, postpad_y],
                [prepad_x, postpad_x],
                [0, 0]]
            )
            assert cyx.shape[0] == crop_size_y
            assert cyx.shape[1] == crop_size_x

            cxx = inputs_2d[ic:end_x, ic:end_x, :]
            if cxx.shape[0] < cyx.shape[0]:
                cxx = np.pad(cxx,
                    [[prepad_x, max(0, i + crop_size_y - L)],
                    [prepad_x, postpad_x],
                    [0, 0]]
                )
            assert cxx.shape[0] == crop_size_y
            assert cxx.shape[1] == crop_size_x

            cyy = inputs_2d[jc:end_y, jc:end_y, :]
            if cyy.shape[1] < cyx.shape[1]:
                cyy = np.pad(cyy,
                    [[prepad_y, postpad_y],
                    [prepad_y, max(0, j + crop_size_x - L)],
                    [0, 0]]
                )
            assert cyy.shape[0] == crop_size_y
            assert cyy.shape[1] == crop_size_x

            augmentation_features = feature_1d_to_2d(inputs_1d, res_idx, L, crop_x, crop_y, crop_size_x, crop_size_y, binary_code_bits) # LxLxD1
            x_2d = np.concatenate([cyx, cxx, cyy, augmentation_features], -1) # LxLx(3D2+D1)
            yield x_2d, crop_x, crop_y

def collate_fn(batch, config):
    assert len(batch) == 1
    protein = batch[0]
    crops = make_crops(protein.inputs_1d,
                       protein.inputs_2d,
                       protein.len,
                       protein.targets.residue_index.flatten(),
                       config.crop_size_x,
                       config.crop_size_x // config.eval_config.crop_shingle_x,
                       config.crop_size_y,
                       config.crop_size_y // config.eval_config.crop_shingle_y,
                       config.network_config.binary_code_bits)
    return protein, crops

def ProteinDataLoader(target_file, config):
    dataset = ProteinDataset(target_file, config)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda b: collate_fn(b, config))
    return dataloader

def test():
    '''
    > T1019s2
    KVEPVGNAYGHWTKHGKEFPEYQNAKQYVDAAHNFMTNPPPGTLTKTRPNGDTLYYNPVTNVFASKDINGVPRTMFKPEKGIEYWNKQ
    KVEPVGNAYGHWTKHGKEFPEYQNAKQYVDAAHNFMTNPPPGTLTKTRPNGDTLYYNPVTNVFA  s0
                       s24  AKQYVDAAHNFMTNPPPGTLTKTRPNGDTLYYNPVTNVFASKDINGVPRTMFKPEKGIEYWNKQ
    '''
    DISTOGRAM_MODEL = 'model/873731'
    replica = 0
    # dataset = ProteinDataset('T1019s2.tfrec', DISTOGRAM_MODEL, replica)
    dataset = ProteinDataset('T1019s2.pkl', DISTOGRAM_MODEL, replica)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    for batch in dataloader:
        for x, protein in batch:
            print(protein.targets.domain_name, x.shape)

if __name__ == '__main__':
    test()

import json
import torch
import numpy as np
from datetime import datetime
from collections import namedtuple

def timer(func):
    def wrapper(*args, **kwds):
        start_t = datetime.now()
        rets = func(*args, **kwds)
        end_t = datetime.now()
        if rets is not None:
            return (*rets, end_t-start_t)
        else:
            return end_t-start_t
    return wrapper

def build_config(model_path, replica):
    config_file = model_path / replica / 'config.json'
    stats_file = model_path / 'stats_train_s35.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    with open(stats_file, 'r') as f:
        norm_stats = json.load(f)

    default = {
        'torsion_multiplier': 0, 'collapsed_batch_norm': False,
        'filters_1d': [], 'is_ca_feature': False,
    }
    config['norm_stats'] = norm_stats
    config['network_config'] = {**default, **config['network_config']}
    exclude = norm_stats.keys()

    def make_nt(d, n):
        return namedtuple(n, d.keys())(**{k: make_nt(v, k) if isinstance(v, dict) and k not in exclude else v for k,v in d.items()})

    return make_nt(config, 'config')

@timer
def load_tf_ckpt(model, model_file_pt):
    import tensorflow as tf

    model_file_tf = model_file_pt.parent / 'tf_graph_data' / 'tf_graph_data.ckpt'
    for tf_name, tf_shape in tf.train.list_variables(str(model_file_tf)):
        tf_var = tf.train.load_variable(str(model_file_tf), tf_name)
        
        main_module, *others = tf_name.split('/')
        if main_module == 'ca_cb_logits': continue
        elif main_module.startswith('collapsed_embed'):
            n = int(main_module.split('_')[-1])
            pointer = model.collapsed_embed[n][1 if others[0] == 'BatchNorm' else 0]
        else:
            pointer = getattr(model, main_module)
            if main_module in ['Deep2D', 'Deep2DExtra']:
                if others[0].startswith('conv'):
                    pointer = getattr(pointer, others[0]).conv
                elif others[0].startswith('res'):
                    pointer = getattr(pointer, others[0].split('_')[0])
                    if others[0].endswith('1x1'):
                        pointer = pointer.conv_1x1
                    elif others[0].endswith('1x1h'):
                        pointer = pointer.conv_1x1h
                    elif others[0].endswith('3x3h'):
                        pointer = pointer.conv_3x3h

                    if others[-1] in ['w', 'b']:
                        pointer = pointer.conv
                    else:
                        pointer = pointer.bn
                elif others[0] == 'output_reshape_1x1h':
                    pointer = model.output_reshape_1x1h
                    if len(others) > 2:
                        pointer = pointer.bn
                    else:
                        pointer = pointer.conv

        if others:
            if others[-1] in ['weights', 'w']:
                pointer = pointer.weight
                if len(tf_var.shape) == 2:
                    # linear w
                    tf_var = tf_var.transpose()
                elif len(tf_var.shape) == 4:
                    #    tf conv w: [filter_height, filter_width, in_channels, out_channels]
                    # torch conv w: [out_channels, in_channels, filter_height, filter_width]
                    tf_var = tf_var.transpose((3, 2, 0, 1))
            elif others[-1] in ['biases', 'b']:
                pointer = pointer.bias
            elif others[-1] == 'beta':
                pointer = pointer.bias
            elif others[-1] == 'moving_mean':
                pointer = pointer.running_mean
            elif others[-1] == 'moving_variance':
                pointer = pointer.running_var

        try:
            assert pointer.shape == tf_var.shape
        except AssertionError as e:
            print(main_module, others)
            e.args += (pointer.shape, tf_var.shape)
            raise

        pointer.data = torch.from_numpy(tf_var)

    # save pytorch model
    torch.save(model.state_dict(), model_file_pt)

def save_seq_prob(prob, seq, out_file):
    SECONDARY_STRUCTURES = '-HETSGBI'
    if len(prob.shape) == 1:
        prob = prob.reshape(-1,1)
    L, n = prob.shape
    label = 'asa' if n == 1 else 'secstruct'

    with out_file.open('w') as f:
        f.write(f"# LABEL {label} CLASSES [{''.join(SECONDARY_STRUCTURES[:n])}]\n\n")
        for i in range(L):
            ss = SECONDARY_STRUCTURES[prob[i].argmax()]
            f.write(f"{i+1:4d} {seq[i]:1s} {ss:1s} {''.join(['%6.3f'%p for p in prob[i]])}\n")

def generate_domains(target, seq, crop_sizes='64,128,256', crop_step=32):
    windows = [int(x) for x in crop_sizes.split(",")]
    num_residues = len(seq)
    domains = []
    domains.append({"name": target, "description": (1, num_residues)})

    for window in windows:
        starts = list(range(0, num_residues - window, crop_step))
        if num_residues >= window:
            starts += [num_residues - window]
        for start in starts:
            name = f'{target}-l{window}_s{start}'
            domains.append({"name": name, "description": (start + 1, start + window)})
    
    return domains

def save_rr_file(probs, seq, domain, filename):
    assert len(seq) == probs.shape[0]
    assert len(seq) == probs.shape[1]
    
    with open(filename, 'w') as f:
        f.write(f'PFRMAT RR\nTARGET {domain}\nAUTHOR DM-ORIGAMI-TEAM\nMETHOD Alphafold - PyTorch\nMODEL 1\n{seq}\n')
        for i in range(probs.shape[0]):
            for j in range(i + 1, probs.shape[1]):
                f.write(f'{i+1:d} {j+1:d} 0 8 {probs[j,i]:f}\n')
        f.write('END\n')

def plot_contact_map(target, mats, out):
    import matplotlib.pyplot as plt
    if len(mats) == 1:
        fig, ax = plt.subplots()
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, len(mats), figsize=(11*len(mats),8))
    fig.subplots_adjust(wspace=0)

    for i, mat in enumerate(mats):
        if len(mat.shape) == 3 and mat.shape[-1] == 64:
            vmax = mat.shape[-1] - 1
            mat = mat.argmax(-1)
            im = axs[i].imshow(mat, cmap=plt.cm.Blues_r, vmin=0, vmax=vmax)
            cb = fig.colorbar(im, ax=axs[i])
            cb.set_ticks(np.linspace(0, vmax, 11))
            cb.set_ticklabels(range(2, 23, 2))
            if len(mats) != 1:
                axs[i].set_title('distance', fontsize=20)
        else:
            im = axs[i].imshow(mat, cmap=plt.cm.Blues, vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=axs[i])
            if len(mats) != 1:
                axs[i].set_title('contact', fontsize=20)

    if len(mats) == 1:
        plt.title(target)
        plt.savefig(out, dpi=300)
    else:
        fig.suptitle(target, fontsize=20)
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.5)

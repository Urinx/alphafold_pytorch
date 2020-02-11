import utils
import argparse
import itertools
import numpy as np
from pathlib import Path
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform

def make_crops(seq_file):
    target_line, *seq_line = seq_file.read_text().split('\n')
    target = seq_file.stem
    suffix = seq_file.suffix
    target_seq = ''.join(seq_line)

    for domain in utils.generate_domains(target, target_seq):
        name = domain['name']
        if name == target: continue
        crop_start, crop_end = domain["description"]
        seq = target_seq[crop_start-1:crop_end]
        (seq_file.parent / f'{name}{suffix}').write_text(f'>{name}\n{seq}')

def sequence_to_onehot(seq):
    mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
    num_entries = max(mapping.values()) + 1
    one_hot_arr = np.zeros((len(seq), num_entries), dtype=np.float32)

    for aa_index, aa_type in enumerate(seq):
        aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr

def extract_hmm_profile(hhm_file, sequence, asterisks_replace=0.0):
    """Extracts information from the hmm file and replaces asterisks."""
    profile_part = hhm_file.split('#')[-1]
    profile_part = profile_part.split('\n')
    whole_profile = [i.split() for i in profile_part]
    # This part strips away the header and the footer.
    whole_profile = whole_profile[5:-2]
    gap_profile = np.zeros((len(sequence), 10))
    aa_profile = np.zeros((len(sequence), 20))
    count_aa = 0
    count_gap = 0
    for line_values in whole_profile:
        if len(line_values) == 23:
            # The first and the last values in line_values are metadata, skip them.
            for j, t in enumerate(line_values[2:-1]):
                aa_profile[count_aa, j] = (2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
            count_aa += 1
        elif len(line_values) == 10:
            for j, t in enumerate(line_values):
                gap_profile[count_gap, j] = (2**(-float(t) / 1000.) if t != '*' else asterisks_replace)
            count_gap += 1
        elif not line_values:
            pass
        else:
            raise ValueError('Wrong length of line %s hhm file. Expected 0, 10 or 23'
                           'got %d'%(line_values, len(line_values)))
    hmm_profile = np.hstack([aa_profile, gap_profile])
    assert len(hmm_profile) == len(sequence)
    return hmm_profile

def read_aln(aln_file):
    aln = []
    aln_id = []
    seq = ''
    for line in aln_file.open():
        line = line.strip()
        if line[0] == '>':
            aln_id.append(line)
            if seq: aln.append(list(seq))
            seq = ''
        else:
            seq += line
    if seq: aln.append(list(seq))
    aln = np.array(aln)
    return aln, aln_id

def write_aln(aln, aln_id, out_file):
    with out_file.open('w') as f:
        for i in range(len(aln_id)):
            seq = ''.join(aln[i])
            f.write(f'{aln_id[i]}\n{seq}\n')

def sequence_weights(sequence_matrix):
    num_rows, num_res = sequence_matrix.shape
    cutoff = 0.62 * num_res
    weights = np.ones(num_rows, dtype=np.float32)
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            similarity = (sequence_matrix[i] == sequence_matrix[j]).sum()
            if similarity > cutoff:
                weights[i] += 1
                weights[j] += 1
    return 1.0 / weights

def calculate_f(align, theta=0.38):
    M, N = align.shape
    q = align.max()

    # W: 1*M
    W = 1 / (1 + np.sum(squareform(pdist(align,'hamming')<theta),0))
    Meff = np.sum(W)

    # cache a align residue table: q*N*M
    residue_table = np.zeros((q, N, M))
    for i in range(q):
        residue_table[i] = align.T == i+1

    # fi: N*q
    fi = np.array([np.sum(W * residue_table[i], 1) for i in range(q)]).T / Meff

    # this cost most time!
    fij = np.empty((N, N, q, q))
    for (A, B) in itertools.product(range(q), range(q)):
        for (i, j) in itertools.combinations(range(N), 2):
            t = np.sum(W * residue_table[A][i].T * residue_table[B][j].T)
            fij[i,j,A,B] = t
            fij[j,i,B,A] = t
    fij /= Meff
    for i in range(N):
        fij[i,i] = np.eye(q) * fi[i]

    del residue_table
    return fi, fij, Meff

def calculate_MI(fi, fij):
    N, q = fi.shape
    MI = np.zeros((N, N, 1), dtype=np.float32)

    for i, j in itertools.combinations(range(N),2):
        m = 0
        for (A, B) in itertools.product(range(q), range(q)):
            if fij[i,j,A,B] > 0:
                m += fij[i,j,A,B] * np.log( fij[i,j,A,B] / fi[i,A] / fi[j,B] )
        MI[i,j,0] = m
        MI[j,i,0] = m

    return MI

def feature_generation(seq_file, out_file):
    target_line, *seq_line = seq_file.read_text().split('\n')
    target = seq_file.stem
    target_seq = ''.join(seq_line)
    data_dir = seq_file.parent
    dataset = []

    for domain in utils.generate_domains(target, target_seq):
        name = domain['name']
        crop_start, crop_end = domain["description"]
        seq = target_seq[crop_start-1:crop_end]
        L = len(seq)
        hhm_file = data_dir / f'{name}.hhm'
        fas_file = data_dir / f'{name}.fas'
        aln_file = data_dir / f'{name}.aln'
        mat_file = data_dir / f'{name}.mat'

        if aln_file.exists():
            aln, _ = read_aln(aln_file)
        else:
            aln, aln_id = read_aln(fas_file)
            aln = aln[:, aln[0] != '-']
            write_aln(aln, aln_id, aln_file)
            exit()

        if mat_file.exists():
            mat = sio.loadmat(mat_file)
            pseudo_bias = np.float32(mat['pseudo_bias'])
            pseudo_frob = np.float32(np.expand_dims(mat['pseudo_frob'], -1))
            pseudolikelihood = np.float32(mat['pseudolikelihood'])
        else:
            pseudo_bias = np.zeros((L, 22), dtype=np.float32)
            pseudo_frob = np.zeros((L, L, 1), dtype=np.float32)
            pseudolikelihood = np.zeros((L, L, 484), dtype=np.float32)

        gap_count = np.float32(aln=='-')
        gap_matrix = np.expand_dims(np.matmul(gap_count.T, gap_count) / aln.shape[0], -1)

        mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX-')}
        seq_weight = sequence_weights(aln)
        hhblits_profile = np.zeros((L, 22), dtype=np.float32)
        reweighted_profile = np.zeros((L, 22), dtype=np.float32)
        for i in range(L):
            for j in range(aln.shape[0]):
                hhblits_profile[i, mapping[aln[j, i]]] += 1
                reweighted_profile[i, mapping[aln[j, i]]] += seq_weight[j]
        hhblits_profile /= hhblits_profile.sum(-1).reshape(-1, 1)
        reweighted_profile /= reweighted_profile.sum(-1).reshape(-1, 1)

        mapping = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYV-')}
        non_gapped_profile = np.zeros((L, 21), dtype=np.float32)
        for i in range(L):
            for j in aln[:, i]:
                non_gapped_profile[i, mapping[j]] += 1
        non_gapped_profile[:, -1] = 0
        non_gapped_profile /= non_gapped_profile.sum(-1).reshape(-1, 1)

        mapping = {aa: i for i, aa in enumerate('-ARNDCQEGHILKMFPSTWYVX')}
        a2n = np.frompyfunc(lambda x: mapping[x], 1, 1)
        fi, fij, Meff = calculate_f(a2n(aln))
        MI = calculate_MI(fi, fij)

        data = {
            'chain_name': target,
            'domain_name': name,
            'sequence': seq,
            'seq_length': np.ones((L, 1), dtype=np.int64)*L,
            'residue_index': np.arange(L, dtype=np.int64).reshape(L, 1),
            'aatype': sequence_to_onehot(seq),
            # profile: A profile (probability distribution over amino acid types)
            # computed using PSI-BLAST. Equivalent to the output of ChkParse.
            'hhblits_profile': hhblits_profile,
            'reweighted_profile': reweighted_profile,
            'hmm_profile': extract_hmm_profile(hhm_file.read_text(), seq),
            'num_alignments': np.ones((L, 1), dtype=np.int64) * aln.shape[0],
            'deletion_probability': np.float32(aln=='-').mean(0).reshape(-1,1),
            'gap_matrix': gap_matrix,
            'non_gapped_profile': non_gapped_profile,
            # plmDCA
            'pseudo_frob': pseudo_frob,
            'pseudo_bias': pseudo_bias,
            'pseudolikelihood': pseudolikelihood,
            'num_effective_alignments': np.float32(Meff),
            'mutual_information': MI,
            # no need features for prediction
            'resolution': np.float32(0),
            'sec_structure': np.zeros((L, 8), dtype=np.int64),
            'sec_structure_mask': np.zeros((L, 1), dtype=np.int64),
            'solv_surf': np.zeros((L, 1), dtype=np.float32),
            'solv_surf_mask': np.zeros((L, 1), dtype=np.int64),
            'alpha_positions': np.zeros((L, 3), dtype=np.float32),
            'alpha_mask': np.zeros((L, 1), dtype=np.int64),
            'beta_positions': np.zeros((L, 3), dtype=np.float32),
            'beta_mask': np.zeros((L, 1), dtype=np.int64),
            'superfamily': '',
            'between_segment_residues': np.zeros((L, 1), dtype=np.int64),
            'phi_angles': np.zeros((L, 1), dtype=np.float32),
            'phi_mask': np.zeros((L, 1), dtype=np.int64),
            'psi_angles': np.zeros((L, 1), dtype=np.float32),
            'psi_mask': np.zeros((L, 1), dtype=np.int64),
            # to be fixed soon
            'profile': np.zeros((L, 21), dtype=np.float32),
            'profile_with_prior': np.zeros((L, 22), dtype=np.float32),
            'profile_with_prior_without_gaps': np.zeros((L, 21), dtype=np.float32)
        }
        dataset.append(data)
    
    np.save(out_file, dataset, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alphafold - PyTorch version')
    parser.add_argument('-s', '--seq', type=str, required=True, help='target protein fasta file')
    parser.add_argument('-o', '--out', type=str, default=None, help='output file')
    parser.add_argument('-c', '--crop', default=False, action='store_true', help='make crops')
    parser.add_argument('-f', '--feature', default=False, action='store_true', help='make features')
    args = parser.parse_args()

    SEQ_FILE = Path(args.seq)
    if args.crop:
        make_crops(SEQ_FILE)
    elif args.feature:
        OUT_FILE = args.out if args.out is not None else SEQ_FILE.parent / SEQ_FILE.stem
        feature_generation(SEQ_FILE, OUT_FILE)

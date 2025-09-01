import torch
import torch.nn.functional as F
import numpy as np


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def evaluation(data, config, probe_idx=None, top_k=5):
    """
    If probe_idx is None:
        run full multi-view, multi-seq rank-k evaluation as before.
    Else:
        compute the rank of the single probe sample at probe_idx
        against a gallery of all other samples.
    """
    # Unpack inputs
    feature, view, seq_type, label, path = data
    print(f'feature length {len(feature)}')
    path = np.array(path).reshape(-1)
    feature = np.asarray(feature)
    label   = np.asarray(label)
    sample_num = feature.shape[0]
    print(len(feature))
    print(f'sample_num = {sample_num}')
    # Quick helper: full original evaluation
    def full_eval():
        dataset = config['dataset'].split('-')[0]
        probe_seq_dict = {
            'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
            'OUMVLP': [['00']]
        }
        gallery_seq_dict = {
            'CASIA': [['nm-01','nm-02','nm-03','nm-04']],
            'OUMVLP': [['01']]
        }
        view_list = sorted(set(view))
        num_rank = top_k
        acc = np.zeros([
            len(probe_seq_dict[dataset]),
            len(view_list),
            len(view_list),
            num_rank
        ])
        for p, probe_seq in enumerate(probe_seq_dict[dataset]):
            for gallery_seq in gallery_seq_dict[dataset]:
                for v1, probe_view in enumerate(view_list):
                    for v2, gallery_view in enumerate(view_list):
                        gmask = np.isin(seq_type, gallery_seq) & (view == gallery_view)
                        pmask = np.isin(seq_type, probe_seq)  & (view == probe_view)
                        gallery_x = feature[gmask]
                        gallery_y = label[gmask]
                        gallery_path = path[gmask]
                        print(f'gmask :{gmask} and gallery_path:{gallery_path}')
                        
                        probe_x   = feature[pmask]
                        probe_y   = label[pmask]
                        probe_path = path[pmask]
                        print(f'pmask :{pmask} and probe_path:{probe_path}')

                        # Compute distances on GPU
                        gx = torch.from_numpy(gallery_x).float().cuda()
                        px = torch.from_numpy(probe_x).float().cuda()
                        dist = cuda_dist(px, gx)                              # [P, G]
                        idx  = dist.sort(dim=1)[1].cpu().numpy()              # [P, G]

                        # rank-k accuracy
                        hits = (probe_y[:,None] == gallery_y[idx[:, :num_rank]])
                        cum  = np.cumsum(hits, axis=1) > 0
                        acc[p, v1, v2, :] = np.round(
                            cum.sum(axis=0) * 100 / probe_x.shape[0], 2
                        )
        return acc

    # If no probe_idx given, do the full evaluation
    if probe_idx is None:
        print(f' its null ')
        return full_eval()
    elif probe_idx < 0:
        for i in range(-(probe_idx -1)):
            px = feature[i:i+1]
            py = label[i]
            other_probes = np.arange(sample_num) != i
            gx = feature[other_probes]
            list_of_feature_distances = cuda_dist(px, gx)
            sorted_idx = list_of_feature_distances.sort(dim=1)[1].cpu().numpy().ravel()                # [N-1]
            index_of_first_predict = sorted_idx[0]
            predicted_label = label[index_of_first_predict]
            if (py != predicted_label):
                print(f'py = {py}, predicted_label = {predicted_label}, correct = {py == predicted_label}')
    # ---- Per-index rank-k computation ----
    # Build probe sample
    px = feature[probe_idx:probe_idx+1]  # [1, D]
    py = label[probe_idx]


    # Build gallery (exclude the probe itself)
    mask = np.arange(sample_num) != probe_idx
    gx = feature[mask]               # [N-1, D]
    gy = label[mask]

    # Compute distances and sort
    dist = cuda_dist(px, gx)                                              # [1, N-1]
    sorted_idx = dist.sort(dim=1)[1].cpu().numpy().ravel()                # [N-1]


    # Find the 1-based rank of the first correct match
    matches = np.where(gy[sorted_idx] == py)[0]
    print(f'py = {py}')
    print(f'sample_num = {sample_num}')
    print('sorted_idx = ', {int(sorted_idx[i]): (view[sorted_idx[i]],seq_type[sorted_idx[i]],str(gy[sorted_idx][i])) for i in range(len(sorted_idx))})
    print(f'matches = {matches}')

    if matches.size == 0:
        # no gallery sample matches label (rare if you excluded all of that ID)
        rank = None
    else:
        rank = int(matches[0]) + 1

    # Top-k gallery indices & their labels
    topk = sorted_idx[:top_k]
    topk_labels = gy[topk]

    return {
        'probe_idx': probe_idx,
        'probe_label': int(py),
        'rank': rank,
        'in_topk': (rank is not None and rank <= top_k),
        'topk_indices': topk.tolist(),
        'topk_labels': topk_labels.tolist()
    }

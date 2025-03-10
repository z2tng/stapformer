import copy
import pickle
import numpy as np
import torch
from torch.autograd import Variable


def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    Flip data horizontally
    Args:
        data (torch.Tensor): data to be flipped
        left_joints (list): left joints indices
        right_joints (list): right joints indices

    Returns:
        torch.Tensor: flipped data
    """
    if isinstance(data, torch.Tensor):
        data_flip = data.clone()
    else:
        data_flip = copy.deepcopy(data)
    data_flip[..., 0] *= -1
    data_flip[..., left_joints + right_joints, :] = data_flip[..., right_joints + left_joints, :]
    return data_flip


def resample(ori_len, target_len, replay=False, randomness=True):
    """Refer https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68"""
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len - target_len)
            return range(st, st + target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel * low + (1 - sel) * high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape) * interval + even
            result = np.clip(result, a_min=0, a_max=ori_len - 1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


def split_clips(vid_list, n_frames, data_stride):
    """Refer https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L91"""
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i < len(vid_list):
        i += 1
        if i - st == n_frames:
            result.append(range(st, i))
            saved.add(vid_list[i - 1])
            st = st + data_stride
            n_clips += 1
        if i == len(vid_list):
            break
        if vid_list[i] != vid_list[i - 1]:
            if not (vid_list[i - 1] in saved):
                resampled = resample(i - st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i - 1])
            st = i
    return result


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def denormalize(pred, seq):
    out = pred.detach().cpu().numpy()
    for idx in range(out.shape[0]):
        if seq[idx] in ['TS5', 'TS6']:
            res_w, res_h = 1920, 1080
        else:
            res_w, res_h = 2048, 2048
        out[idx, :, :, :2] = (out[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        out[idx, :, :, 2:] = out[idx, :, :, 2:] * res_w / 2
    out = out - out[..., 0:1, :]
    return torch.tensor(out).to(device=pred.device)


def read_pkl(data_url):
    file = open(data_url, 'rb')
    content = pickle.load(file)
    file.close()
    return content

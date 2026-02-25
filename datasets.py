import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
import re
from torchvision import transforms
from config import rgb_dirs, pose_dirs
from normalization import (local_keypoint_normalization, global_keypoint_normalization)


def all_same(keypoints):
    return np.sum(keypoints == keypoints[0, 0]) == keypoints.size


def sign_space_normalization(raw_keypoints, missing_values=None, layout='default'):
    local_landmarks = {}
    global_landmarks = {}
    kp_normalization = ('global-body', 'local-right', 'local-left', 'local-face_all')
    part_order = [i.removeprefix('local-').removeprefix('global-') for i in kp_normalization]
    part_order = {k: v for v, k in enumerate(part_order)}

    for idx, landmarks in enumerate(kp_normalization):
        prefix, landmarks = landmarks.split("-")
        if prefix == "local":
            local_landmarks[idx] = landmarks
        elif prefix == "global":
            global_landmarks[idx] = landmarks

    # local normalization
    for idx, landmarks in local_landmarks.items():
        normalized_keypoints = local_keypoint_normalization(raw_keypoints, landmarks, padding=0.2)
        local_landmarks[idx] = normalized_keypoints

    # global normalization
    additional_landmarks = list(global_landmarks.values())
    if "body" in additional_landmarks:
        additional_landmarks.remove("body")

    if layout == 'default':
        l_shoulder_idx, r_shoulder_idx = 11, 12
    else:
        l_shoulder_idx, r_shoulder_idx = 3, 4
    keypoints, additional_keypoints = global_keypoint_normalization(
        raw_keypoints,
        "body",
        additional_landmarks,
        l_shoulder_idx=l_shoulder_idx,
        r_shoulder_idx=r_shoulder_idx,
    )

    for k, landmark in global_landmarks.items():
        if landmark == "body":
            global_landmarks[k] = keypoints
        else:
            global_landmarks[k] = additional_keypoints[landmark]

    all_landmarks = {**local_landmarks, **global_landmarks}
    all_landmarks_per_part = {k: all_landmarks[v] for k, v in part_order.items()}

    if missing_values is not None:
        for part, data in all_landmarks_per_part.items():
            for fidx in range(len(data)):
                if not all_same(data[fidx]):
                    continue
                all_landmarks_per_part[part][fidx] = np.zeros_like(data[fidx]) + missing_values

    return all_landmarks_per_part


def load_part_kp_YTASL(skeletons, confs, normalization, layout):
    thr = 0.3
    # kps_with_scores = {}
    kps_all_parts = {}
    confs_all_parts = {}
    scale = None

    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        for i, (skeleton, conf) in enumerate(zip(skeletons, confs)):

            if part == 'body':
                if layout == 'default':
                    hand_kp2d = np.stack(skeleton['pose_landmarks'][:25])
                    confidence = np.stack(conf['pose_landmarks'][:25])
                elif layout == 'pruned':
                    pose_landmarks = [0, 7, 8, 11, 12, 13, 14, 15, 16]
                    hand_kp2d = np.stack([skeleton['pose_landmarks'][i] for i in pose_landmarks])
                    confidence = np.stack([conf['pose_landmarks'][i] for i in pose_landmarks])
                elif layout == 'isharah':
                    pose_landmarks = [0, 7, 8, 11, 12, 13, 14, 15, 16]
                    hand_kp2d = np.stack([skeleton['pose_landmarks'][i] for i in pose_landmarks])
                    confidence = np.stack([conf['pose_landmarks'][i] for i in pose_landmarks])

            elif part == 'left':
                if layout in ['default', 'pruned', 'isharah']:
                    hand_kp2d = np.stack(skeleton['left_hand_landmarks'])
                    confidence = np.stack(conf['left_hand_landmarks'])

            elif part == 'right':
                if layout in ['default', 'pruned', 'isharah']:
                    hand_kp2d = np.stack(skeleton['right_hand_landmarks'])
                    confidence = np.stack(conf['right_hand_landmarks'])

            elif part == 'face_all':
                if layout == 'default':
                    face_landmarks = [
                        0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
                        93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
                        282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
                    ]
                    hand_kp2d = np.stack([skeleton['face_landmarks'][i] for i in face_landmarks])
                    confidence = np.stack([conf['face_landmarks'][i] for i in face_landmarks])
                elif layout == 'pruned':
                    face_landmarks = [4, 13, 14, 61, 81, 93, 152, 159, 172, 178, 291, 311, 323, 386, 397, 402, 472, 477]
                    hand_kp2d = np.stack([skeleton['face_landmarks'][i] for i in face_landmarks])
                    confidence = np.stack([conf['face_landmarks'][i] for i in face_landmarks])
                elif layout == 'isharah':
                    face_landmarks = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321, 375, 405]
                    hand_kp2d = np.stack([skeleton['face_landmarks'][i] for i in face_landmarks])
                    confidence = np.stack([conf['face_landmarks'][i] for i in face_landmarks])

            else:
                raise NotImplementedError
            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        kps_all_parts[part] = kps
        confs_all_parts[part] = confidences[..., None]

    if normalization == 'signspace':
        normalized_kps = sign_space_normalization(kps_all_parts.copy(), layout=layout)
    else:
        normalized_kps = kps_all_parts

    kps_with_scores = {}
    for part in normalized_kps.keys():
        kps_with_scores[part] = np.concatenate([normalized_kps[part], confs_all_parts[part]], axis=-1)

    kps_with_scores = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in kps_with_scores.items()}
    return kps_with_scores


def load_part_kp_Isharah(skeletons, confs, normalization, layout):
    # kps_with_scores = {}
    kps_all_parts = {}
    confs_all_parts = {}

    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        for i, (skeleton, conf) in enumerate(zip(skeletons, confs)):

            if part == 'body':
                hand_kp2d = np.stack(skeleton['pose_landmarks'])
                confidence = np.stack(conf['pose_landmarks'])

            elif part == 'left':
                hand_kp2d = np.stack(skeleton['left_hand_landmarks'])
                confidence = np.stack(conf['left_hand_landmarks'])

            elif part == 'right':
                hand_kp2d = np.stack(skeleton['right_hand_landmarks'])
                confidence = np.stack(conf['right_hand_landmarks'])

            elif part == 'face_all':
                hand_kp2d = np.stack(skeleton['face_landmarks'])
                confidence = np.stack(conf['face_landmarks'])

            else:
                raise NotImplementedError
            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        kps_all_parts[part] = kps
        confs_all_parts[part] = confidences[..., None]

    if normalization == 'signspace':
        normalized_kps = sign_space_normalization(kps_all_parts.copy(), layout=layout)
    else:
        normalized_kps = kps_all_parts

    kps_with_scores = {}
    for part in normalized_kps.keys():
        kps_with_scores[part] = np.concatenate([normalized_kps[part], confs_all_parts[part]], axis=-1)

    kps_with_scores = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in kps_with_scores.items()}
    return kps_with_scores


YTASL_GROUP_SIZES = {
    'pose_landmarks': 33,
    'right_hand_landmarks': 21,
    'left_hand_landmarks': 21,
    'face_landmarks': 478,
}

YTASL_GROUP_ERROR_LABELS = {
    'pose_landmarks': 'a pose group',
    'right_hand_landmarks': 'a Rhand group',
    'left_hand_landmarks': 'a Lhand group',
    'face_landmarks': 'a face group',
}

ISHARAH_GROUP_SIZES = {
    'pose_landmarks': 25,
    'right_hand_landmarks': 21,
    'left_hand_landmarks': 21,
    'face_landmarks': 19,
}


def _fill_missing_landmarks(
    skeleton,
    conf,
    group_name,
    expected_size,
    clip_name,
    frame_idx,
    error_group_label=None,
    include_size_details=True,
    strict_key_access=False,
):
    points = skeleton[group_name] if strict_key_access else skeleton.get(group_name, [])
    if len(points) == 0:
        conf[group_name] = [0] * expected_size
        skeleton[group_name] = [[0.0, 0.0]] * expected_size
    elif len(points) != expected_size:
        group_label = error_group_label or f"group '{group_name}'"
        if include_size_details:
            raise NotImplementedError(
                f"Unexpected number of keypoints in {group_label}: {clip_name}, frame {frame_idx}, "
                f"expected {expected_size}, got {len(points)}"
            )
        raise NotImplementedError(f"Unexpected number of keypoints in {group_label}: {clip_name}, {frame_idx}")
    else:
        conf[group_name] = [1] * expected_size


# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None

    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []

        for skeleton, conf in zip(skeletons, confs):
            if skeleton.ndim == 4:  # if (1,133,2) - wrapped in list
                skeleton = skeleton[0]
                conf = conf[0]

            if part == 'body':  # [0, 3, 4, 5, 6, 7, 8, 9, 10]
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':  # [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':  # [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':  # [23, 25, 27, 29, 31, 33, 35, 37, 39, 83, 84, 85, 86, 87, 88, 89, 90, 53]
                hand_kp2d = skeleton[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23, 23 + 17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError

            kps.append(hand_kp2d)
            confidences.append(confidence)

        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)

        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[..., 2] <= thr] = 0

        kps_with_scores[part] = torch.tensor(result)

    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


# bbox of hands
def bbox_4hands(left_keypoints, right_keypoints, hw):
    # keypoints --> T,21,2
    # keypoints --> T,21,2

    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)

        return (max_x + min_x) / 2, (max_y + min_y) / 2, (max_x - min_x), (max_y - min_y)

    H, W = hw

    if left_keypoints is None:
        left_keypoints = np.zeros([1, 21, 2])

    if right_keypoints is None:
        right_keypoints = np.zeros([1, 21, 2])
    # [T, 21, 2]
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W * left_mean_x
    left_mean_y = H * left_mean_y

    left_diff_x = W * left_diff_x
    left_diff_y = H * left_diff_y

    left_diff_x = max(left_diff_x)
    left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x, left_diff_y)

    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W * right_mean_x
    right_mean_y = H * right_mean_y

    right_diff_x = W * right_diff_x
    right_diff_y = H * right_diff_y

    right_diff_x = max(right_diff_x)
    right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x, right_diff_y)

    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)

    left_new_box = np.stack([left_mean_x - box_hw / 2, left_mean_y - box_hw / 2, left_mean_x + box_hw / 2,
                             left_mean_y + box_hw / 2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw / 2, right_mean_y - box_hw / 2, right_mean_x + box_hw / 2,
                              right_mean_y + box_hw / 2]).astype(np.int16)

    return left_new_box.transpose(1, 0), right_new_box.transpose(1, 0), box_hw


def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform):
    support_rgb_dict = {}

    confs = np.array(confs)
    skeletons = np.array(skeletons)

    # sample index of low scores
    left_confs_filter = confs[:, 0, 91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
    else:

        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:, 0, [95, 99, 103, 107, 111]].min(-1)

        left_weights = np.max(left_confs) - left_confs + 1e-5
        left_probabilities = left_weights / np.sum(left_weights)

        left_sample_size = int(np.ceil(0.1 * len(left_confs_filter_indices)))

        left_sampled_indices = np.random.choice(left_confs_filter_indices.tolist(),
                                                size=left_sample_size,
                                                replace=False,
                                                p=left_probabilities)
        # left_sampled_indices: values: 0-255(0,max_len)
        # tmp: values: 0-(end-start)
        left_sampled_indices = np.sort(left_sampled_indices)

        left_skeletons = skeletons[left_sampled_indices, 0, 91:112]

    right_confs_filter = confs[:, 0, 112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None

    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:, 0, [95 + 21, 99 + 21, 103 + 21, 107 + 21, 111 + 21]].min(-1)

        right_weights = np.max(right_confs) - right_confs + 1e-5
        right_probabilities = right_weights / np.sum(right_weights)

        right_sample_size = int(np.ceil(0.1 * len(right_confs_filter_indices)))

        right_sampled_indices = np.random.choice(right_confs_filter_indices.tolist(),
                                                 size=right_sample_size,
                                                 replace=False,
                                                 p=right_probabilities)
        right_sampled_indices = np.sort(right_sampled_indices)

        right_skeletons = skeletons[right_sampled_indices, 0, 112:133]

    image_size = 112
    all_indices = []
    if not left_sampled_indices is None:
        all_indices.append(left_sampled_indices)
    if not right_sampled_indices is None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)

        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = tmp[sampled_indices]

    # load image sample
    imgs = load_video_support_rgb(full_path, sampled_indices_real)

    # get hand bbox
    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons,
                                                      right_skeletons,
                                                      imgs[0].shape[:2])

    # crop left and right hand
    image_size = 112
    if box_hw == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)

        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    factor = image_size / box_hw

    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)

    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)

        left_skeletons_norm = left_skeletons * imgs[0].shape[:2][::-1] - left_new_box[:, None, [0, 1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0, 1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)

    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)

        right_skeletons_norm = right_skeletons * imgs[0].shape[:2][::-1] - right_new_box[:, None, [0, 1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0, 1)
    left_idx = 0
    right_idx = 0

    for idx, img in enumerate(imgs):
        mapping_idx = sampled_indices[idx]
        if not left_sampled_indices is None and left_idx < len(left_sampled_indices) and mapping_idx == \
                left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]

            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3], box[0]:box[2], :]
            img_draw = np.pad(img_draw,
                              ((0, max(0, box_hw - img_draw.shape[0])), (0, max(0, box_hw - img_draw.shape[1])),
                               (0, 0)), mode='constant', constant_values=0)

            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1

        if not right_sampled_indices is None and right_idx < len(right_sampled_indices) and mapping_idx == \
                right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]

            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3], box[0]:box[2], :]
            img_draw = np.pad(img_draw,
                              ((0, max(0, box_hw - img_draw.shape[0])), (0, max(0, box_hw - img_draw.shape[1])),
                               (0, 0)), mode='constant', constant_values=0)

            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1

    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])

    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    # get index, images and keypoints priors
    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices)
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)

    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)

    return support_rgb_dict


# use split rgb video for save time
def load_video_support_rgb(path, tmp):
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))

    vr.seek(0)
    buffer = vr.get_batch(tmp).asnumpy()
    batch_image = buffer
    del vr

    return batch_image


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_valid_metric_label(text):
    if text is None:
        return False
    text = " ".join(str(text).split()).strip()
    if not text:
        return False
    # Require at least one alnum/letter token after punctuation/symbol stripping.
    return re.search(r"\w", text, flags=re.UNICODE) is not None

def select_frame_indices(duration, max_length, phase):
    if duration <= max_length:
        return list(range(duration))
    if phase == 'train':
        return sorted(random.sample(range(duration), k=max_length))
    # Deterministic, near-uniform coverage for dev/test.
    return ((np.arange(max_length) * duration) // max_length).tolist()


# build base dataset
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch, src_length_batch, name_batch, pose_tmp, gloss_batch = [], [], [], [], []

        for name_sample, pose_sample, text, gloss, _ in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])

            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]

            img_batch = torch.stack(padded_video, 0)

            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0, batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch

        if self.rgb_support:
            support_rgb_dicts = {key: [] for key in batch[0][-1].keys()}
            for _, _, _, _, support_rgb_dict in batch:
                for key in support_rgb_dict.keys():
                    support_rgb_dicts[key].append(support_rgb_dict[key])

            for part in ['left', 'right']:
                index_key = f'{part}_sampled_indices'
                skeletons_key = f'{part}_skeletons_norm'
                rgb_key = f'{part}_hands'
                len_key = f'{part}_rgb_len'

                index_batch = torch.cat(support_rgb_dicts[index_key], 0)
                skeletons_batch = torch.cat(support_rgb_dicts[skeletons_key], 0)
                img_batch = torch.cat(support_rgb_dicts[rgb_key], 0)

                src_input[index_key] = index_batch
                src_input[skeletons_key] = skeletons_batch
                src_input[rgb_key] = img_batch
                src_input[len_key] = [len(index) for index in support_rgb_dicts[index_key]]

        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch

        return src_input, tgt_input


class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]

        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif self.args.dataset == "Isharah":
            self.pose_dir = pose_dirs[args.dataset]  # pose only
            self.rgb_dir = rgb_dirs[args.dataset]  # ""

        else:
            raise NotImplementedError(f"dataset {self.args.dataset} not supported")

        self.list = list(self.raw_data.keys())
        before = len(self.list)
        self.list = [k for k in self.list if is_valid_metric_label(self.raw_data[k].get('text', ''))]
        removed = before - len(self.list)
        if removed > 0:
            print(f"[dataset-filter] {phase}: removed {removed}/{before} samples with invalid labels")

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss = " ".join(sample['gloss'])
        else:
            gloss = ''

        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))

        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0

        tmp = select_frame_indices(duration, self.max_length, self.phase)

        tmp = np.array(tmp) + start

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp

        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = os.path.join(self.rgb_dir, path)
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'


class S2T_Dataset_YTASL(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_YTASL, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.phase = phase
        self.annotation = load_json(path)
        self.rgb_support = self.args.rgb_support
        self.normalization = self.args.normalization
        self.layout = args.layout

        self.pose_dir = pose_dirs[args.dataset]
        self.rgb_dir = rgb_dirs[args.dataset]

        self.list_data = []  # [(video_id, clip_id), ...]
        self.clip_order_to_int = {}
        self.clip_order_from_int = {}

        for video_id in self.annotation.keys():
            co = self.annotation[video_id]['clip_order']
            self.clip_order_from_int[video_id] = dict(zip(range(len(co)), co))
            self.clip_order_to_int[video_id] = dict(zip(co, range(len(co))))

        total_candidates = 0
        kept_samples = 0
        for video_id, clip_dict in self.annotation.items():
            for clip_name in clip_dict['clip_order']:
                total_candidates += 1
                translation = clip_dict[clip_name]['translation']
                has_sentence = len([" ".join(_.split()) for _ in translation.split(".") if len(_) > 0]) > 0
                if not has_sentence:
                    continue
                if not is_valid_metric_label(translation):
                    continue
                self.list_data.append((video_id, self.clip_order_to_int[video_id][clip_name]))
                kept_samples += 1
        removed = total_candidates - kept_samples
        if removed > 0:
            print(f"[dataset-filter] {phase}: removed {removed}/{total_candidates} samples with invalid labels")

        video_clips = set()
        for clip in os.listdir(self.pose_dir):
            video_id, clip_id, _ = clip.split(".")
            if video_id in self.clip_order_to_int:
                clip_full_id = f'{video_id}.{clip_id}'
                if clip_full_id in self.clip_order_to_int[video_id]:
                    video_clips.add((video_id, self.clip_order_to_int[video_id][clip_full_id]))

        self.remove_missing_annotation(video_clips)  # Remove data in annotations that are missing in h5 file

    def remove_missing_annotation(self, h5_video_clip):
        annotations_to_delete = set(self.list_data) - h5_video_clip
        for a in annotations_to_delete:
            self.list_data.remove(a)

    def __getitem__(self, index):
        video_id, clip_id = self.list_data[index]
        clip_name = self.clip_order_from_int[video_id][clip_id]

        # Get translation
        clip_dict = self.annotation[video_id][clip_name]
        text = clip_dict['translation']

        # Get the pose features
        pose_sample = self.load_pose(clip_name)

        # TODO: rgb support
        video_path = ""
        support_rgb_dict = {}

        # sample = {"name": clip_name,
        #           "video_path": video_path,
        #           "pose_features": pose_features,
        #           "text": translation}
        # Crop long sequences to desired max length. Random sample

        # skeletons = pose['keypoints']
        # confs = pose['scores']
        # skeletons_tmp = []
        # confs_tmp = []
        # for index in tmp:
        #     skeletons_tmp.append(skeletons[index])
        #     confs_tmp.append(confs[index])
        #
        # skeletons = skeletons_tmp
        # confs = confs_tmp

        # confs = [np.ones(int(pose_features.shape[1]/2)) for _ in range(pose_features.shape[0])]
        # confs = [np.ones(pose_features.shape[0])] * pose_features.shape[1]
        # skeletons = [] # List of ndarrays (133,2) - full keypoints
        # kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        name_sample = clip_name
        gloss = ''

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self, clip_name):
        path = os.path.join(self.pose_dir, f"{clip_name}.json")
        pose_data = load_json(path)
        pose = pose_data['cropped_keypoints']

        duration = len(pose)
        start = 0

        tmp = select_frame_indices(duration, self.max_length, self.phase)
        tmp = np.array(tmp) + start
        skeletons = [pose[i] for i in tmp]

        confs = []
        for i, skeleton in enumerate(skeletons):
            conf = {}
            for group_name, expected_size in YTASL_GROUP_SIZES.items():
                _fill_missing_landmarks(
                    skeleton=skeleton,
                    conf=conf,
                    group_name=group_name,
                    expected_size=expected_size,
                    clip_name=clip_name,
                    frame_idx=i,
                    error_group_label=YTASL_GROUP_ERROR_LABELS[group_name],
                    include_size_details=False,
                    strict_key_access=True,
                )

            confs.append(conf)

        kps_with_scores = load_part_kp_YTASL(skeletons, confs, self.normalization, self.layout)
        return kps_with_scores

    def __len__(self):
        return len(self.list_data)

    def __str__(self):
        return f'#total {len(self)}'


class S2T_Dataset_Isharah(S2T_Dataset_YTASL):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_Isharah, self).__init__(path=path, args=args, phase=phase)

    def load_pose(self, clip_name):
        path = os.path.join(self.pose_dir, f"{clip_name}.json")
        pose_data = load_json(path)
        pose = pose_data['cropped_keypoints']

        duration = len(pose)
        tmp = select_frame_indices(duration, self.max_length, self.phase)
        tmp = np.array(tmp)
        skeletons = [pose[i] for i in tmp]

        confs = []
        for i, skeleton in enumerate(skeletons):
            conf = {}
            for group_name, expected_size in ISHARAH_GROUP_SIZES.items():
                _fill_missing_landmarks(
                    skeleton=skeleton,
                    conf=conf,
                    group_name=group_name,
                    expected_size=expected_size,
                    clip_name=clip_name,
                    frame_idx=i,
                )
            confs.append(conf)

        kps_with_scores = load_part_kp_Isharah(skeletons, confs, self.normalization, self.layout)
        return kps_with_scores


# class S2T_Dataset_YTASL_h5(Base_Dataset):
#     def __init__(self, path, args, phase):
#         super(S2T_Dataset_YTASL_h5, self).__init__()
#         self.args = args
#         self.max_length = args.max_length
#         self.phase = phase
#         self.annotation = load_json(path)
#         self.rgb_support = self.args.rgb_support
#
#         # Load poses
#         self.list_data = []  # [(video_id, clip_id), ...]
#         self.h5_data = {}
#         self.h5shard = defaultdict(lambda: defaultdict(dict))
#         self.clip_order_to_int = {}
#         self.clip_order_from_int = {}
#
#         for video_id in self.annotation.keys():
#             co = self.annotation[video_id]['clip_order']
#             self.clip_order_from_int[video_id] = dict(zip(range(len(co)), co))
#             self.clip_order_to_int[video_id] = dict(zip(co, range(len(co))))
#
#         for video_id, clip_dict in self.annotation.items():
#             for clip_name in clip_dict:
#                 if clip_name != "clip_order":
#                     self.list_data.append((video_id, self.clip_order_to_int[video_id][clip_name]))
#
#         self.vf_path = os.path.join(pose_dirs["YTASL"], "YouTubeASL.keypoints.{}.json".format(phase))
#
#         h5_video_clip = self.read_multih5_json(self.vf_path)
#         self.remove_missing_annotation(h5_video_clip)  # Remove data in annotations that are missing in h5 file
#
#     def read_multih5_json(self, json_filename):
#         """Helper function for reading json specifications of multiple H5 files for visual features"""
#         h5_video_clip = set()
#         with open(json_filename, 'r') as F:
#             self.h5shard = json.load(F)
#         self.h5_data = {}
#         print(f"Pose {self.phase} data are loaded from: ")
#         for k in set(self.h5shard.values()):
#             h5file = json_filename.replace('metadata_', '').replace('.json', ".%s.h5" % k)
#             print("--" + h5file)  # ,k,json_filename,data_dir)
#             self.h5_data[k] = h5py.File(h5file, 'r')
#
#             for vi in self.h5_data[k].keys():
#                 for ci in self.h5_data[k][vi].keys():
#                     if vi in self.clip_order_to_int:
#                         if ci in self.clip_order_to_int[vi]:
#                             clip_id = self.clip_order_to_int[vi][ci]
#                             h5_video_clip.add((vi, clip_id))
#         return h5_video_clip
#
#     def remove_missing_annotation(self, h5_video_clip):
#         annotations_to_delete = set(self.list_data) - h5_video_clip
#         for a in annotations_to_delete:
#             self.list_data.remove(a)
#
#     def __getitem__(self, index):
#         video_id, clip_id = self.list_data[index]
#         clip_name = self.clip_order_from_int[video_id][clip_id]
#
#         # Get the pose features
#         shard = self.h5shard[video_id]
#         pose_features = torch.tensor(np.array(self.h5_data[shard][video_id][clip_name]))
#
#         # TODO: rgb support
#         video_path = ""
#
#         # Get translation
#         clip_dict = self.annotation[video_id][clip_name]
#         translation = clip_dict['translation']
#
#         # sample = {"name": clip_name,
#         #           "video_path": video_path,
#         #           "pose_features": pose_features,
#         #           "text": translation}
#         #
#         # # Crop long sequences to desired max length. Random sample
#         # duration = len(pose_features) # TODO: works?
#         # if duration > self.max_length:
#         #     tmp = sorted(random.sample(range(duration), k=self.max_length))
#         # else:
#         #     tmp = list(range(duration))
#         #
#         # tmp = np.array(tmp)
#
#         # skeletons = pose['keypoints']
#         # confs = pose['scores']
#         # skeletons_tmp = []
#         # confs_tmp = []
#         # for index in tmp:
#         #     skeletons_tmp.append(skeletons[index])
#         #     confs_tmp.append(confs[index])
#         #
#         # skeletons = skeletons_tmp
#         # confs = confs_tmp
#
#         # confs = [np.ones(int(pose_features.shape[1]/2)) for _ in range(pose_features.shape[0])]
#         # confs = [np.ones(pose_features.shape[0])] * pose_features.shape[1]
#         # skeletons = [] # List of ndarrays (133,2) - full keypoints
#         # kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)
#
#         # decoded = self.tokenizer(
#         #     translation,
#         #     max_length=self.max_token_length,
#         #     padding="max_length",
#         #     truncation=True,
#         #     return_tensors="pt",
#         # )
#         # labels = decoded.input_ids
#
#         # Skip frames for the keypoints
#         # if self.skip_frames:
#         #     if type(self.skip_frames) == bool:
#         #         for input_type in INPUT_TYPES:
#         #             if visual_features[input_type] is not None:
#         #                 visual_features[input_type] = visual_features[input_type][::2]
#         #     elif type(self.skip_frames) == int:
#         #         for input_type in INPUT_TYPES:
#         #             if visual_features[input_type] is not None:
#         #                 visual_features[input_type] = visual_features[input_type][::self.skip_frames]
#         #
#         # # Trim the keypoints to the max sequence length
#         # if self.max_sequence_length:
#         #     for input_type in INPUT_TYPES:
#         #         if visual_features[input_type] is not None:
#         #             visual_features[input_type] = visual_features[input_type][: self.max_sequence_length]
#         #             seq_len = len(visual_features[input_type])
#         #
#         # assert seq_len, "No modality provided or clip has no length!"
#         # attention_mask = torch.ones(seq_len)
#         #
#         # return {
#         #     "sign_inputs": {'pose': visual_features['pose'],
#         #                     'mae': visual_features['mae'],
#         #                     'dino': visual_features['dino'],
#         #                     'sign2vec': visual_features['sign2vec']},
#         #     "sentence": translation,
#         #     "labels": labels,
#         #     "attention_mask": attention_mask,
#         # }
#         return kps
#
#     def __len__(self):
#         return len(self.list_data)
#
#     def __str__(self):
#         return f'#total {len(self)}'


class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_news, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        before = len(self.annotation)
        self.annotation = [x for x in self.annotation if is_valid_metric_label(x.get("text", ""))]
        removed = before - len(self.annotation)
        if removed > 0:
            print(f"[dataset-filter] {phase}: removed {removed}/{before} samples with invalid labels")

        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]

        else:
            raise NotImplementedError
        sum_sample = len(self.annotation)
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, index):
        num_retries = 10

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx:self.end_idx][index]

            text = sample['text']
            name_sample = sample['video']

            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])

            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break

        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return name_sample, pose_sample, text, _, support_rgb_dict

    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        full_path = os.path.join(self.rgb_dir, rgb_name)

        duration = len(pose['scores'])

        tmp = select_frame_indices(duration, self.max_length, self.phase)

        tmp = np.array(tmp)

        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []

        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp

        kps_with_scores = load_part_kp(skeletons, confs)

        support_rgb_dict = {}
        if self.rgb_support:
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

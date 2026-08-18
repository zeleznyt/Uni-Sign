import torch
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import gzip
import pickle
from decord import VideoReader, cpu
import json
import pathlib
import re
from bisect import bisect_right
from torchvision import transforms
from data_config import spec_name, spec_pose_roots, spec_rgb_config, spec_rgb_root
from normalization import (
    local_keypoint_normalization,
    global_keypoint_normalization,
    normalize_text,
)


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
                pose_landmarks = [0, 7, 8, 11, 12, 13, 14, 15, 16]
                hand_kp2d = np.stack([skeleton['pose_landmarks'][i] for i in pose_landmarks])
                confidence = np.stack([conf['pose_landmarks'][i] for i in pose_landmarks])

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


def load_gzip_pickle(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def index_pose_jsons(pose_roots):
    pose_index = {}
    duplicate_clips = []
    root_json_counts = []
    for pose_root in pose_roots:
        if not os.path.isdir(pose_root):
            root_json_counts.append((pose_root, 0))
            continue
        json_files = [
            filename for filename in os.listdir(pose_root)
            if filename.endswith(".json")
        ]
        root_json_counts.append((pose_root, len(json_files)))
        for filename in json_files:
            clip_name = pathlib.Path(filename).stem
            if clip_name in pose_index:
                duplicate_clips.append(clip_name)
                continue
            pose_index[clip_name] = os.path.join(pose_root, filename)
    return pose_index, duplicate_clips, root_json_counts


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


# One configured dataset may contain one or more format-specific loaders.
class ConfiguredDataset(Dataset.Dataset):
    def __init__(self, specs, args, phase):
        self.args = args
        self.phase = phase
        self.specs = list(specs)
        self.loaders = [_build_loader(spec, args, phase) for spec in self.specs]
        self.names = [spec_name(spec) for spec in self.specs]
        self.weights = [float(spec.get("weight", 1.0)) for spec in self.specs]
        self.uses_dataset_weights = any("weight" in spec for spec in self.specs)
        self.rgb_support = bool(args.rgb_support)

        self.cumulative_sizes = []
        total = 0
        for loader in self.loaders:
            if bool(loader.rgb_support) != self.rgb_support:
                raise ValueError(
                    f"Loader '{loader.name}' RGB mode does not match the configured run-wide RGB mode."
                )
            total += len(loader)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        loader_index = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if loader_index == 0 else self.cumulative_sizes[loader_index - 1]
        return self.loaders[loader_index].get_sample(index - previous_size)

    def sample_weights(self):
        if not self.uses_dataset_weights:
            return None

        sample_weights = []
        for loader, dataset_weight in zip(self.loaders, self.weights):
            if len(loader) == 0:
                continue
            per_sample_weight = dataset_weight / len(loader)
            sample_weights.extend([per_sample_weight] * len(loader))
        return sample_weights

    def get_setup_summaries(self):
        return [loader.get_setup_summary() for loader in self.loaders]

    def get_sampling_summary(self):
        source_sizes = [len(loader) for loader in self.loaders]
        nominal_epoch_draws = sum(source_sizes)

        if self.uses_dataset_weights:
            total_weight = sum(
                weight for weight, source_size in zip(self.weights, source_sizes)
                if source_size > 0
            )
            shares = [
                weight / total_weight if source_size > 0 else 0.0
                for weight, source_size in zip(self.weights, source_sizes)
            ]
            mode = "dataset_weighted"
            replacement = True
        else:
            shares = [
                source_size / nominal_epoch_draws if nominal_epoch_draws > 0 else 0.0
                for source_size in source_sizes
            ]
            mode = "concatenation"
            replacement = False

        datasets = []
        for name, source_size, weight, share in zip(
            self.names, source_sizes, self.weights, shares
        ):
            datasets.append({
                "name": name,
                "source_samples": source_size,
                "configured_weight": weight if self.uses_dataset_weights else None,
                "expected_share": share,
                "expected_draws": nominal_epoch_draws * share,
            })

        return {
            "mode": mode,
            "replacement": replacement,
            "nominal_epoch_draws": nominal_epoch_draws,
            "datasets": datasets,
        }

    def __str__(self):
        parts = ", ".join(
            f"{name}: {len(loader)}" for name, loader in zip(self.names, self.loaders)
        )
        return f"#{self.phase} total {len(self)} ({parts})"

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


class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_replicas, rank, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = (len(self.weights) + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _resolve_file_from_roots(roots, relative_path):
    candidates = [os.path.join(root, relative_path) for root in roots]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def _rgb_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class _OriginalPickleLoader:
    def __init__(self, spec, args, phase):
        self.args = args
        self.name = spec_name(spec)
        self.loader = spec["loader"]
        self.annotation_path = spec["annotation_path"]
        self.pose_roots = spec_pose_roots(spec)
        self.rgb_config = spec_rgb_config(spec)
        self.rgb_root = spec_rgb_root(spec)
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length
        self.raw_data = load_gzip_pickle(self.annotation_path)
        self.phase = phase
        self.list = list(self.raw_data.keys())
        self.data_transform = _rgb_transform()

    def __len__(self):
        return len(self.list)

    def get_sample(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        text = sample['text']
        if "gloss" in sample.keys():
            gloss_value = sample['gloss']
            gloss = " ".join(gloss_value) if isinstance(gloss_value, list) else str(gloss_value)
        else:
            gloss = ''

        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self, path):
        pose_relative_path = path.replace(".mp4", ".pkl")
        pose_path = _resolve_file_from_roots(self.pose_roots, pose_relative_path)
        with open(pose_path, 'rb') as pose_file:
            pose = pickle.load(pose_file)

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
            full_path = os.path.join(self.rgb_root, path)
            support_rgb_dict = load_support_rgb_dict(tmp, skeletons, confs, full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def get_setup_summary(self):
        return {
            "name": self.name,
            "loader": self.loader,
            "phase": self.phase,
            "annotation_path": self.annotation_path,
            "pose_roots": self.pose_roots,
            "annotation_samples": len(self.raw_data),
            "usable_samples": len(self),
            "missing_pose_samples": None,
            "missing_pose_examples": [],
            "rgb": self.rgb_config,
        }


class _LocalJsonLoader:
    def __init__(self, spec, args, phase):
        self.args = args
        self.max_length = args.max_length
        self.phase = phase
        self.annotation_path = spec["annotation_path"]
        self.name = spec_name(spec)
        self.loader = spec["loader"]
        self.annotation = load_json(self.annotation_path)
        self.rgb_config = spec_rgb_config(spec)
        self.rgb_support = False
        self.normalization = self.args.normalization
        self.normalize_text = bool(
            getattr(args, "normalize_text", False)
            and self.loader == "ytasl_json"
            and phase in ("train", "dev")
        )
        self.layout = args.layout

        self.pose_roots = spec_pose_roots(spec)

        self.list_data = []

        for video_id, video_data in self.annotation.items():
            for clip_name in video_data['clip_order']:
                self.list_data.append((video_id, clip_name))

        self.annotation_clip_count = len(self.list_data)
        self.pose_index, self.duplicate_pose_clips, self.root_json_counts = index_pose_jsons(self.pose_roots)
        available_clip_names = set(self.pose_index.keys())
        available_samples = []
        missing_clip_names = []
        for video_id, clip_name in self.list_data:
            if clip_name in available_clip_names:
                available_samples.append((video_id, clip_name))
            else:
                missing_clip_names.append(clip_name)

        self.missing_clip_names = missing_clip_names
        self.list_data = available_samples

    def get_sample(self, index):
        video_id, clip_name = self.list_data[index]

        # Get translation
        clip_dict = self.annotation[video_id][clip_name]
        text = clip_dict['translation']
        if self.normalize_text:
            text = normalize_text(text)

        pose_sample = self.load_pose(clip_name)
        support_rgb_dict = {}
        name_sample = clip_name
        gloss = ''

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self, clip_name):
        path = self.pose_index[clip_name]
        pose_data = load_json(path)
        pose = pose_data['cropped_keypoints']

        duration = len(pose)
        tmp = select_frame_indices(duration, self.max_length, self.phase)
        tmp = np.array(tmp)
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

    def get_setup_summary(self):
        return {
            "name": self.name,
            "loader": self.loader,
            "phase": self.phase,
            "annotation_path": self.annotation_path,
            "pose_roots": self.pose_roots,
            "annotation_samples": self.annotation_clip_count,
            "usable_samples": len(self),
            "missing_pose_samples": len(self.missing_clip_names),
            "missing_pose_examples": self.missing_clip_names[:10],
            "duplicate_pose_clips": len(self.duplicate_pose_clips),
            "root_json_counts": self.root_json_counts,
            "rgb": self.rgb_config,
        }


class _IsharahJsonLoader(_LocalJsonLoader):

    def load_pose(self, clip_name):
        path = self.pose_index[clip_name]
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


class _CSLNewsLoader:
    def __init__(self, spec, args, phase):
        self.args = args
        self.name = spec_name(spec)
        self.loader = spec["loader"]
        self.annotation_path = spec["annotation_path"]
        self.pose_roots = spec_pose_roots(spec)
        self.rgb_config = spec_rgb_config(spec)
        self.rgb_root = spec_rgb_root(spec)
        self.rgb_support = self.args.rgb_support
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(self.annotation_path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)

        sum_sample = len(self.annotation)
        self.data_transform = _rgb_transform()

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)

    def __len__(self):
        return self.end_idx - self.start_idx

    def get_sample(self, index):
        num_retries = 10

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx + index]

            text = sample['text']
            name_sample = sample['video']

            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])

            except Exception:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break

        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return name_sample, pose_sample, text, '', support_rgb_dict

    def load_pose(self, pose_name, rgb_name):
        pose_path = _resolve_file_from_roots(self.pose_roots, pose_name)
        with open(pose_path, 'rb') as pose_file:
            pose = pickle.load(pose_file)
        full_path = os.path.join(self.rgb_root, rgb_name) if self.rgb_support else None

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

    def get_setup_summary(self):
        return {
            "name": self.name,
            "loader": self.loader,
            "phase": self.phase,
            "annotation_path": self.annotation_path,
            "pose_roots": self.pose_roots,
            "annotation_samples": len(self.annotation),
            "usable_samples": len(self),
            "missing_pose_samples": None,
            "missing_pose_examples": [],
            "rgb": self.rgb_config,
        }


def _build_loader(spec, args, phase):
    loader_name = spec["loader"]
    if loader_name == "ytasl_json":
        return _LocalJsonLoader(spec, args, phase)
    if loader_name == "isharah_json":
        return _IsharahJsonLoader(spec, args, phase)
    if loader_name == "original_pickle":
        return _OriginalPickleLoader(spec, args, phase)
    if loader_name == "csl_news":
        return _CSLNewsLoader(spec, args, phase)
    raise NotImplementedError(f"Data config loader '{loader_name}' is not implemented.")

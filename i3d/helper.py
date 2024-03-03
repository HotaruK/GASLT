import os
import numpy as np
from PIL import Image
from i3d.pytorch_i3d import InceptionI3d
import torch
from torch.autograd import Variable


def _load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert (data.max() <= 1.0)
    assert (data.min() >= -1.0)

    return data


def _load_rgb_batch(frames_dir, rgb_files, frame_indices, ):
    batch_data = np.zeros(frame_indices.shape + (224, 224, 3))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = _load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))

    return batch_data


def _forward_batch(b_data, i3d):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)  # b,c,t,h,w  # 40x3x16x224x224

    b_data = Variable(b_data.cuda(), volatile=True).float()
    b_features = i3d(b_data)

    b_features = b_features.data.cpu().numpy()[:, :, 0, 0, 0]
    return b_features


def get_data_by_video_name(i3d: InceptionI3d, video_names: [str]):
    result = []
    for video_name in video_names:
        frames_dir = os.path.join(i3d.input_dir, video_name)
        rgb_files = [i for i in os.listdir(frames_dir) if i.startswith('image')]
        rgb_files.sort()
        frame_cnt = len(rgb_files)

        # Cut frames
        frame_indices = []  # Frames to chunks
        if frame_cnt <= i3d.frequency:
            # If total frames is less than or equal to frequency, use all frames for each index
            frame_indices = [[j for j in range(frame_cnt)] for _ in range(frame_cnt)]
        else:
            span_half = i3d.frequency // 2
            for i in range(frame_cnt):
                start = max(0, i - span_half)
                end = min(frame_cnt - 1, i + span_half)
                if start == 0 and i < span_half:
                    end = min(frame_cnt - 1, end + (span_half - i))
                if end == frame_cnt - 1 and i > span_half:
                    start = max(0, start - (i + span_half - (frame_cnt - 1)))
                if end - start > i3d.frequency:
                    end -= 1
                frame_indices.append([j for j in range(start, end + 1)])

        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]
        batch_num = int(np.ceil(chunk_num / i3d.batch_size))  # Chunks to batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        full_features = [[]]

        for batch_id in range(batch_num):
            batch_data = _load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id])
            assert (batch_data.shape[-2] == 224)
            assert (batch_data.shape[-3] == 224)
            full_features[0].append(_forward_batch(batch_data, i3d))

        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)
        result.append(full_features)

    tensor_list = [torch.from_numpy(arr) for arr in result]
    max_length = max(arr.shape[1] for arr in tensor_list)
    padded_data = torch.zeros(len(tensor_list), max_length, 1024)
    for i, arr in enumerate(tensor_list):
        padded_data[i, :arr.shape[1], :] = arr

    return padded_data

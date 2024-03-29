# from PIL import Image
import cv2
import numpy as np
from torchvision import transforms, io
import torch
import os
import torch.nn.functional as F

layer_output = None

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x.float() / 255),  # for io.read_image
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def hook(module, input, output):
    global layer_output
    layer_output = output


def __process_image(mobilenet, image_path, use_cuda=True):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = io.read_image(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    if use_cuda:
        input_batch = input_batch.to('cuda')
        mobilenet.to('cuda')

    # register hook
    handle = mobilenet.features[-2].register_forward_hook(hook)

    mobilenet(input_batch)
    handle.remove()

    op = layer_output.view(1, -1)
    return op


def __process_video(video_path, mobilenet, use_cuda):
    files = os.listdir(video_path)
    v = []
    for file in files:
        if file.endswith('.png'):
            img_path = os.path.join(video_path, file)
            v.append(__process_image(mobilenet, img_path, use_cuda))

    return torch.stack(v).permute(1, 0, 2)


def pad_and_stack(v):
    max_len = max(tensor.size(1) for tensor in v)
    v_padded = [F.pad(tensor, (0, 0, 0, max_len - tensor.size(1))) for tensor in v]

    return torch.stack(v_padded)


def get_tensor_from_images(name, root_path, mobilenet, use_cuda):
    v = []
    for video_name in name:
        video_path = os.path.join(root_path, video_name)
        v.append(__process_video(video_path, mobilenet, use_cuda))

    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # batch norm
    t = pad_and_stack(v).squeeze(1)
    t_norm = (t - torch.mean(t)) / torch.std(t)
    del t, v

    t_norm = t_norm.to(device)
    sgn_mask = (t_norm != torch.zeros(t_norm.shape[2], device=device))[..., 0].unsqueeze(1)
    # sgn_mask = (t_norm != torch.zeros(t_norm.shape[2]))[..., 0].unsqueeze(1)
    sgn_mask = sgn_mask.to(device)

    return t_norm, sgn_mask

from __future__ import print_function

import argparse
import os
import torch
from torch.utils.data import DataLoader
import cv2 as cv
from dataset import get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_path', type=str, default='datasets/test/', help='input path to use')
parser.add_argument('--sub_path', type=str, default='000/', help='sub path, example 000, 011, 015, 020')
parser.add_argument('--model', type=str, default='checkpoints/model.pth', help='model file to use')
parser.add_argument('--output_path', default='results/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # input_tensor = cv.cvtColor(input_tensor, cv.COLOR_RGB2BGR)
    cv.imwrite(filename, input_tensor)


cuda = opt.cuda
root_path = opt.input_path
sub_path = opt.sub_path

test_set = get_test_set(root_path=root_path, sub_path=sub_path)
training_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

with torch.no_grad():
    model = torch.load(opt.model, map_location='cuda:0')
    if opt.cuda:
        model = model.cuda()
    model.eval()
    img_num = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        print('processing image [{}]'.format(img_num))
        input, target = batch[0], batch[1]
        if opt.cuda:
            input = input.cuda()
        out = model(input)
        output_save_path = '{}output/{}'.format(opt.output_path, sub_path)
        gt_save_path = '{}gt/{}'.format(opt.output_path, sub_path)
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)
        if not os.path.exists(gt_save_path):
            os.makedirs(gt_save_path)
        save_image_tensor2cv2(out, '{}{}.png'.format(output_save_path, img_num))
        save_image_tensor2cv2(target, '{}{}.png'.format(gt_save_path, img_num))
        img_num = img_num + 1

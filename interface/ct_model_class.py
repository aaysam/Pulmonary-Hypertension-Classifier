import os
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np

from albumentations.augmentations import geometric, transforms
from albumentations.core.composition import Compose, OneOf

import pydicom


### Our model config

CONFIG = {
    "deep_supervision": False,
    "input_channels": 3,
    "num_classes": 2,
    "input_h": 256,
    "input_w": 256,
    "device": "cpu",
    "model_path": "model.pth",
    "save_images": True,
    "images_dir": "test_images",
}

### Model API wrapper

class Model:
    def __init__(self):
        # init model
        model = NestedUNet(CONFIG["num_classes"], CONFIG["input_channels"], CONFIG["deep_supervision"])
        self.model = model.to(CONFIG["device"])
        self.model.load_state_dict(torch.load(CONFIG["model_path"], map_location=torch.device('cpu')))
        self.model.eval()
        # init preproc
        self.val_transform = Compose([
            geometric.Resize(CONFIG['input_h'], CONFIG['input_w']),
            transforms.Normalize(),
        ])

    def decode_outputs(self, output: torch.Tensor) -> np.ndarray:
        # move models outputs from gpu to cpu
        if CONFIG["device"] == "cuda":
            output = torch.sigmoid(output).detach().cpu().numpy()
        else:
            output = torch.sigmoid(output).detach().numpy()
        # decode outputs according to papers autorhs method
        outputs = []
        for i in range(len(output)):
            out_pack = []
            for c in range(CONFIG["num_classes"]):
                out_pack.append(
                    (output[i, c] * 255).astype("uint8")
                )
            outputs.append(out_pack)
        return outputs

    def process_images(self, inputs: np.ndarray):
        # preprocess inputs
        processed_inputs = self.val_transform(image=inputs)["image"]
        processed_inputs = processed_inputs.astype('float32') / 255
        processed_inputs = processed_inputs.transpose(2, 0, 1)
        # move inputs to gpu if required
        if CONFIG["device"] == "cuda":
            processed_inputs = torch.tensor(processed_inputs).to("cuda")
        # unsqueeze if required
        if len(processed_inputs.shape) == 3:
            processed_inputs = torch.tensor(processed_inputs).unsqueeze(0)
        # inference preprocessed images
        output = self.model(processed_inputs)
        confidence_aorta, confidence_arterial = output[0][0].max(), output[0][1].max()
        # create masks from images
        decoded_outputs = self.decode_outputs(output)
        # save images to dir or return mixed and raw masks
        if CONFIG["save_images"]:
            for idx, output in enumerate(decoded_outputs):
                for cls_idx in range(CONFIG["num_classes"]):
                    label_class = 'pulmonary_arterial'
                    if cls_idx == 0:
                        label_class = 'aorta'
                    cv2.imwrite(os.path.join(CONFIG["images_dir"], f"{label_class}_{idx}.png"), output[cls_idx])
                return (output[0], confidence_aorta), (output[1], confidence_arterial)
        else:
            return decoded_outputs, self.mix_masks(inputs, decoded_outputs)
    
    def mix_masks(self, pre_preprocessing_inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:        
        for jdx in range(CONFIG["num_classes"]):
            mask = outputs[0][jdx]
            mask //= 2
            pre_preprocessing_inputs[:, :, jdx] += np.repeat(mask, 2, axis=1).repeat(2, axis=0)
        return pre_preprocessing_inputs

### Model definition

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    return parser.parse_args()


def adapt_dicom_format(img):
    img = np.array(img, dtype = float) 
    img = (img - img.min()) / (img.max() - img.min()) * 255.0  
    img = img.astype(np.uint8)
    img = np.expand_dims(img, 2)
    img = np.dstack((img, img[:, :, 0], img[:, :, 0]))
    
    return img


def take_aorta_diameter(image):
    area_in_pixels = 0
    for i in range(len(image[0])):
        for j in range(len(image[0][0])):
            if image[0][i][j] > 50:
                area_in_pixels += 1
    diameter = math.sqrt(area_in_pixels/math.pi)

    return diameter
# diameter = take_aorta_diameter(output[cls_idx])

    
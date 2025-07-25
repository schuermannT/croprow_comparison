# coding: utf-8
"""
SegCropNet model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.UNet import UNet_Encoder, UNet_Decoder
from .backbone.ENet import ENet_Encoder, ENet_Decoder

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SegCropNet(nn.Module):
    def __init__(self, in_ch = 3, arch="ENet"):
        super(SegCropNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 3  # if you want to output RGB instance map, it should be 3.
        print("Use {} as backbone".format(arch))
        self._arch = arch
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = UNet_Decoder(2)
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        else:
            raise("Please select right model.")

        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch == 'ENet':
            c = self._encoder(input_tensor)
            binary = self._decoder_binary(c)
            instance = self._decoder_instance(c)
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2)
            instance = self._decoder_instance(c1, c2)
        else:
            raise("Please select right model.")

        # Improvement idea -> lowers the threshold for positive probabilities. Binary lines are longer but can grow together at the top
        # probs = F.softmax(binary, dim=1)
        # positive_probs = probs[:, 1:2, :, :]
        # binary_seg_ret = (positive_probs > 0.3).float()

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)   #original Code

        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }

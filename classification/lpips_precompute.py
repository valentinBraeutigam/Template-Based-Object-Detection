
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import cv2
import lpips

'''
    class that inherits from the LPIPS class and provides functionality to pre-compute the features of one images and compare pre-computed features from two images.
'''
class LPIPS_pre(lpips.LPIPS):
    def normalize_image_cv2(self, in_img):
        assert not in_img is None and len(in_img) > 0

        img = cv2.resize(in_img, (224, 224), 
                interpolation = cv2.INTER_LINEAR)
        
        img = np.expand_dims(2* (img / 255) - 1.0, axis=0)

        # cast to torch.float32 to match the bias
        img = torch.tensor(img, dtype=torch.float32).permute((0,3,1,2)) # torch.zeros(1,3,224,224) # image should be RGB, IMPORTANT: normalized to [-1,1]
        return img
    
    def precompute_feats(self, in_img, normalize=True):
        feats = {}
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in_img = self.normalize_image_cv2(in_img)
        in0_input = self.scaling_layer(in_img) if self.version=='0.1' else in_img
        outs0 = self.net.forward(in0_input)
        for kk in range(self.L):
            feats[kk] = lpips.normalize_tensor(outs0[kk])
        return feats
        
    def forward(self, in0, in1, retPerLayer=False, normalize=False, precomputed_feats_in0=None, precomputed_feats_in1=None):
        feats0, feats1, diffs = {}, {}, {}
        if precomputed_feats_in0 is None:
            if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
                in0 = 2 * in0  - 1
            in0_input = self.scaling_layer(in0) if self.version=='0.1' else in0
            outs0 = self.net.forward(in0_input)
            for kk in range(self.L):
                feats0[kk] = lpips.normalize_tensor(outs0[kk])
        else:
            feats0 = precomputed_feats_in0
        
        if precomputed_feats_in1 is None:
            if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
                in1 = 2 * in1  - 1
            in1_input = self.scaling_layer(in1) if self.version=='0.1' else (in0, in1)
            outs1 = self.net.forward(in1_input)
            for kk in range(self.L):
                feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
        else:
            feats1 = precomputed_feats_in1
            
        for kk in range(self.L):            
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [lpips.upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [lpips.spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [lpips.upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [lpips.spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import mxnet as mx
from mxnet.contrib import autograd

class PS_DFROI_OFFSET(nn.Conv2d):
    def __init__(self, stride, p_hieght, p_width, o_dim = 256, pooled_height = 7, pooled_width = 7):
        super(PS_DFROI_OFFSET, self).__init__()
        self.stride = stride
        self.p_hiehgt = p_hieght
        self.p_width = p_width
        self.o_dim = o_dim
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.roiP = [None for i in range(self.stride)]
        self.feat_idx = [None for i in range(self.num_strides)]
        
    def forward(self, x_in, y_out):
        rois = np.array(x_in[-1])
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224)), 0, len(self.feat_strides) - 1)
        
        pyramid_idx = []
        rois_p = [None for i in range(self.strides)]
        for i in range(self.strides):
            self.feat_idx[i] = np.where(feat_id == i)[0]
            if len(self.feat_idx[i]) == 0:
                # padding dummy roi
                rois_p[i] = np.zeros((1, 6))
                pyramid_idx.append(-1)
            else:
                rois_p[i] = rois[self.feat_idx[i]]
                pyramid_idx.append(self.feat_idx[i])

        for i in range(self.num_strides):
            self.in_grad_hist_list.append(mx.nd.zeros_like(x_in[i]))

        for i in range(self.num_strides, self.num_strides * 3):
            self.in_grad_hist_list.append(mx.nd.zeros_like(x_in[i]))
        autograd.mark_variables([x_in[i] for i in range(self.num_strides * 3)], self.in_grad_hist_list)

        with autograd.train_section():
            for i in range(self.strides):
                roi_offset_t = mx.contrib.nd.DeformablePSROIPooling(data=x_in[i], rois=mx.nd.array(rois_p[i], x_in[i].context), group_size=1, pooled_size=7,
                                                                    sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=1.0 / self.feat_strides[i])
                roi_offset = mx.nd.FullyConnected(data=roi_offset_t, num_hidden=7 * 7 * 2, weight=x_in[i * 2 + self.strides], bias=x_in[i * 2 + 1 + self.num_strides])
                roi_offset_reshape = mx.nd.reshape(data=roi_offset, shape=(-1, 2, 7, 7))
                self.roi_pool[i] = mx.contrib.nd.DeformablePSROIPooling(data=x_in[i], rois=mx.nd.array(rois_p[i], x_in[i].context), trans=roi_offset_reshape,
                                                                        group_size=1, pooled_size=7, sample_per_part=4, no_trans=False, part_size=7,
                                                                        output_dim=self.output_dim, spatial_scale=1.0 / self.feat_strides[i], trans_std=0.1)
        
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)

        with autograd.train_section():
            for i in range(self.num_strides):
                if len(self.feat_idx[i] > 0):
                    autograd.compute_gradient([mx.nd.take(out_grad[0], mx.nd.array(self.feat_idx[i], out_grad[0].context)) * self.roi_pool[i]])

        if self.with_deformable:
            for i in range(0, self.num_strides * 3):
                self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])
        else:
            for i in range(0, self.num_strides):
                self.assign(in_grad[i], req[i], self.in_grad_hist_list[i])

        
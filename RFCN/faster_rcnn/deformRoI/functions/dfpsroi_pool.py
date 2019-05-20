import torch
from torch.autograd import Function
from .._ext import PS_DFROI_OFFSET 


class DFPSRoIPoolingFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)
        self.output = None
        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois, x_in, y_out, st):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, self.output_dim, self.pooled_height, self.pooled_width)
        mappingchannel = torch.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        output = output.cuda()
        mappingchannel = mappingchannel.cuda()
        PS_DFROI_OFFSET.forward(st, self.pooled_height, self.pooled_width, x_in, y_out)
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = rois
        self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()

        return grad_input, None
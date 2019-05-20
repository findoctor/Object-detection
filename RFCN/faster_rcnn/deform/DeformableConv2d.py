import torch
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# some help functions


def check_bounds(x, y, h, w):
    if x >= 0 and x < h and y >= 0 and y < w:
        return True
    else:
        return False


def build_grid(nc, h, w, dtype):
    grid = np.meshgrid(range(h), range(w), indexing='ij')  # (2,h,w)
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)  # (h*w, 2)

    grid = np.expand_dims(grid, 0)
    grid = np.tile(grid, [nc, 1, 1])  # (n*c, h*w, 2)
    grid = torch.from_numpy(grid).type(dtype)

    return Variable(grid, requires_grad=False)


def map_offset(x, offset):
    '''
    x: (n*c, h, w)
    offset: (n*c, h, w, 2)
    Return x_offset: (n*c, h*w, 2)
    '''
    nc = x.size()[0]
    h = x.size()[1]
    w = x.size()[2]

    grid = build_grid(nc, h, w, offset.data.type())  # (n*c, h*w, 2)
    offset = offset.view(nc, -1, 2)  # (n*c, h*w, 2)
    x_coord = grid + offset
    #print('new x_offset is' + str(x_coord))
    x_vals = map_vals(x, x_coord)

    return x_vals

'''
****************  old version with 3 for loop, very slow  ********************
def map_vals(x, x_coord):
    #x        (n*c, h, w)
    #x_coord  (n*c, h*w, 2)
    #Return x_vals: (n*c, h, w)
    
    nc = x.size()[0]
    h = x.size()[1]
    w = x.size()[2]
    x_vals = torch.randn(nc, h, w)
    for i in range(nc):
        for j in range(h):
            for k in range(w):
                newx = x_coord[i][j * h + k][0].item()
                newy = x_coord[i][j * h + k][1].item()
                x1 = math.floor(newx)  # top-left
                y1 = math.ceil(newy)
                x2 = math.ceil(newx)  # top-right
                y2 = math.ceil(newy)
                x3 = math.floor(newx)  # down-left
                y3 = math.floor(newy)
                x4 = math.ceil(newx)  # down-right
                y4 = math.floor(newy)
                if check_bounds(x1, y1, h, w):
                    pix1 = x[i][x1][y1]
                else:
                    pix1 = 0
                if check_bounds(x2, y2, h, w):
                    pix2 = x[i][x2][y2]
                else:
                    pix2 = 0
                if check_bounds(x3, y3, h, w):
                    pix3 = x[i][x3][y3]
                else:
                    pix3 = 0
                if check_bounds(x4, y4, h, w):
                    pix4 = x[i][x4][y4]
                else:
                    pix4 = 0
                tmp1 = (1 - abs(x1 - newx)) * pix1 + (1 - abs(x2 - newx)) * pix2
                tmp2 = (1 - abs(x3 - newx)) * pix3 + (1 - abs(x4 - newx)) * pix4
                res = (1 - abs(y1 - newy)) * tmp1 + (1 - abs(y4 - newy)) * tmp2
                x_vals[i][j][k] = res
    return x_vals
'''

# New version of mapping offsets to input map
def map_vals(input, coords):
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)
    
    n_coords = coords.size(1)
    
    # coords = torch.clamp(coords, 0, input_size - 1)
    
    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)
    
    assert (coords.size(1) == n_coords)
    
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
                               idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
                               ], 1)
                               inds = indices[:, 0] * input.size(1) * input.size(2) + indices[:, 1] * input.size(2) + indices[:, 2]
                               vals = th_flatten(input).index_select(0, inds)
                               vals = vals.view(batch_size, n_coords)
                               return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach())
    vals_rb = _get_vals_by_coords(input, coords_rb.detach())
    vals_lb = _get_vals_by_coords(input, coords_lb.detach())
    vals_rt = _get_vals_by_coords(input, coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0] * (vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0] * (vals_rb - vals_lb) + vals_lb
    mapped_vals = coords_offset_lt[..., 1] * (vals_b - vals_t) + vals_t
    return mapped_vals


class ConvOffset2d(nn.Conv2d):
    def __init__(self, in_channels):
        super(ConvOffset2d, self).__init__(in_channels, 2 * in_channels, 3, padding=1)

    def forward(self, x):
        n = x.size()[0]
        c = x.size()[1]
        h = x.size()[2]
        w = x.size()[3]
        offset = super(ConvOffset2d, self).forward(x)  # (n,2c,h,w)
        offset = offset.view(-1, offset.size()[2], offset.size()[3], 2)  # (n*c, h, w, 2)
        #print('offset is' + str(offset))
        x = x.view(-1, x.size()[2], x.size()[3])  # (n*c, h, w)

        x_vals = map_offset(x, offset)  # (n*c, h, w)
        x_vals = x_vals.contiguous().view(-1, c, h, w)  # contiguous will make a copy; View() will directly change it!

        return x_vals
import torch, torch_sht
from functools import partial

import config
from model import sht_func
from model.layers import SHConv, WindowAttentionLayer, DownSample, UpSample

Nonlinearity = partial(torch.nn.SiLU, inplace=True)

class SHBlock(torch.nn.Module):
    def __init__(self, sht, num_channel):
        super(SHBlock, self).__init__()
        self.sht = sht
        self.in_conv = torch.nn.Conv2d(num_channel, num_channel//2, 1)
        self.sh_conv = SHConv(sht, num_channel//2, num_channel//2)
        self.out_conv = torch.nn.Conv2d(num_channel//2, num_channel, 1)
        self.norm = torch.nn.LayerNorm(num_channel)
        self.nonlinearity = Nonlinearity()
        self.dropout = torch.nn.Dropout(config.DROPOUT)

    def forward(self, x):
        y = self.in_conv(self.dropout(self.nonlinearity(self.norm(x.permute(0,2,3,1)).permute(0,3,1,2))))
        y = self.nonlinearity(y)[0]
        y = sht_func.grid_to_sh(y.reshape((-1,self.sht.batch_size,y.shape[1],y.shape[2])).permute(0,1,3,2), self.sht)
        y = y.reshape((-1,y.shape[-1]))
        y = self.sh_conv(y)
        y = sht_func.sh_to_grid(y.reshape((-1,self.sht.batch_size,y.shape[1])), self.sht).permute(0,1,3,2)
        y = y.reshape((1,-1,y.shape[2],y.shape[3]))
        y = self.out_conv(self.dropout(self.nonlinearity(y)))
        return x + y

class AllBlocks(torch.nn.Module):
    def __init__(self, sht, size_x, size_y, num_channel, num_sh_conv, in_channel=None, out_channel=None, kernel_sizes=(8, 8)):
        super(AllBlocks, self).__init__()
        self.in_conv = self.out_conv = None
        if in_channel is not None and in_channel != num_channel:
            self.in_conv = torch.nn.Conv2d(in_channel, num_channel, 1)
        if out_channel is not None and out_channel != num_channel:
            self.out_conv = torch.nn.Conv2d(num_channel, out_channel, 1)
        self.num_sh_conv = num_sh_conv
        self.grid_blocks = torch.nn.ModuleList([WindowAttentionLayer(kernel_sizes, num_channel, num_channel//32, num_channel*4, config.DROPOUT, config.DROPOUT, Nonlinearity) for _ in range(num_sh_conv)])
        self.sh_blocks = torch.nn.ModuleList([SHBlock(sht, num_channel) for _ in range(num_sh_conv)])
        self.dropout = torch.nn.Dropout(config.DROPOUT)

    def forward(self, x):
        if self.in_conv is not None: x = self.in_conv(self.dropout(x))
        for i in range(self.num_sh_conv):
            x = self.sh_blocks[i](x)
            x = self.grid_blocks[i](x)
        if self.out_conv is not None: x = self.out_conv(self.dropout(x))
        return x

class Kunyu(torch.nn.Module):
    def __init__(self, is_legacy=False):
        super(Kunyu, self).__init__()
        self.nonlinearity = Nonlinearity()
        self.loc_embedding = torch.nn.Parameter(torch.Tensor(1, 16, config.FULLSIZE_X, config.FULLSIZE_Y))
        torch.nn.init.normal_(self.loc_embedding)
        self.input_layers = torch.nn.Sequential(
                torch.nn.Dropout(config.DROPOUT), torch.nn.Conv2d(config.INPUT_CHANNEL, 1024, 1), self.nonlinearity,
                torch.nn.Dropout(config.DROPOUT), torch.nn.Conv2d(1024, 384, 1), self.nonlinearity)
        self.down_pass, self.up_pass = [], []
        channel, lmax, size_x, size_y = 384, config.LMAX, config.FULLSIZE_X, config.FULLSIZE_Y
        channels = [384, 768, 1024, 1024]
        for i, depth in enumerate([3, 4, 6, 6]):
            new_channel = channels[i]
            sht = torch_sht.SHT(lmax)
            sht.set_batch(32)
            sht.set_grid(size_x, size_y, 2|4096|262144)
            self.down_pass.append(AllBlocks(sht, size_x, size_y, new_channel, depth, in_channel=channel, kernel_sizes=(8, 16) if i < 2 else (16, 32)))
            self.up_pass.append(AllBlocks(sht, size_x, size_y, new_channel, depth, out_channel=channel, kernel_sizes=(8, 16) if i < 2 else (16, 32)))
            channel, lmax, size_x, size_y = new_channel, lmax//2, size_x//2, size_y//2
        self.down_pass, self.up_pass = [torch.nn.ModuleList(mod_list) for mod_list in [self.down_pass, self.up_pass]]
        self.down_sample = DownSample(2)
        self.up_sample = UpSample(2)
        self.output_layers = torch.nn.Sequential(
                torch.nn.Dropout(config.DROPOUT), torch.nn.Conv2d(384, 1024, 1), self.nonlinearity,
                torch.nn.Dropout(config.DROPOUT), torch.nn.Conv2d(1024, config.OUTPUT_CHANNEL, 1))
        std_deltas = [config.VAR_STATS[var]['std_delta'] / config.VAR_STATS[var]['std'] for var in config.VAR_LIST['main']]
        std_deltas += [1.0 for _ in config.VAR_LIST['output_only']]
        self.register_buffer('std_deltas', torch.tensor(std_deltas).view(1, -1, 1, 1))
        self.is_legacy = is_legacy

    def forward(self, input):
        assert input.shape[0] == 1
        x = torch.cat([input, self.loc_embedding], dim=1)
        x = self.input_layers(x)
        xs = []
        for i in range(4):
            x = self.down_pass[i](x)
            xs.append(x)
            if i < 3:
                x = self.down_sample(x)
        for i in range(3, -1, -1):
            if i < 3:
                x = xs[i] + self.up_sample(x)
                xs[i] = None
            x = self.up_pass[i](x)
        x = self.output_layers(x) * self.std_deltas
        x[:,:config.MAIN_CHANNEL] += input[:,:config.MAIN_CHANNEL]
        if not self.is_legacy:
            x[:,config.R_VARS] = torch.clip(x[:,config.R_VARS], min=0.)
        return x

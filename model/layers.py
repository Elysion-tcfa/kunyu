import numpy as np, scipy.signal, torch

import config

def complex_dropout(tensor, prob):
    if prob == 0.0: return tensor
    mask = torch.nn.functional.dropout(torch.ones_like(tensor.real), prob)
    return tensor * mask

class SHConv(torch.nn.Module):
    def __init__(self, sht, in_channel, out_channel):
        super(SHConv, self).__init__()
        self.l = sht.lmax + 1
        self.w = torch.nn.Parameter(torch.Tensor(sht.lmax + 1, in_channel, out_channel))
        self.w2 = torch.nn.Parameter(torch.Tensor(sht.lmax + 1, in_channel, out_channel, 2))
        self.b = torch.nn.Parameter(torch.zeros(out_channel))
        self.in_channel = in_channel
        torch.nn.init.normal_(self.w, std=1./np.sqrt(in_channel))
        torch.nn.init.normal_(self.w2, std=1./np.sqrt(2*in_channel))
        perm = []
        inv_perm = [0 for i in range(sht.nlm)]
        for i in range(sht.lmax + 1):
            for j in range(sht.lmax + 1):
                if i > j:
                    perm.append(sht.idx(i, i))
                else:
                    inv_perm[sht.idx(j, i)] = j * (sht.lmax + 1) + i
                    perm.append(sht.idx(j, i))
        perm, inv_perm = [torch.tensor(x) for x in (perm, inv_perm)]
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', inv_perm)

    def forward(self, input):
        tensor = torch.view_as_complex(torch.view_as_real(input.permute(1,0))[self.perm]).reshape((self.l, self.l, -1))
        w2 = torch.view_as_complex(self.w2)
        w2 = torch.cat([w2[0:1].real.to(torch.complex64), w2[1:]], dim=0)
        tensor = torch.matmul(complex_dropout(tensor, config.DROPOUT), w2).permute(1,0,2)
        tensor = torch.matmul(complex_dropout(tensor, config.DROPOUT), self.w.to(torch.complex64)).reshape((self.l * self.l, -1))
        output = torch.view_as_complex(torch.view_as_real(tensor)[self.inv_perm]).permute(1,0)
        output[:,0] += self.b
        return output

class SelfAttention(torch.nn.Module):
    def __init__(self, input_sizes, num_channel, num_head, rpr_types='none', rpr_limits=None, dropout_prob=0.):
        super().__init__()
        if type(rpr_types) not in [tuple, list]:
            rpr_types = (rpr_types, rpr_types)
        if type(rpr_limits) not in [tuple, list]:
            rpr_limits = (rpr_limits, rpr_limits)
        self.input_size = input_sizes[0]*input_sizes[1]
        self.num_head = num_head
        self.head_dim = num_channel // num_head
        self.qkv = torch.nn.Linear(num_channel, num_channel*3)
        coords = torch.stack(torch.meshgrid([torch.arange(input_sizes[0]), torch.arange(input_sizes[1])], indexing='ij'), dim=-1).reshape((-1, 2))
        coords_diff = coords[:,None,:] - coords[None]
        rpr_sizes = []
        for i in range(2):
            if rpr_types[i] == 'none':
                coords_diff[...,i] = 0
                rpr_sizes.append(1)
            elif rpr_types[i] == 'circular':
                if rpr_limits[i] is not None:
                    coords_diff[...,i] = (coords_diff[...,i] + input_sizes[i] // 2) % input_sizes[i] - input_sizes[i] // 2
                    coords_diff[...,i] = (torch.clamp(coords_diff[...,i], min=-rpr_limits[i], max=rpr_limits[i]) + rpr_limits[i]) % (2 * rpr_limits[i])
                    rpr_sizes.append(rpr_limits[i]*2)
                else:
                    coords_diff[...,i] = coords_diff[...,i] % input_sizes[i]
                    rpr_sizes.append(input_sizes[i])
            else:
                rpr_limit = rpr_limits[i] if rpr_limits[i] is not None else input_sizes[i] - 1
                coords_diff[...,i] = torch.clamp(coords_diff[...,i], min=-rpr_limit, max=rpr_limit) + rpr_limit
                rpr_sizes.append(rpr_limit*2+1)
        rpr_index = coords_diff[...,0] * rpr_sizes[1] + coords_diff[...,1]
        self.register_buffer('rpr_index', rpr_index)
        self.rpr_table = torch.nn.Parameter(torch.zeros(num_head, rpr_sizes[0]*rpr_sizes[1]))
        torch.nn.init.normal_(self.rpr_table, std=0.02)
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        qkv = self.qkv(x).reshape((-1, self.input_size, 3, self.num_head, self.head_dim)).permute(2,0,3,1,4)
        attn = (qkv[0] @ qkv[1].transpose(-2, -1)) / self.head_dim ** 0.5 #b,h,n,n
        attn += self.rpr_table[:,self.rpr_index.reshape(-1)].reshape((self.num_head, self.input_size, self.input_size))
        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        v = qkv[2].clone()
        del qkv
        x = attn @ v #b,h,n,m
        return x.permute(0,2,1,3).reshape((-1, self.input_size, self.num_head*self.head_dim))

class AttentionMLP(torch.nn.Module):
    def __init__(self, num_channel, mlp_channel, dropout_prob=0., nonlinearity=torch.nn.SiLU):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_channel, mlp_channel)
        self.fc2 = torch.nn.Linear(mlp_channel, num_channel)
        self.nonlinearity = nonlinearity()
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

class WindowAttentionLayer(torch.nn.Module):
    def __init__(self, window_sizes, num_channel, num_head, mlp_channel, dropout_prob=0., attn_dropout_prob=0., nonlinearity=torch.nn.SiLU):
        super().__init__()
        self.window_sizes = window_sizes
        self.num_channel = num_channel
        self.norm1 = torch.nn.LayerNorm(num_channel)
        self.norm2 = torch.nn.LayerNorm(num_channel)
        self.fc = torch.nn.Linear(num_channel, num_channel)
        self.attns = torch.nn.ModuleList([
            SelfAttention(window_sizes, num_channel//2, num_head//2, 'normal', dropout_prob=attn_dropout_prob),
            SelfAttention(window_sizes, num_channel//2, num_head//2, 'normal', dropout_prob=attn_dropout_prob)])
        self.mlp = AttentionMLP(num_channel, mlp_channel, dropout_prob, nonlinearity=nonlinearity)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3] #b,c,h,w
        wh, ww = self.window_sizes
        c = self.num_channel//2
        x = x.permute(0,2,3,1)
        y = self.norm1(x)
        y1 = y[...,:c].reshape((-1, h//wh, wh, w//ww, ww, c)).permute(0,1,3,2,4,5).reshape((-1, wh*ww, c))
        y2 = torch.roll(y[...,c:], ww//2, -2)
        del y
        y1 = self.attns[0](y1)
        y1 = y1.reshape((-1, h//wh, w//ww, wh, ww, c)).permute(0,1,3,2,4,5).reshape((-1, h, w, c))
        y2 = torch.cat([
            torch.cat([torch.flip(y2[:,:wh//2,w//2:], [1]), y2[:,:wh//2,:w//2]], dim=1).reshape((-1, wh, w//(2*ww), ww, c)).permute(0,2,1,3,4),
            y2[:,wh//2:-wh//2].reshape((-1, h//wh-1, wh, w//ww, ww, c)).permute(0,1,3,2,4,5).reshape((-1, (h//wh-1)*(w//ww), wh, ww, c)),
            torch.cat([y2[:,-wh//2:,:w//2], torch.flip(y2[:,-wh//2:,w//2:], [1])], dim=1).reshape((-1, wh, w//(2*ww), ww, c)).permute(0,2,1,3,4)
            ], dim=1).reshape((-1, wh*ww, c))
        y2 = self.attns[1](y2).reshape((-1, h//wh*w//ww, wh, ww, c))
        y2_1 = y2[:,:w//(2*ww)].permute(0,2,1,3,4).reshape((-1, wh, w//2, c))
        y2_2 = y2[:,w//(2*ww):-w//(2*ww)].reshape((-1, h//wh-1, w//ww, wh, ww, c)).permute(0,1,3,2,4,5).reshape((-1, h-wh, w, c))
        y2_3 = y2[:,-w//(2*ww):].permute(0,2,1,3,4).reshape((-1, wh, w//2, c))
        y2 = torch.cat([
            torch.cat([y2_1[:,wh//2:], torch.flip(y2_1[:,:wh//2], [1])], dim=2),
            y2_2,
            torch.cat([y2_3[:,:wh//2], torch.flip(y2_3[:,wh//2:], [1])], dim=2)], dim=1)
        y2 = torch.roll(y2, -ww//2, -2)
        y = x + self.fc(torch.cat([y1, y2], dim=-1))
        return (y + self.mlp(self.norm2(y))).permute(0,3,1,2)

class DownSample(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        kernel = torch.tensor(scipy.signal.firwin(numtaps=6*m, cutoff=1, width=2*(np.sqrt(2)-1), fs=2*m), dtype=torch.float32)
        self.register_buffer('kernel', kernel)
        self.padding = (5*m)//2

    def forward(self, x):
        tgt_size_x = x.shape[2] + self.padding * 2
        while x.shape[2] < tgt_size_x:
            padding = (tgt_size_x - x.shape[2]) // 2
            x = torch.cat([
                torch.flip(torch.cat([x[:,:,:padding,x.shape[3]//2:], x[:,:,:padding,:x.shape[3]//2]], dim=3), [2]),
                x,
                torch.flip(torch.cat([x[:,:,-padding:,x.shape[3]//2:], x[:,:,-padding:,:x.shape[3]//2]], dim=3), [2])
            ], dim=2)
        x = torch.nn.functional.conv2d(x, self.kernel[None,None,:,None].repeat([x.shape[1],1,1,1]), groups=x.shape[1], stride=(self.m,1))
        x = torch.cat([x[:,:,:,-self.padding:], x, x[:,:,:,:self.padding]], dim=3)
        x = torch.nn.functional.conv2d(x, self.kernel[None,None,None,:].repeat([x.shape[1],1,1,1]), groups=x.shape[1], stride=(1,self.m))
        return x

class UpSample(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        kernel = torch.tensor(scipy.signal.firwin(numtaps=6*m, cutoff=1, width=2*(np.sqrt(2)-1), fs=2*m), dtype=torch.float32)
        self.register_buffer('kernel', kernel)
        self.padding = (5*m)//2

    def forward(self, x):
        x = torch.nn.functional.pad(x[:,:,:,None,:,None], [0, self.m-1, 0, 0, 0, self.m-1]).reshape(x.shape[:2] + (x.shape[2] * self.m, x.shape[3] * self.m)) * (self.m * self.m)
        x = torch.nn.functional.pad(x, [0, 0, self.m-1, 0])
        x = torch.cat([
            torch.flip(torch.cat([x[:,:,self.m-1:self.m-1+self.padding,x.shape[3]//2:], x[:,:,self.m-1:self.m-1+self.padding,:x.shape[3]//2]], dim=3), [2]),
            x,
            torch.flip(torch.cat([x[:,:,-self.padding-self.m+1:-self.m+1,x.shape[3]//2:], x[:,:,-self.padding-self.m+1:-self.m+1,:x.shape[3]//2]], dim=3), [2])
        ], dim=2)
        x = torch.nn.functional.pad(x, [self.m-1, 0])
        x = torch.nn.functional.conv2d(x, self.kernel[None,None,:,None].repeat([x.shape[1],1,1,1]), groups=x.shape[1])
        x = torch.cat([x[:,:,:,-self.padding-self.m+1:-self.m+1], x, x[:,:,:,self.m-1:self.m-1+self.padding]], dim=3)
        x = torch.nn.functional.conv2d(x, self.kernel[None,None,None,:].repeat([x.shape[1],1,1,1]), groups=x.shape[1])
        return x

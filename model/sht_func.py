import torch
import math

class GridToSHFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sht):
        if input.dtype == torch.float16: input = input.float()
        output = sht.grid_to_sh(input)
        ctx.sht = sht
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sht = ctx.sht
        grad = grad_output.clone().detach()
        grad[..., :sht.lmax+1] *= 2
        batch_dims = len(grad_output.shape) - 1
        grad = sht.sh_to_grid(grad)
        w = sht.gauss_weights().to(grad.dtype).to(grad.device)
        w = torch.cat([w, torch.flip(w, [0])], dim=0) / 2
        w = w.reshape((1,) * batch_dims + (1, -1))
        grad *= w * math.pi / sht.nlat
        return grad, None

class SHToGridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sht):
        output = sht.sh_to_grid(input)
        ctx.sht = sht
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sht = ctx.sht
        batch_dims = len(grad_output.shape) - 2
        w = sht.gauss_weights().to(grad_output.dtype).to(grad_output.device)
        w = torch.cat([w, torch.flip(w, [0])], dim=0) / 2
        w = w.reshape((1,) * batch_dims + (1, -1))
        grad = sht.grid_to_sh(grad_output / w)
        grad *= sht.nlat / math.pi
        grad[..., :sht.lmax+1] /= 2
        return grad, None

grid_to_sh = GridToSHFunction.apply
sh_to_grid = SHToGridFunction.apply

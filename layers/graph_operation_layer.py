import torch
import torch.nn as nn

class HeteroGraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A, transform_matrix, **kwargs):
        assert A.size(1) == self.kernel_size # x: (batch_size, 64, T, V) = (batch, 64, 15, 260)
        x = self.conv(x)
        n, kc, t, v = x.size()
        assert kc // self.kernel_size == transform_matrix.size(1) == transform_matrix.size(2)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x_transformed = torch.einsum('nkctv,kcx->nkxtv', [x, transform_matrix]) # the shape of x_transformed is the same as x
        x_convoluted = torch.einsum('nkxtv,nkvw->nxtw', [x_transformed, A])
        #x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x_convoluted.contiguous(), A

class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A, **kwargs):
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()

        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x.contiguous(), A

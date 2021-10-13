from ...imports.torch_imports import *
from .layers import *
from ...utils.net_utils import ifnone
from ...utils.common_utils import OrderedDict

class ResidualBlock(nn.Module):
    """Residual block without batch normalization"""

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        out = self.conv(x) + x

        return out


class Interpolate(nn.Module):
    def __init__(self, factor=2, mode="nearest", align_corners=True):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.factor = factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if self.mode != "nearest":
            x = self.interp(
                x,
                scale_factor=self.factor,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        else:
            x = self.interp(x, scale_factor=self.factor, mode=self.mode)
        return x
    


class PixelShuffle_ICNR(nn.Module):
    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float = 0.01,
    ) -> None:
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(
            ni,
            nf * (scale ** 2),
            kernel_size=1,
            use_activation=False,
            use_batch_norm=False,
        )
        self.icnr(self.conv[0].weight, scale)
        self.shuffle = nn.PixelShuffle(scale)
        self.pad = nn.ReflectionPad2d(1)
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur
        self.relu = relu(True, leaky=leaky)

    def forward(self, x: torch.tensor):
        x = self.conv(x)
        x = self.relu(x)
        x = self.shuffle(x)

        if self.do_blur:
            x = self.pad(x)
            return self.blur(x)
        else:
            return x

    def icnr(
        self, x: torch.tensor, scale: int = 2, init: Callable = nn.init.kaiming_normal_
    ):
        ni, nf, h, w = x.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale ** 2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        x.data.copy_(k)



class Upsampler(nn.Module):
    def __init__(self,upscale=2,nf=64,out_nc=3):
        super().__init__()
        self.out_nc = out_nc
        self.nf = nf
        self.upscale = upscale
        if self.upscale == 2:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
            
        elif self.upscale == 3:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=3, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
            
        elif self.upscale == 6:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=3, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
        elif self.upscale == 8:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate3",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv3", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
        elif self.upscale == 16:
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate3",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv3", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate4",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv4", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu4", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )
        elif self.upscale==4:
            
            self.tail = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "interpolate1", 
                            Interpolate(factor=2,mode="nearest", align_corners=False),
                        ),
                        ("up_conv1", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        (
                            "interpolate2",
                            Interpolate(factor=2, mode="nearest", align_corners=False),
                        ),
                        ("up_conv2", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),
                        ("hrconv", nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)),
                        (
                            "hrconv_lrelu",
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                        (
                            "conv_last",
                            nn.Conv2d(self.nf, self.out_nc, 3, 1, 1, bias=True),
                        ),
                    ]
                )
            )

    def forward(self,x):
        return self.tail(x)
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class PlAnimeGANv2(pl.LightningModule):
    def __init__(self, args=None):
        super(PlAnimeGANv2, self).__init__()
        self.args = args
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')


    def forward(self, tensor):
        x = self.model(tensor)
        return x

    def adversarial_loss(self, x, y):
        return F.binary_cross_entropy(x, y)

    def l2_loss(self, x, y):
        return F.mse_loss(x, y)

    def perceptual_loss(self, x, y):
        return self.lpips(x, y)

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        src_image, tgt_image = train_batch

        if optimizer_idx == 0:
            self.generated_imgs = self.generator(src_image)

            grid = torchvision.utils.make_grid(torch.cat([src_image[:6], self.generated_imgs[:6], tgt_image[:6]]),
                                               nrow=src_image.size(0) if src_image.size(0) < 6 else 6)
            self.logger.log_image("generated_images", [grid])

            valid = torch.ones(src_image.size(0), 1)
            valid = valid.type_as(src_image)

            g_loss_value = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            perceptual_loss_value = self.perceptual_loss(self.generated_imgs, tgt_image)
            l2_loss_value = self.l2_loss(self.generated_imgs, tgt_image)

            total_loss = g_loss_value * 0.1 + perceptual_loss_value + l2_loss_value

            self.log("train/g_loss", g_loss_value, prog_bar=True)
            self.log("train/perceptual_loss", perceptual_loss_value, prog_bar=True)
            self.log("train/l2_loss", l2_loss_value, prog_bar=True)
            return {'loss': total_loss}

        if optimizer_idx == 1:
            valid = torch.ones(tgt_image.size(0), 1)
            valid = valid.type_as(tgt_image)

            real_loss = self.adversarial_loss(self.discriminator(tgt_image), valid)

            fake = torch.zeros(src_image.size(0), 1)
            fake = fake.type_as(src_image)

            fake_loss = self.adversarial_loss(self.discriminator(self.generator(src_image).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2 * 0.1
            self.log("train/d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.learning_rate)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.learning_rate)
        return [opt_g, opt_d]


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, tensor, align_corners=True):
        out = self.block_a(tensor)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, tensor.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, tensor):
        x = self.conv1(tensor)
        x = F.leaky_relu(x, negative_slope=0.2) * (2 ** 0.5)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2) * (2 ** 0.5)

        skip_x = self.skip(tensor)
        return (x + skip_x) / (2 ** 0.5)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.stddev_group = 4
        self.stddev_feat = 1

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            ResBlock(64, 128),
            ResBlock(128, 256),
            ResBlock(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.final_conv = nn.Conv2d(512 + 1, 512, kernel_size=3, padding=1)
        self.final_linear = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.Linear(512, 1)
        )

    def forward(self, tensor):
        out = self.model(tensor)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        out = torch.sigmoid(out)
        return out


class SPNormResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPNormResBlock, self).__init__()
        self.conv1 = torch.nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        self.conv2 = torch.nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))

        self.skip = torch.nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False))

    def forward(self, tensor):
        x = self.conv1(tensor)
        x = F.leaky_relu(x, negative_slope=0.2) * (2 ** 0.5)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2) * (2 ** 0.5)

        skip_x = self.skip(tensor)
        return (x + skip_x) / (2 ** 0.5)


class SPNormDiscriminator(nn.Module):
    def __init__(self):
        super(SPNormDiscriminator, self).__init__()
        self.stddev_group = 4
        self.stddev_feat = 1

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            SPNormResBlock(64, 128),
            SPNormResBlock(128, 256),
            SPNormResBlock(256, 512),
            SPNormResBlock(512, 512),
            SPNormResBlock(512, 512),
            SPNormResBlock(512, 512),
            SPNormResBlock(512, 512),
        )
        self.final_conv = nn.Conv2d(512 + 1, 512, kernel_size=3, padding=1)
        self.final_linear1 = nn.Linear(512 * 2 * 2, 512) # 해상도별 조절 필요
        self.final_linear2 = nn.Linear(512, 1)

    def forward(self, tensor):
        out = self.model(tensor)
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear1(out)
        out = F.leaky_relu(out, negative_slope=0.2) * (2 ** 0.5)
        out = self.final_linear2(out)
        out = torch.sigmoid(out)
        return out

import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F
from torch.autograd import Variable
import util.util as util



# pingjunmoxing  test_net_netG.pth
#Inception score = (3.3380485, 0.09124949); SSIM score = 0.774556061674636  

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, kernel_size):
        super(double_conv, self).__init__()
        if kernel_size == 1:
            p = 0
        else:
            p = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=p),
            #nn.BatchNorm2d(in_ch),
            nn.InstanceNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=p),
            #nn.BatchNorm2d(out_ch), 
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5)
        )

    def forward(self, x, y):
        x1 = x
        x = self.conv(x)
        if y:
            x = x1+x
        return x

class Guided_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, num=None):
        super(Guided_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv = double_conv(in_ch=in_dim, out_ch=in_dim,kernel_size=3)
        self.dconv = double_conv(in_ch=in_dim*2, out_ch=in_dim*2,kernel_size=3)
        self.conv2 = double_conv(in_ch=in_dim, out_ch=in_dim,kernel_size=3)
        self.dconv2 = double_conv(in_ch=in_dim*2, out_ch=in_dim,kernel_size=3)
        self.att2 = Self_Attn(in_dim*2, 'relu')
        self.softmax = nn.Softmax(dim=-1)
        self.num = num
    def forward(self, x, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                x2 : input feature maps( B X C X W X H) x2 guided x
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x2 = torch.cat((x, x2), 1)
        x2, _ = self.att2(x2, x2)
        x2 = self.dconv(x2, True)
        x = self.conv(x, True)
        x = self.conv2(x, True)
        x2 = self.dconv2(x2, False)

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (N) X C
        proj_key = self.key_conv(x2).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        
        
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
       
        out = self.gamma * out + x


        return out, attention, x2

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x, x0):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + self.beta * x + x0
        return out, attention



     

class PoNAModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert(n_blocks >= 0 and type(input_nc) == list)
        super(PATNModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        n_blocks = 3
    
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_stream1_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2),
                            nn.ReLU(True)]

        mult = 2**n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(Guided_Attn(ngf*mult, 'relu', num=i))
        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2)),
                            nn.ReLU(True)]

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]

        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)

    def forward(self, input): # x from stream 1 and stream 2
        # here x should be a tuple
        x1, x2 = input
        # down_sample
        x1 = self.stream1_down(x1)
        x2 = self.stream2_down(x2)
        #res = x2
        # att_block
        for model in self.att:
            x1, _ , x2= model(x1, x2)
        x1 = self.stream1_up(x1)

        return x1


class PoNANetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(PoNANetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = PoNAModel(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)







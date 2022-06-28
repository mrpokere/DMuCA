import torch.nn as nn
import math
from torch.nn import functional as F
import torch
from torch import optim
import torch.nn.init as init
from torchsummary import summary

class SeMCA(nn.Module):
    def __init__(self,channel= 64,img_size=9,d=9,spe_head_Sidelength=5,padsize=15,share_head=9,spe_head=9):
        super().__init__()  
        self.d = d
        self.pad1 = self.d//2
        self.img_size = img_size
        self.spe_head_Sidelength = spe_head_Sidelength#spe_head_Sidelength*==spe_head
        self.padsize = padsize
        self.share_head = share_head
        self.spe_head = spe_head
        self.pad_space = padsize-img_size
        self.pad_spe_start = self.pad_space//2
        self.v_Pad = nn.ConstantPad3d((self.pad_space,0,self.pad_space,0,self.pad1,self.pad1), 0)
        self.kq_Pad = nn.ZeroPad2d(padding=(0,self.pad_space,0,self.pad_space))
        self.Unfold = nn.Unfold(kernel_size=(int((self.share_head)**0.5),\
            int((self.share_head)**0.5)), dilation=int((self.spe_head)**0.5), padding=0, stride=1)
        
        self.convd_key =nn.Conv1d(in_channels=self.img_size*self.img_size,\
            out_channels=self.img_size*self.img_size, kernel_size=self.d,\
                stride=1, padding=self.d//2,groups=self.img_size*self.img_size)
           
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*self.spe_head*self.share_head,self.spe_head*self.share_head, 1,groups=self.spe_head,bias=0),
            nn.BatchNorm2d(self.spe_head*self.share_head),
            nn.ReLU(),
            nn.Conv2d(self.spe_head*self.share_head,self.spe_head*self.d,1,groups=self.spe_head)
        )
        self.convd_value = nn.Conv2d(channel,channel,3,padding=1,groups=channel)


    def forward(self, x):
        bs,c,h,w = x.shape
        k = x.reshape(bs,c,h*w).permute(0,2,1)
        k1 = self.convd_key(k).permute(0,2,1).reshape(bs,c,h,w)
        k1_pad,q_pad = self.kq_Pad(k1),self.kq_Pad(x)
        key_pad = self.Unfold(k1_pad).permute(0,2,1)
        q_pad = self.Unfold(q_pad).permute(0,2,1)
        key_q = torch.stack((q_pad,key_pad),dim=2).reshape(bs,2*self.spe_head,self.share_head,c)
        key_q = key_q.reshape(bs,2*self.spe_head*self.share_head,c).unsqueeze(-1) 
        attention = self.attention_embed(key_q)#[32, 225, 64, 1])
        attention = attention.view(bs,self.spe_head,self.d,c).repeat(1,self.share_head,1,1).permute(0,3,1,2)
        # [32, 225, 64, 1]--[32, 25, 9, 64]--[32, 225, 9, 64]--[32, 64, 225, 9]
        attention = attention.reshape(bs,c,self.padsize//self.spe_head_Sidelength,\
            self.padsize//self.spe_head_Sidelength,self.spe_head_Sidelength,self.spe_head_Sidelength,self.d)
        attention = attention.transpose(3,4).reshape(bs,c,self.padsize,self.padsize,self.d)
        attention = F.softmax(attention,dim=-1).permute(0,1,4,2,3)
        v = self.convd_value(x)
        pad_v = self.v_Pad(v)
        v = pad_v.reshape(bs,-1,self.padsize*self.padsize)
        v = v.unfold(1, self.d, 1).permute(0,1,3,2).reshape(bs,-1,self.d,self.padsize,self.padsize)
        temp_out = torch.sum(v*attention,dim=-3)
        out = temp_out+k1_pad
        out = out[:,:,self.pad_spe_start:h+self.pad_spe_start,self.pad_spe_start:w+self.pad_spe_start]
        return out


class SaMCA(nn.Module):

    def __init__(self,dim=64,kernel_size=5,spa_head=1):
        super().__init__()
        self.dim=dim
        self.spa_head = spa_head
        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=dim,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
            )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
            )
        factor=2
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor, 1,bias=False,groups=self.spa_head),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,self.spa_head,1,groups=self.spa_head)
            )
        
    def forward(self, x):
        
        bs,c,h,w=x.shape
        k1=self.key_embed(x) 
        v = self.value_embed(x)  
        v=x.view(bs,c,-1) 
        k_q = torch.stack((k1,x),2).reshape(bs,c*2,h,w)
        att=self.attention_embed(k_q).reshape(bs,self.spa_head,h*w)
        k2=F.softmax(att,dim=-1).unsqueeze(-2)
        k2 = k2.repeat(1,1,c//self.spa_head,1).reshape(bs,c,-1)
        k2 = k2*v
        k2=k2.view(bs,c,h,w)
        x = k1+k2
        return x

class DMuCA(nn.Module):

    def __init__(self, channel=512,spa_kerner=3,img_size=9,spe_kerner=9,spa_head=64,spe_head=1):
        super().__init__()
        spe_head_Sidelength = int(pow(spe_head,0.5))
        padsize = ((img_size-1)//spe_head_Sidelength + 1)*spe_head_Sidelength
        share_head = int(padsize**2 / spe_head)
        self.SeMCA = SeMCA(channel=channel,img_size=img_size,d=spe_kerner,\
            spe_head_Sidelength=spe_head_Sidelength,padsize=padsize,share_head=share_head,spe_head=spe_head)
        self.SaMCA = SaMCA(dim=channel,kernel_size=spa_kerner,spa_head=spa_head,)
        self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out1=self.SeMCA(x)
        out2=self.SaMCA(x)
        out = self.alpha*out1 + (1-self.alpha)*out2
        return out
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, spe_head,spa_head,spa_kerner,spe_kerner,img_size,inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        self.relu = nn.ReLU(inplace=True)
        self.DMuCA = nn.Sequential(
            DMuCA(channel=planes,img_size=img_size,spe_kerner=spe_kerner,\
                spa_kerner=spa_kerner,spa_head=spa_head,spe_head=spe_head),
            nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = x 
        x = self.conv(x) 
        out = self.DMuCA(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, num_classes, channel = 103,imgsize=15,spa_head=1,spe_head=1):
        super(ResNet, self).__init__()
        
        self.dim = 64
        self.imgsize =imgsize
        self.spe_kerner = 9
        self.spa_kerner = 5
        self.spe_head = spe_head
        self.spa_head = spa_head
        block = BasicBlock
        
        self.conv1 = nn.Conv2d(channel, self.dim, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.dim)
        self.layer2 = self._make_layer(block,self.dim, stride=2)
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(self.dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, stride=1):
        layers = []
        layers.append(block(self.spe_head,self.spa_head,self.spa_kerner,self.spe_kerner,self.imgsize,self.dim, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.contiguous().view(-1,self.dim)
        x = self.fc(x)
        return x

def get_model(name, **kwargs):

    kwargs.setdefault('device', torch.device('cpu'))
    weights = torch.ones(kwargs['n_classes'])
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(kwargs['device'])
    kwargs.setdefault('weights', weights)
    kwargs.setdefault('batch_size', 32)
   
    if name == 'IndianPines':
        kwargs.setdefault('validation_percentage', 0.1)
        kwargs.setdefault('bands', 200)
        kwargs.setdefault('spa_head', 16)
        kwargs.setdefault('spe_head', 25)
    elif name == 'PaviaU':
        kwargs.setdefault('validation_percentage', 0.02)
        kwargs.setdefault('bands', 103)
        kwargs.setdefault('spa_head', 32)
        kwargs.setdefault('spe_head', 49)
    elif name == 'Houston':
        kwargs.setdefault('validation_percentage', 0.005)
        kwargs.setdefault('bands', 144)
        kwargs.setdefault('spa_head', 32)
        kwargs.setdefault('spe_head', 49)
        
    model = ResNet(num_classes=kwargs['n_classes'], channel = kwargs['bands'],
    imgsize=kwargs['patch_size'],spa_head=kwargs['spa_head'],spe_head=kwargs['spe_head'])
    optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], weight_decay=0.0001, momentum=0.9)
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(kwargs['device'])
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                                        patience=10, verbose=True))
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('center_pixel', True)
    return model, optimizer, criterion, kwargs



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(17, channel = 200,imgsize=13,spe_kerner=9,spa_kerner=5,
    spa_head=16,spe_head=49).to(device)

    with torch.no_grad():
        summary(model, (200, 13, 13))
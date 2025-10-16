import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import numpy as np
import cv2
import random
import time
from torch.nn.utils import spectral_norm as sn

from models.archs.transformer.Models import Encoder_patch66



###############################
class low_light_transformer4(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(low_light_transformer4, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

        
        self.num_feature = 16
        self.channel = nf
        self.common_feature = nn.Parameter(torch.zeros(self.num_feature, self.channel).float())
        self.common_feature.require_grad=False
        self.patch_size = 4
        self.common_feature_patch = nn.Parameter(torch.zeros(self.num_feature, self.channel, self.patch_size, self.patch_size).float())
        self.common_feature_patch.require_grad=False
        self.embedding = nn.Embedding(self.num_feature, nf)

        
        self.update_feature = nn.Sequential(nn.Conv2d(3*nf, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(),
                                            nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(),
                                            nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(),
                                            nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(),
                                            nn.Conv2d(nf*4, nf, 1, 1, 0, bias=True), nn.BatchNorm2d(nf), nn.ReLU(),
                                            nn.Conv2d(nf, nf, 1, 1, 0, bias=True))
        self.update_feature_patch = nn.Sequential(nn.Conv2d(3*nf, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True), nn.BatchNorm2d(nf), nn.ReLU(), 
                                            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        
        self.propagate = nn.Sequential(nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                       nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                       nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                       nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                       nn.Conv2d(nf*4, nf*4, 1, 1, 0, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                       nn.Conv2d(nf*4, nf, 1, 1, 0, bias=True), nn.BatchNorm2d(nf), nn.ReLU(), 
                                       nn.Conv2d(nf, nf, 1, 1, 0, bias=True))
        self.propagate_patch = nn.Sequential(nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf*4, 3, 1, 1, bias=True), nn.BatchNorm2d(nf*4), nn.ReLU(), 
                                            nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True), nn.BatchNorm2d(nf), nn.ReLU(), 
                                            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        
        self.feature_unshuffle = nn.PixelUnshuffle(self.patch_size)
        self.feature_shuffle = nn.PixelShuffle(self.patch_size)
        self.norm_query1 = nn.LayerNorm(nf)
        self.norm_query2 = nn.LayerNorm(nf)
        self.norm_query3 = nn.LayerNorm(nf*self.patch_size*self.patch_size)
        self.norm_query4 = nn.LayerNorm(nf*self.patch_size*self.patch_size)

        self.norm_feature = nn.LayerNorm(nf)
        self.norm_patch = nn.LayerNorm([nf, self.patch_size, self.patch_size])
        self.norm_feature2 = nn.LayerNorm(nf)
        self.norm_patch2 = nn.LayerNorm([nf, self.patch_size, self.patch_size])
    
    def print_self(self):
        print(torch.max(self.common_feature), torch.min(self.common_feature), torch.mean(self.common_feature))
        print(torch.max(self.common_feature_patch), torch.min(self.common_feature_patch), torch.mean(self.common_feature_patch))

    def update_feature_with_common(self, feature_map_origin, common_feature_this, common_feature_patch_this):
        batch_size = feature_map_origin.shape[0]
        channel = feature_map_origin.shape[1]
        assert channel == self.channel
        height = feature_map_origin.shape[2]
        width = feature_map_origin.shape[3]

        feature_map = feature_map_origin
        feature_map_vector = feature_map.reshape(batch_size, channel, height*width).permute(0, 2, 1) ##b hw c
        feature_map_vector = self.norm_query1(feature_map_vector)
        common_map_vector = common_feature_this.reshape(1, self.num_feature, channel).repeat(batch_size, 1, 1) ##.permute(0, 2, 1) # b c n
        common_map_vector = self.norm_query2(common_map_vector).permute(0, 2, 1)
        attention_map = torch.bmm(feature_map_vector, common_map_vector) ## b hw n
        attention_map = torch.nn.Softmax(dim=2)(attention_map)

        value_vector = common_feature_this.reshape(1, self.num_feature, channel).repeat(batch_size, 1, 1) #b n c
        update_vector = torch.bmm(attention_map, value_vector) # b hw c
        update_vector = update_vector.reshape(batch_size, height, width, channel).permute(0, 3, 1, 2) # b c h w


        feature_map_patch = self.feature_unshuffle(feature_map) # b 16c, h/4, w/4
        feature_map_patch = feature_map_patch.reshape(batch_size, channel*self.patch_size*self.patch_size, height*width//(self.patch_size*self.patch_size)).permute(0, 2, 1)
        feature_map_patch = self.norm_query3(feature_map_patch)
        common_map_patch = common_feature_patch_this.reshape(1, self.num_feature, channel*self.patch_size*self.patch_size).repeat(batch_size, 1, 1)## .permute(0, 2, 1) # b c n
        common_map_patch = self.norm_query4(common_map_patch).permute(0, 2, 1)
        attention_map_patch = torch.bmm(feature_map_patch, common_map_patch) ## b hw n
        attention_map_patch = torch.nn.Softmax(dim=2)(attention_map_patch)
        
        value_patch = common_feature_patch_this.reshape(1, self.num_feature, channel*self.patch_size*self.patch_size).repeat(batch_size, 1, 1) #b n c
        update_map = torch.bmm(attention_map_patch, value_patch) # b hw c
        update_map = update_map.reshape(batch_size, height//self.patch_size, width//self.patch_size, channel*self.patch_size*self.patch_size).permute(0, 3, 1, 2) # b c h w
        update_map = self.feature_shuffle(update_map)
        
        fusion_out = update_vector+update_map
        feature_map_origin = feature_map_origin + fusion_out
        return feature_map_origin


    def update_common_feature(self, feature_map):
        height = feature_map.shape[2]
        width = feature_map.shape[3]
        random_index_height = random.randint(0, height-1)
        random_index_width = random.randint(0, width-1)

        random_index_height_patch = random.randint(0, height-self.patch_size)
        random_index_width_patch = random.randint(0, width-self.patch_size)
        
        choosen_feature_vector = feature_map[:, :, random_index_height, random_index_width]
        choosen_feature_map = feature_map[:, :, random_index_height_patch:random_index_height_patch+self.patch_size, random_index_width_patch:random_index_width_patch+self.patch_size]
        
        batch_size = choosen_feature_vector.shape[0]
        common_feature_vector = self.common_feature.reshape(self.num_feature, 1, self.channel).repeat(1, batch_size, 1)
        common_feature_vector = common_feature_vector.reshape(self.num_feature*batch_size, self.channel, 1, 1)
        choosen_feature_vector = choosen_feature_vector.reshape(1, batch_size, self.channel).repeat(self.num_feature, 1, 1)
        choosen_feature_vector = choosen_feature_vector.reshape(self.num_feature*batch_size, self.channel, 1, 1)

        index_array = torch.LongTensor([x for x in range(self.num_feature)]).to(feature_map.device)
        channel = feature_map.shape[1]
        position_embedding = self.embedding(index_array).reshape(self.num_feature, 1, channel).repeat(1, batch_size, 1)
        position_embedding = position_embedding.reshape(self.num_feature*batch_size, channel, 1, 1)

        update_vector = self.update_feature(torch.cat([common_feature_vector, choosen_feature_vector, position_embedding], dim=1))
        update_vector = update_vector.reshape(self.num_feature, batch_size, self.channel)
        update_vector = torch.mean(update_vector, dim=1)
        common_feature_this = self.norm_feature(update_vector)

        #################################################################################
        common_feature_map = self.common_feature_patch.reshape(self.num_feature, 1, self.channel, self.patch_size, self.patch_size).repeat(1, batch_size, 1, 1, 1)
        common_feature_map = common_feature_map.reshape(self.num_feature*batch_size, self.channel, self.patch_size, self.patch_size)
        choosen_feature_map = choosen_feature_map.reshape(1, batch_size, self.channel, self.patch_size, self.patch_size).repeat(self.num_feature, 1, 1, 1, 1)
        choosen_feature_map = choosen_feature_map.reshape(self.num_feature*batch_size, self.channel, self.patch_size, self.patch_size)

        position_embedding2 = position_embedding.repeat(1, 1, self.patch_size, self.patch_size)

        update_map = self.update_feature_patch(torch.cat([common_feature_map, choosen_feature_map, position_embedding2], dim=1))
        update_map = update_map.reshape(self.num_feature, batch_size, self.channel, self.patch_size, self.patch_size)
        update_map = torch.mean(update_map, dim=1)
        common_feature_patch_this = self.norm_patch(update_map)
        return common_feature_this, common_feature_patch_this

    def propagate_feature(self, common_feature_this, common_feature_patch_this):

        choosen_list = [random.randint(0, self.num_feature-1) for x in range(self.num_feature)]
        choosen_list2 = [random.randint(0, self.num_feature-1) for x in range(self.num_feature)]
        choosen_feature_vector = common_feature_this[choosen_list, :].reshape(self.num_feature, self.channel, 1, 1)
        choosen_feature_map = common_feature_patch_this[choosen_list2, :, :, :]
        
        index_array = torch.LongTensor([x for x in range(self.num_feature)]).to(common_feature_this.device)
        channel = common_feature_this.shape[1]
        position_embedding = self.embedding(index_array).reshape(self.num_feature, channel, 1, 1)
        index_array2 = torch.LongTensor(choosen_list).to(common_feature_this.device)
        position_embeddingxx = self.embedding(index_array2).reshape(self.num_feature, channel, 1, 1)

        common_feature_vector = common_feature_this.reshape(self.num_feature, self.channel, 1, 1)
        update_feature_vector = self.propagate(torch.cat([common_feature_vector, choosen_feature_vector, position_embedding, position_embeddingxx], dim=1))
        update_feature_vector = update_feature_vector.reshape(self.num_feature, self.channel)
        common_feature_this = self.norm_feature2(update_feature_vector)

        common_feature_map = common_feature_patch_this
        position_embedding2 = position_embedding.repeat(1, 1, self.patch_size, self.patch_size)
        position_embeddingxx2 = position_embeddingxx.repeat(1, 1, self.patch_size, self.patch_size)
        update_feature_map = self.propagate_patch(torch.cat([common_feature_map, choosen_feature_map, position_embedding2, position_embeddingxx2], dim=1))
        common_feature_patch_this = self.norm_patch2(update_feature_map)
        return common_feature_this, common_feature_patch_this

    def forward_feature(self, x_center, mask):
        with torch.no_grad():
            L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
            L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
            L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))
            fea = self.feature_extraction(L1_fea_3)
        return fea

    def decode_image(self, fea, mask, L1_fea_3, L1_fea_2, L1_fea_1, x_center):
        fea_light = self.recon_trunk_light(fea)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        fea = self.lrelu(self.HRconv(out_noise))
        
        out_noise = self.conv_last(fea)
        out_noise = out_noise + x_center
        return out_noise


    def forward(self, x, mask=None, x_gt=None):
        height_ori = x.shape[2]
        width_ori = x.shape[3]
        height_new = height_ori // 16 * 16
        width_new = width_ori // 16 * 16
        x = F.interpolate(x, size=[height_new, width_new], mode='bilinear')
        mask = F.interpolate(mask, size=[height_new, width_new], mode='bilinear')

        x_center = x

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)

        #####################################################################################################################
        if self.training:
            out_noise1 = self.decode_image(fea, mask, L1_fea_3, L1_fea_2, L1_fea_1, x_center)
            common_feature_this, common_feature_patch_this = self.update_common_feature(fea)
            common_feature_this, common_feature_patch_this = self.propagate_feature(common_feature_this, common_feature_patch_this)
            fea = self.update_feature_with_common(fea, common_feature_this, common_feature_patch_this)
            self.common_feature.data = common_feature_this.clone().detach()
            self.common_feature_patch.data = common_feature_patch_this.clone().detach()
        else:
            fea = self.update_feature_with_common(fea, self.common_feature, self.common_feature_patch)
        #####################################################################################################################

        fea_light = self.recon_trunk_light(fea)
        
        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        xs = np.linspace(-1, 1, fea.size(3) // 4)
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)

        channel = fea.shape[1]
        mask = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask) + fea_light * mask

        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        fea = self.lrelu(self.HRconv(out_noise))
        
        out_noise = self.conv_last(fea)
        out_noise = out_noise + x_center

        #####################################################################################################################
        if self.training:
            out_noise = F.interpolate(out_noise, size=[height_ori, width_ori], mode='bilinear')
            out_noise1 = F.interpolate(out_noise1, size=[height_ori, width_ori], mode='bilinear')
            return out_noise, out_noise1
        else:
            out_noise = F.interpolate(out_noise, size=[height_ori, width_ori], mode='bilinear')
            return out_noise
        #####################################################################################################################

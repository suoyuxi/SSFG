import os
import cv2
import copy
import torch
import random
import numpy as np
import scipy.io as scio
from scipy import signal, interpolate
from torch.utils.data import Dataset

import xml
import xml.dom.minidom
import xml.etree.ElementTree as ET

class predst(Dataset):

    def __init__(self,sl_dir="/workspace/VOC-SL/trainval/SLCMats/",fsi_dir="/workspace/VOC-FSI/trainval/SLCMats/"):
        super(predst, self).__init__()

        sl_list = [ os.path.join(sl_dir, mat) for mat in os.listdir(sl_dir) ]
        fsi_list = [ os.path.join(fsi_dir, mat) for mat in os.listdir(fsi_dir) ]

        # self.slc_list = sl_list + fsi_list
        self.slc_list = sl_list

    def __len__(self):
        return len(self.slc_list)

    def __getitem__(self, i):

        # get img
        slc_path = self.slc_list[i]
        mat = scio.loadmat(slc_path)
        img = mat['data']

        # time domain
        img_t = self.RayleighQuan(img)
        # cv2.imwrite('/workspace/mmrotate/pretrain/img.png', np.uint8((img[0]+1.0)*255/2))

        # mca
        if 'VOC-SL' in slc_path:
            img = self.mca(img)
        elif 'VOC-FSI' in slc_path:
            img = self.mca(img, look_num=9, stride=50, aperture_size=500, rm_taylor=26, ad_taylor=30)

        # get gt
        xml_path = slc_path.replace('SLCMats', 'METAXmls')
        xml_path = xml_path.replace('.mat', '.xml')
        gt = self.get_gt(xml_path)
        # cv2.imwrite('/workspace/mmrotate/pretrain/map.png', np.uint8(gt*255))
        gt = np.expand_dims(gt, axis=0)
        
        return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(gt).type(torch.FloatTensor), img_t

    def RayleighQuan(self, SLCData, SelRatio=7.5, ScaRatio=4.5, random_range=0.3):

        random_factor = np.random.uniform(low=-random_range, high=random_range)
        SelRatio = SelRatio + random_factor
        ScaRatio = ScaRatio + random_factor

        ImgData = np.abs(SLCData)
        RetData = np.zeros( ImgData.shape )

        hist = cv2.calcHist(ImgData.astype(np.uint16), [0], None, [65536], [1,65536])
        Peak = np.argmax(hist)
        # top trunction
        MaxBound = np.floor( (Peak+1) * (SelRatio+1) )
        MaxBoundIndex = ( ImgData > MaxBound ).astype(np.float)
        ImgData = ImgData*(1-MaxBoundIndex) + MaxBound*MaxBoundIndex
        # part division
        PartValue = np.floor( (Peak+1) * ScaRatio )
        SmallPartIndex = ( ImgData <= PartValue ).astype(np.float)
        LargePartIndex = 1 - SmallPartIndex
        RetData = (ImgData*127.0/PartValue)*SmallPartIndex + ((ImgData-PartValue-1)*127.0/(MaxBound-PartValue-1)+128.0)*LargePartIndex
        RetData = (RetData - 127.5) / 127.5

        imgs = np.zeros([3,1024,1024], dtype=np.float32)
        imgs[0,:,:] = RetData
        imgs[1,:,:] = RetData
        imgs[2,:,:] = RetData

        return imgs

    def mca(self, img_slc, look_num=9, stride=50, aperture_size=500, rm_taylor=24, ad_taylor=30, random_factor=0.1):
        
        random_range = int( random_factor * aperture_size )
        random_range = np.random.randint(-random_range, random_range)
        aperture_size = aperture_size + random_range
        edge = ( 1024 - aperture_size - (look_num-1) * stride ) // 2
        if edge < 0:
            edge = 0
            aperture_size = 1024 - (look_num-1) * stride

        # multi-chromatic decomposition
        spectrum = np.fft.fft(img_slc, axis=1) # fft
        spectrum = np.fft.fftshift(spectrum, axes=1) # fftshift along the range

        taylor_window = signal.windows.taylor(1024, nbar=4, sll=rm_taylor, norm=True, sym=True) # calculate taylor window along the range
        remove_taylor = 1.0 / taylor_window 
        spectrum_corr = spectrum * remove_taylor # remove the taylor window

        extra_taylor_window = signal.windows.taylor(aperture_size, nbar=4, sll=ad_taylor, norm=True, sym=True) # apply extra taylor window for each sub-band
        aperture_group = np.zeros([1024,1024,look_num], dtype=np.complex128)
        img_group = np.zeros([1024,1024,look_num], dtype=np.float32)
        for look_cnt in range(look_num):
            st = 62 + look_cnt * stride
            ed = st + aperture_size
            aperture_group[ :, st:ed, look_cnt ] = spectrum_corr[:, st:ed] * extra_taylor_window
            img_group[ :, :, look_cnt ] = np.abs( np.fft.ifft( aperture_group[ :, :, look_cnt ], axis=1 ) )

        # pca
        sample = img_group.reshape(-1, look_num) # reshape
        sample_std = ( sample - np.mean( sample, axis=0 ) ) / np.std( sample, axis=0 ) # normalization
        cov_matrix = np.cov(sample_std.T) # covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix) # get eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1] # sort eigenvalues
        selected_eieigenvector = eigenvectors[:, idx[0]] # get first eieigenvector
        sample_pca = sample_std.dot(selected_eieigenvector) # project

        # Normalization
        img_pca = sample_pca.reshape(1024,1024)
        mu = np.mean(img_pca)
        var = np.std(img_pca)

        img_mca = (img_pca - mu) / var
        img_mca[ img_pca>1.0 ] = 1.0
        img_mca[ img_pca<-1.0 ] = -1.0

        # triple-channel
        imgs_mca = np.zeros([3,1024,1024], dtype=np.float32)
        imgs_mca[0,:,:] = img_mca
        imgs_mca[1,:,:] = img_mca
        imgs_mca[2,:,:] = img_mca

        # img = -img_mca
        # img[img<0.0] = 0.0
        # img = img * 255.0 
        # cv2.imwrite('./nmca.png', img.astype(np.uint8))

        # img = img_mca
        # img[img<0.0] = 0.0
        # img = img * 255.0 
        # cv2.imwrite('./pmca.png', img.astype(np.uint8))

        return imgs_mca.astype(np.float32)

    def band_trunc(self, img):
        
        spectrum = np.fft.fft(img, axis=1)
        spectrum = np.fft.fftshift(spectrum, axes=1)

        band_width = random.sample(range(400,1024,2),1)[0]
        sub_aperture = np.zeros([1024,1024], dtype=np.complex128)
        sub_img = np.zeros([1024,1024], dtype=np.float32)
        
        st = 512 - band_width // 2
        ed = st + band_width
        sub_aperture[ :, st:ed ] = spectrum[:, st:ed]
        sub_img = np.abs( np.fft.ifft( sub_aperture, axis=1 ) )
        
        # quantization
        mu = np.mean(sub_img)
        var = np.std(sub_img)
        sub_img = (sub_img - mu) / var
        sub_img[sub_img>1.0] = 1.0
        sub_img[sub_img<-1.0] = -1.0

        sub_imgs = np.zeros([3,1024,1024], dtype=np.float32)
        sub_imgs[0,:,:] = sub_img
        sub_imgs[1,:,:] = sub_img
        sub_imgs[2,:,:] = sub_img

        return sub_imgs

    def get_gt(self, xml_path):

        tree = ET.parse(xml_path)
        objs = tree.findall('object')

        polygons = []
        gaussian = np.zeros([1024, 1024])
        for idx, obj in enumerate(objs):
            polygon_node = obj.find('polygon')
            x1 = float(polygon_node.find('x1').text)
            y1 = float(polygon_node.find('y1').text)
            x2 = float(polygon_node.find('x2').text)
            y2 = float(polygon_node.find('y2').text)
            x3 = float(polygon_node.find('x3').text)
            y3 = float(polygon_node.find('y3').text)
            x4 = float(polygon_node.find('x4').text)
            y4 = float(polygon_node.find('y4').text)
            polygons.append( ( (int(x1),int(y1)),(int(x2),int(y2)),(int(x3),int(y3)),(int(x4),int(y4)) ) )

            sigmax = np.power( np.power((x2-x1),2) + np.power((y2-y1),2), 0.5 ) / 3 + 1e-6 
            sigmay = np.power( np.power((x3-x2),2) + np.power((y3-y2),2), 0.5 ) / 3 + 1e-6 

            if x2-x1 == 0:
                Theta = np.pi/2
            else:
                Theta = np.arctan((y2-y1) / (x2-x1))
            cosTheta = np.cos(Theta)
            sinTheta = np.sin(Theta)

            mux = (x1+x2+x3+x4) / 4.0
            muy = (y1+y2+y3+y4) / 4.0

            a = cosTheta**2/sigmax**2 + sinTheta**2/sigmay**2
            b = sinTheta**2/sigmax**2 + cosTheta**2/sigmay**2
            c = 2*cosTheta*sinTheta*(1/sigmax**2 - 1/sigmay**2)

            x = np.arange(0, 1024)
            y = np.arange(0, 1024)
            X, Y = np.meshgrid(x, y)

            gaussian = self.gaussian_function(X, Y, mux, muy, a, b, c) + gaussian

        return gaussian / np.max(gaussian)
    
    def gaussian_function(self, x, y, mux, muy, a, b, c):
        gaussian = np.exp( -1/2 * ( a*(x - mux)**2 + b*(y-muy)**2 + c*(x-mux)*(y-muy) ) ) 
        return gaussian / ( np.max( gaussian ) + 1e-6 )

if __name__ == '__main__':

    dst = predst()
    print(dst[6546][0].size(), dst[6546][1].size())
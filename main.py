#!/usr/bin/env python
# coding: utf-8

import argparse, os, math
import numpy
from PIL import Image
import piexif
import cv2
import chainer
from chainer import cuda
from chainer import serializers
import network

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='File path of input image.', default='./data/Forest.png')
parser.add_argument('-o', help='Output directory.', default='./results')
parser.add_argument('-gpu', help='GPU device specifier. Two GPU devices must be specified, such as 0,1.', default='-1')
parser.add_argument('-dm', help='File path of a downexposure model.', default='./models/downexposure_model.chainer')
parser.add_argument('-um', help='File path of a upexposure model.', default='./models/upexposure_model.chainer')
args = parser.parse_args()

f_path = args.i
model_path_list = [args.dm, args.um]
base_outdir_path = args.o
gpu_list = []
if args.gpu != '-1':
    for gpu_num in (args.gpu).split(','):
        gpu_list.append(int(gpu_num))

'Estimate up-/donwn-exposed images'
model_list = [network.CNNAE3D512(), network.CNNAE3D512()]
xp = cuda.cupy if len(gpu_list) > 0 else numpy
if len(gpu_list) > 0:
    for i, gpu in enumerate(gpu_list):
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        model_list[i].to_gpu()
        serializers.load_npz(model_path_list[i], model_list[i])
else:
    for i in range(2):
        serializers.load_npz(model_path_list[i], model_list[i])

def estimate_images(input_img, model):

    model.train_dropout = False
    input_img_ = (input_img.astype(numpy.float32)/255.).transpose(2,0,1)
    input_img_ = chainer.Variable(xp.array([input_img_]), volatile=True)
    res  = model(input_img_).data[0]
    if len(gpu_list)>0:
        res = cuda.to_cpu(res)

    out_img_list = list()
    for i in range(res.shape[1]):
        out_img = (255.*res[:,i,:,:].transpose(1,2,0)).astype(numpy.uint8)
        out_img_list.append(out_img)

    return out_img_list

img = cv2.imread(f_path)
out_img_list = list()
if len(gpu_list)>0:
    for i, gpu in enumerate(gpu_list):
        cuda.get_device(gpu).use()
        out_img_list.extend(estimate_images(img, model_list[i]))
        if i == 0:
            out_img_list.reverse()
            out_img_list.append(img)
else:
    for i in range(2):
        out_img_list.extend(estimate_images(img, model_list[i]))
        if i == 0:
            out_img_list.reverse()
            out_img_list.append(img)

'Select and Merge'
threshold = 64
stid = 0
prev_img = out_img_list[8].astype(numpy.float32)
out_img_list.reverse()
for out_img in out_img_list[9:]:
    img = out_img.astype(numpy.float32)
    if (img>(prev_img+threshold)).sum() > 0:
        break
    prev_img = img[:,:,:]
    stid+=1

edid = 0
prev_img = out_img_list[8].astype(numpy.float32)
out_img_list.reverse()
for out_img in out_img_list[9:]:
    img = out_img.astype(numpy.float32)
    if (img<(prev_img-threshold)).sum() > 0:
        break
    prev_img = img[:,:,:]
    edid+=1

out_img_list = out_img_list[8-stid:9+edid]
outdir_path = base_outdir_path+'/'+f_path.split('/')[-1]
os.system('mkdir ' + outdir_path)

exposure_times = list()
lowest_exp_time = 1/1024.
for i in range(len(out_img_list)):
    exposure_times.append(lowest_exp_time*math.pow(math.sqrt(2.),i))
exposure_times = numpy.array(exposure_times).astype(numpy.float32)

for i, out_img in enumerate(out_img_list):
    numer, denom = float(exposure_times[i]).as_integer_ratio()
    if int(math.log10(numer)+1)>9:
        numer = int(numer/10*(int(math.log10(numer)+1)-9))
        denom = int(denom/10*(int(math.log10(numer)+1)-9))
    if int(math.log10(denom)+1)>9:
        numer = int(numer/10*(int(math.log10(denom)+1)-9))
        denom = int(denom/10*(int(math.log10(denom)+1)-9))
    exif_ifd = {piexif.ExifIFD.ExposureTime:(numer,denom)}
    exif_dict = {"Exif":exif_ifd}
    exif_bytes = piexif.dump(exif_dict)
    out_img_ = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    out_img_pil = Image.fromarray(out_img_)
    out_img_pil.save(outdir_path+"/exposure_"+str(i)+".jpg", exif=exif_bytes)

merge_debvec = cv2.createMergeDebevec()
hdr_debvec = merge_debvec.process(out_img_list, times=exposure_times.copy())
cv2.imwrite(outdir_path+'/MergeDebevec.hdr', hdr_debvec)

merge_mertens = cv2.createMergeMertens(1.,1.,1.e+38)
res_mertens = merge_mertens.process(out_img_list)
cv2.imwrite(outdir_path+'/MergeMertens.hdr', res_mertens)
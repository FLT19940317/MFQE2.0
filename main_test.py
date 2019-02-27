import numpy as np
import scipy.io as sio
import yuv_process, test_DSCNN, test_MFCNN
import cv2
import os
from skimage.measure import compare_psnr

### option
output_bmp = False # output enhanced frames (time-consuming and requires space)
calculate_YPSNR = True # calculate Y-PSNR (requires raw video and its path)


### video information
QP = 37
WIDTH = 416
HEIGHT = 240
start_frame = 0
NUM_FRAMES = 500 # count from start frame
raw_path = './Input/BasketballPass_416x240_500.yuv' # for calculating Y-PSNR
cmp_path = './Input/BasketballPass_416x240_500_qp37.yuv'
label_path = './Input/label_BasketballPass_416x240_500_qp37.mat'
output_dir = './Output/QP37'

if output_bmp:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

print("\nInformation check:")
print("     %d x %d - %d frames (start from frame %d) - QP%d" % (WIDTH, HEIGHT, NUM_FRAMES, start_frame, QP))
print("     compressed video path: %s" % cmp_path)
print("     label path:            %s" % label_path)
if output_bmp:
    print("     output dir:            %s" % output_dir)
if calculate_YPSNR:
    print("     raw video path:        %s" % raw_path)

### import Y,U,V
print("\nimport cmp Y,U,V...")
Y,U,V = yuv_process.yuv_import(cmp_path,(HEIGHT,WIDTH),NUM_FRAMES,start_frame,israw=False)


### import PQF label
print("\nimport PQF label...")
label = sio.loadmat(label_path)
label = label['label']
dims = np.shape(label)
if dims[0]<dims[1]:
    label = np.transpose(label)


### check PQF label
PQF_indices,waste = np.where(label==1)
min_PQF_index = np.min(PQF_indices)
max_PQF_index = np.max(PQF_indices)
tmp1 = 0
tmp2 = 0
if min_PQF_index != start_frame:
    tmp1 = min_PQF_index-start_frame
    if tmp1 == 1:
        print("Warning: The first non-PQF has no 'previous' PQF, and thus can't be enhanced by MF-CNN.")
    else:
        print("Warning: The first %d non-PQFs have no 'previous' PQFs, and thus can't be enhanced by MF-CNN." % tmp1)
    label[0:min_PQF_index-1] = 1
if max_PQF_index != (start_frame + NUM_FRAMES - 1):
    tmp2 = start_frame+NUM_FRAMES-1-max_PQF_index
    if tmp2 == 1:
        print("Warning: The last non-PQF has no 'subsequent' PQF, and thus can't be enhanced by MF-CNN.")
    else:
        print("Warning: The last %d non-PQFs have no 'subsequent' PQFs, and thus can't be enhanced by MF-CNN." % tmp2)
    label[max_PQF_index+1:] = 1
if (min_PQF_index != start_frame) or (max_PQF_index != (start_frame + NUM_FRAMES - 1)):
    if (tmp1 + tmp2) == 1:
        print("         For simplicity, we enhance the non-PQF by DS-CNN.")
    else:
        print("         For simplicity, we enhance these non-PQFs by DS-CNN.")


### find previous and subsequent PQFs for each non-PQF
PQF_indices,waste = np.where(label==1)
PQF_indices = [int(a) for a in PQF_indices]
Non_PQF_indices,waste = np.where(label==0)
Non_PQF_indices = [int(a) for a in Non_PQF_indices]
pre_PQF_indices = np.array(Non_PQF_indices, copy=True) # initialize
sub_PQF_indices = np.array(Non_PQF_indices, copy=True) # initialize
for ite_nonPQF in range(len(Non_PQF_indices)):
    Non_PQF_index = Non_PQF_indices[ite_nonPQF]
    for ite_PQF in range(len(PQF_indices)-1):
        if (Non_PQF_index>PQF_indices[ite_PQF]) and (Non_PQF_index<PQF_indices[ite_PQF+1]):
            pre_PQF_indices[ite_nonPQF] = PQF_indices[ite_PQF]
            sub_PQF_indices[ite_nonPQF] = PQF_indices[ite_PQF+1]
            break


### enhance PQFs
print("\n\nenhance PQFs by DS-CNN...")
enhanced_PQFs, average_fps = test_DSCNN.enhance(QP,Y[PQF_indices])
print("ave. fps: %.1f" % average_fps)


### enhance non-PQFs
print("\nenhance non-PQFs by MF-CNN...")
enhanced_nonPQFs, average_fps = test_MFCNN.enhance(QP,Y,Non_PQF_indices,pre_PQF_indices,sub_PQF_indices)
print("ave. fps: %.1f" % average_fps)


### combine PQFs and non-PQFs
Y_enhanced = np.zeros(Y.shape)
Y_enhanced[PQF_indices] = enhanced_PQFs
Y_enhanced[Non_PQF_indices] = enhanced_nonPQFs


### export bmp
if output_bmp:
    print("\nexport enhanced frames (.bmp)...")
    nfs = len(Y_enhanced)
    for ite_frame in range(nfs):
        enhanced = yuv_process.YUV2RGB(Y_enhanced[ite_frame]*255.0,U[ite_frame],V[ite_frame])
        output_path = output_dir+'/'+str(ite_frame)+'_enhanced.bmp'
        cv2.imwrite(output_path, enhanced)
        print("\r"+str(ite_frame+1)+" | "+str(nfs), end="", flush=True)
    print("")


### calculate delta Y-PSNR
if calculate_YPSNR:
    print("\nimport raw Y...")
    Y_raw = yuv_process.yuv_import(raw_path,(HEIGHT,WIDTH),NUM_FRAMES,start_frame,israw=True)

    print("\ncalculate Y-PSNR...")
    psnr_ori = 0
    psnr_enh = 0
    for ite_frame in range(nfs):
        raw = Y_raw[ite_frame]
        psnr_ori += compare_psnr(Y[ite_frame],raw,data_range=255)
        psnr_enh += compare_psnr(Y_enhanced[ite_frame]*255.0,raw,data_range=255)
        print("\r"+str(ite_frame+1)+" | "+str(nfs), end="", flush=True)
    print("")

    print("\nave. original PSNR: %.3f" % (psnr_ori/nfs))
    print("ave. present PSNR:  %.3f" % (psnr_enh/nfs))
    print("ave. delta PSNR:    %.3f" % ((psnr_enh-psnr_ori)/nfs))

print("\nDone!")

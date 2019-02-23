from numpy import *
import cv2

def yuv_import(video_path,dims,nfs,startfrm,israw):

    fp = open(video_path,'rb')

    blk_size = int(prod(dims) * 3 / 2) # 4:2:2
    fp.seek(blk_size*startfrm,0)

    d0 = dims[0]
    d1 = dims[1]
    d01 = dims[0] // 2
    d11 = dims[1] // 2

    Yt = zeros([d0,d1],uint8) # 0-255

    if not israw: # cmp
        Ut = zeros([d01,d11],uint8)
        Vt = zeros([d01,d11],uint8)

        for ite_frame in range(nfs):

            for m in range(d0):
                for n in range(d1):
                    Yt[m,n] = ord(fp.read(1))
            for m in range(d01):
                for n in range(d11):
                    Ut[m,n] = ord(fp.read(1))
            for m in range(d01):
                for n in range(d11):
                    Vt[m,n] = ord(fp.read(1))

            if ite_frame == 0:
                Y = Yt[newaxis,:,:]
                U = Ut[newaxis,:,:]
                V = Vt[newaxis,:,:]
            else:
                Y = vstack((Y,Yt[newaxis,:,:]))
                U = vstack((U,Ut[newaxis,:,:]))
                V = vstack((V,Vt[newaxis,:,:]))

            print("\r"+str(ite_frame+1)+" | "+str(nfs), end="", flush=True)

        print("")
        fp.close()
        return Y,U,V

    if israw:

        for ite_frame in range(nfs):

            for m in range(d0):
                for n in range(d1):
                    Yt[m,n] = ord(fp.read(1))
            for m in range(d01):
                for n in range(d11):
                    fp.read(1)
            for m in range(d01):
                for n in range(d11):
                    fp.read(1)

            if ite_frame == 0:
                Y = Yt[newaxis,:,:]
            else:
                Y = vstack((Y,Yt[newaxis,:,:]))
            
            print("\r"+str(ite_frame+1)+" | "+str(nfs), end="", flush=True)

        print("")

        fp.close()
        return Y


def YUV2RGB(Y,U,V):

    Y = uint8(clip(Y,0,255))
    U = uint8(clip(U,0,255))
    V = uint8(clip(V,0,255))

    # YUV420p
    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # merge
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    # convert
    RGB = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

    return RGB

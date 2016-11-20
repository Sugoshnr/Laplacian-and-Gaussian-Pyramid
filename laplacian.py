from cv2 import *
import numpy as np
import math
from matplotlib import pyplot as plt


path="Lenna.png"


def gauss(img,i,j):

    sum1=0.0;
    if(j>1 and i>1):
        sum1+=float(img[i-1][j-1])/16
    if(j>1):
        sum1+=float(img[i][j-1])/8
    if(i<len(img)-1 and j>1):
        sum1+=float(img[i+1][j-1])/16
    if(i>1):
        sum1+=float(img[i-1][j])/8
    sum1+=float(img[i][j])/4
    if(i<len(img)-1):
        sum1+=float(img[i+1][j])/8
    if(i>1 and j<len(img[0]-1)):
        sum1+=float(img[i-1][j+1])/16
    if(j<len(img[0]-1)):
        sum1+=float(img[i][j+1])/8
    if(i<len(img)-1 and j<len(img[0]-1)):
        sum1+=float(img[i+1][j+1])/16
    
    return sum1;


def down_sample(img):
    rows,cols=img.shape
    gauss_img=img.copy()
    #imshow("Original",img)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            gauss_img[i][j]=gauss(img,i,j)
    gauss_img=np.array(gauss_img,dtype='uint8')
    img=np.array(img,dtype='uint8')
    #imshow("Gauss",gauss_img1)
    sub=np.subtract(img,gauss_img)
    #sub=np.array(sub,dtype='uint8')
    #imshow("Sub",sub1)
    return sub,gauss_img


##def up_sample(gauss, lapl):
##    #lapl=resize(lapl,gauss.shape)
##    img=np.add(gauss,lapl)
##    rows, cols=img.shape
##    up_img=[[0.0 for k in range(rows*2)] for l in range(cols*2)]
##    up_img=np.array(up_img,dtype='uint8')
##    k=0;l=0;
##    for i in range(rows):
##        for j in range(cols):
##            up_img[k][l]=img[i][j]
##            if j < cols-1:
##                #up_img[k][l+1]=(float(img[i][j])+float(img[i][j+1]))/float(2);
##                up_img[k][l+1]=img[i][j]
##            l=l+2;
##        #print up_img[k]
##        k=k+2;
##        l=0;
##    up_img=np.asarray(up_img)
##    up_img=np.array(up_img,dtype='uint8')
##    k=1;l=0;
##    for i in range(rows*2):
##        for j in range(cols*2):
##            if k < rows*2:
##                up_img[k][j]=(float(up_img[k-1][j])+float(up_img[k+1][j]))/float(2);
##        #print up_img[i]
##        k=k+2;
##    up_img=np.array(up_img,dtype='uint8')
##    return up_img



def up_sample_extra(gauss):
    img=gauss.copy()
    rows, cols=img.shape
    up_img=[[0.0 for k in range(rows*2)] for l in range(cols*2)]
    up_img=np.array(up_img,dtype='uint8')
    k=0;l=0;
    for i in range(rows):
        for j in range(cols):
            up_img[k][l]=img[i][j]
            if j < cols:
                #up_img[k][l+1]=(float(img[i][j])+float(img[i][j+1]))/float(2);
                up_img[k][l+1]=img[i][j]
            l=l+2;
        #print up_img[k]
        k=k+2;
        l=0;
    up_img=np.asarray(up_img)
    up_img=np.array(up_img,dtype='uint8')
    k=1;l=0;
    for i in range(rows*2):
        for j in range(cols*2):
            if k < rows*2:
                #up_img[k][j]=(float(up_img[k-1][j])+float(up_img[k+1][j]))/float(2);
                up_img[k][j]=up_img[k-1][j]
        #print up_img[i]
        k=k+2;
    up_img=np.array(up_img,dtype='uint8')
    return up_img
            

def mse(img1,img2):
    rows,cols=img1.shape
    mse_val=0;
    for i in range(rows):
        for j in range(cols):
            mse_val+=(img1[i][j]-img2[i][j])**2
    return mse_val


img1=imread(path,0)

#img1=copyMakeBorder(img1,1,1,1,1,BORDER_CONSTANT,value=0)

#imshow("Gauss1",gauss_img)

sub_sample1,gauss_img1=down_sample(img1)
img2=gauss_img1[0::2,0::2]
img_2=img1[0::2,0::2]
sub_sample2,gauss_img2=down_sample(img2)
img3=gauss_img2[0::2,0::2]
img_3=img_2[0::2,0::2]
sub_sample3,gauss_img3=down_sample(img3)
img4=gauss_img3[0::2,0::2]
img_4=img_3[0::2,0::2]
sub_sample4,gauss_img4=down_sample(img4)
img5=gauss_img4[0::2,0::2]
img_5=img_4[0::2,0::2]
sub_sample5,gauss_img5=down_sample(img5)
img6=gauss_img5[0::2,0::2]
img_6=img_5[0::2,0::2]
sub_sample6,gauss_img6=down_sample(img6)



gauss_img2_enlarged=up_sample_extra(gauss_img2)
l1=np.subtract(img1,gauss_img2_enlarged)
gauss_img3_enlarged=up_sample_extra(gauss_img3)
l2=np.subtract(gauss_img2,gauss_img3_enlarged)
gauss_img4_enlarged=up_sample_extra(gauss_img4)
l3=np.subtract(gauss_img3,gauss_img4_enlarged)
gauss_img5_enlarged=up_sample_extra(gauss_img5)
l4=np.subtract(gauss_img4,gauss_img5_enlarged)
gauss_img6_enlarged=up_sample_extra(gauss_img6)
l5=np.subtract(gauss_img5,gauss_img6_enlarged)


gauss_reconst_img5=np.add(l5,gauss_img6_enlarged)
gauss_reconst_img5=up_sample_extra(gauss_reconst_img5)
gauss_reconst_img4=np.add(l4,gauss_reconst_img5)
gauss_reconst_img4=up_sample_extra(gauss_reconst_img4)
gauss_reconst_img3=np.add(l3,gauss_reconst_img4)
gauss_reconst_img3=up_sample_extra(gauss_reconst_img3)
gauss_reconst_img2=np.add(l2,gauss_reconst_img3)
gauss_reconst_img2=up_sample_extra(gauss_reconst_img2)
gauss_reconst_img1=np.add(l1,gauss_reconst_img2)


mse_val=mse(gauss_reconst_img1,img1)
print "Mean Square Error= %d"%int(mse_val)


plt.figure(1)
plt.subplot(321),plt.imshow(l1-128,cmap='gray')
plt.title("l1")
plt.subplot(322),plt.imshow(l2-128,cmap='gray')
plt.title("l2")
plt.subplot(323),plt.imshow(l3-128,cmap='gray')
plt.title("l3")
plt.subplot(324),plt.imshow(l4-128,cmap='gray')
plt.title("l4")
plt.subplot(325),plt.imshow(l5-128,cmap='gray')
plt.title("l5")



plt.figure(2)
plt.subplot(321),plt.imshow(gauss_reconst_img5,cmap='gray')
plt.title("Reconstructed Image 5")
plt.subplot(322),plt.imshow(gauss_reconst_img4,cmap='gray')
plt.title("Reconstructed Image 4")
plt.subplot(323),plt.imshow(gauss_reconst_img3,cmap='gray')
plt.title("Reconstructed Image 3")
plt.subplot(324),plt.imshow(gauss_reconst_img2,cmap='gray')
plt.title("Reconstructed Image 2")
plt.subplot(325),plt.imshow(gauss_reconst_img1,cmap='gray')
plt.title("Reconstructed Image 1")


plt.show()


waitKey(0)
destroyAllWindows()
        

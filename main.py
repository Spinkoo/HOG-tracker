


    

import cv2
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import collections

points=[]

def MS(src,req):
    result=0
    for j in range(0,len(src)):
            for i in range(0,8):

                val1=src[j][i]
                val2=req[j][i]
                result+=(val1-val2)**2
    return result**0.5  
       
def MA(src,req):
    result=0
    for j in range(0,len(src)):            
            result+=(src[j]-req[j])**2
    return result


            
def calculate(start,startj,appendI,appendJ):
    global h
    global w
    histogram = np.zeros((8),np.double)
    for i in range (start,start+appendI):    
        for j in range (startj,startj+appendJ): 
            if(i<lowI or i > highI or j<lowJ or j > highJ):
                histogram[0]+=1
        
            else:
               
                index=int(IConv[i][j])
                if(IConv[i][j]-index>0.5):
                    index+=1
                index=index%8

                histogram[index]+=1

    return histogram;          
                

                
def getHOG(x,y):
    
    #fenetre de comparasion 6*6 block
    val=6
    i=-val

    
    k=0
    l=0
    point=[y,x]
    temp=(calculate(point[1]-int(val/2),point[0]-int(val/2),val,val))
    return temp



    
    
Selected=True
        
def mouseCallback(event,x,y,flags,param):
    global points
    global Selected
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()
        cv2.rectangle(colored,(points[0][0],points[0][1]),(points[1][0],points[1][1]), (0,255,0), lineType=cv2.LINE_AA)
        cv2.imshow("Selected ",colored)
        cv2.destroyWindow('image')
        Selected=False

#init
img=cv2.imread("data/00000000.jpg",0)
colored=cv2.imread("data/00000000.jpg",1)
dim=(80,60)
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouseCallback)
while(Selected):
    cv2.imshow('image',colored)
    cv2.waitKey(1000)
ratio=30
prevX=-1
prevY=-1
angles = []



debutX=int(points[0][0]/4)
debutY=int(points[0][1]/4)
finX=int(points[1][0]/4)
finY=int(points[1][1]/4)



#espace de recherche 
initk=k=10



if(debutX>finX):
    debutX,finX=finX,debutX
if(debutY>finY):
    debutY,finY=finY,debutY
img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

img=img[debutY-k:finY+k,debutX-k:finX+k]

I=np.array(img)
c=1


h,w = np.shape(I)


IConv=np.zeros((np.shape(I)[0],np.shape(I)[1]),np.double)
once = True

I =cv2.GaussianBlur(I,(3,3),0)
gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=1)
                
mag, IConv = cv2.cartToPolar(gx, gy, angleInDegrees=True)

IConv/=45

histogram={}
HOGS1=[]
HOGS2=[]

#IConv=IConv[k:h-k,k:w-k]
#Ic=IConv    
h,w = np.shape(IConv)
h1,w1=h-2*k,w-2*k
lowJ=lowI=0
highI=h-1
highJ=w-1
for i in range(k,h-k):
    for j in range (k,w-k):      
        HOGS1.append(getHOG(i,j))

image=0
images=[]
colored=[]
print("Loading images...")
while image< 602:
    if image < 10:
        string="data/0000000"
    elif image >=10 and image <100:
        string="data/000000"
    elif image >99:
        string="data/00000"

 

    images.append(cv2.imread(string+str(image)+".jpg",0))
    images[len(images)-1]=cv2.GaussianBlur( images[len(images)-1],(3,3),0)
    images[len(images)-1]=cv2.resize( images[len(images)-1], dim, interpolation = cv2.INTER_AREA)
    colored.append(cv2.imread(string+str(image)+".jpg"))

    image+=1
image=0
prevposx=prevposy=0
while(image<500):
    prevh=h1
    prevw=w1
    i=0
    j=0
    
    hei,wid=np.shape(images[image])
    
    if(debutY-k<0):
        debutY+=k
    if(debutX-k<0):
      debutX+=k
    img=images[image][debutY-k:debutY+prevh+k,debutX-k:debutX+prevw+k]

   
    if(not img.any()):
        print(string+str(image)+".jpg")
        break
        print("error loading")
        continue
    I=np.array(img)
    h,w = np.shape(I)
    IVal=[]
    matchI=-1
    matchJ=-1
    excep=False
    l=-k
    I =cv2.GaussianBlur(I,(3,3),0)

    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=1)
    mag, IConv = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    IConv/=45
    temp={}
    Histos=[]
    count=0
    hack=0
    lowI=0
    lowJ=0
    highJ=w-1
    highI=h-1
    for i in range(0,h):
        temp=[]
        for j in range (0,w):
            lowI=0
            lowJ=0
            highJ=w-1
            highI=h-1
            temp.append(getHOG(i,j))

        IVal.append(temp.copy())


    minim=9999
    while(l<=k):
        m=-k
        while(m<=k):
            sums=0
            ff=0
            try:
                for i in range(l+k,k+prevh+l):
                    for j in range (m+k,k+m+prevw):
                        sums+=MA((IVal[i][j]),HOGS1[ff])
                        ff+=1
            except Exception as e:
                m+=1
                excep=True

                continue

            a=sums**0.5

            if(minim>a):
                minim=a
                matchI=l
                matchJ=m
            HOGS2.clear()
            m+=2
        l+=2
    l=matchI
    m=matchJ
    if excep:
        #regler l'exception en centrant la recherche au milieu de l'limage
        debutY=20
        debutX=25
        excep=False


    cv2.rectangle(colored[image],(4*(m+debutX),4*(l+debutY)),(4*(m+debutX+prevw),4*(l+debutY+prevh)), (0,255,0), lineType=cv2.LINE_AA)


    cv2.imshow("",colored[image])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    debutY+=l
    debutX+=m
    image+=1



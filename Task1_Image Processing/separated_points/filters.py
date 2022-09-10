import matplotlib.pyplot as plt
import numpy as np

img = plt.imread('jaguar.jpg')

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N % 2 > 0:
        # raise ValueError("must be a power of 2")
        Height = int(np.ceil(np.log10(N)/np.log10(2)))
        diff = np.abs(N-Height)
        N = N+diff
    elif N <= 2:
        return dft(x)
    else:
        X_even = fft(x[::2])
         X_odd = fft(x[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])

def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")
        
    N_min = min(N, 2)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
            X_even = X[:, :int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2):]
            terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                            / X.shape[0])[:, None]
            X = np.vstack([X_even + terms * X_odd,
                        X_even - terms * X_odd])
    return X.ravel()

x = np.random.random(410)
y =np.allclose(dft(x), np.fft.fft(x))
z= np.allclose(fft(x), np.fft.fft(x))


def draw_cicle(shape,diamiter):
    '''
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape,dtype=bool)
    center = np.array(TF.shape)/2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy,ix] = (iy- center[0])**2 + (ix - center[1])**2 < diamiter **2
    return(TF)

TFcircleIN   = draw_cicle(shape=img.shape[:2],diamiter=50)
TFcircleOUT  = ~TFcircleIN
#Low pass and High pass filters
fig = plt.figure(figsize=(30,10))
ax  = fig.add_subplot(1,2,1)
im  = ax.imshow(TFcircleIN,cmap="gray")
plt.colorbar(im)
ax  = fig.add_subplot(1,2,2)
im  = ax.imshow(TFcircleOUT,cmap="gray")
plt.colorbar(im)
plt.show()


def Image_transform(Image):
    Image=Image/float(2**8)
    fft_img = np.zeros_like(Image,dtype=complex)
    for ichannel in range(fft_img.shape[2]):
        fft_img[:,:,ichannel] = np.fft.fftshift(np.fft.fft2(Image[:,:,ichannel]))
    return fft_img

def filter_circle(filter_type,Shape,fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2],dtype=complex)
    if filter_type=="low":
        TFcircle=draw_cicle(shape=Shape,diamiter=20)
    else:
        TFcircle=~draw_cicle(shape=Shape,diamiter=5)
    temp[TFcircle] = fft_img_channel[TFcircle]
    return(temp)

def low_pass_filter(image):
    fft_image = Image_transform(image)
    fft_image_filtered_IN = []
    for ichannel in range(fft_image.shape[2]):
        fft_image_channel  = fft_image[:,:,ichannel]
        ## circle IN
        temp = filter_circle("low",image.shape[:2],fft_image_channel)
        fft_image_filtered_IN.append(temp)
    fft_image_filtered_IN = np.array(fft_image_filtered_IN)
    fft_image_filtered_IN = np.transpose(fft_image_filtered_IN,(1,2,0))
    return fft_image_filtered_IN

def High_pass_filter(image):
    fft_image = Image_transform(image)
    fft_image_filtered_OUT = []
    for ichannel in range(fft_image.shape[2]):
        fft_image_channel  = fft_image[:,:,ichannel]
        ## circle OUT
        temp = filter_circle("high",image.shape[:2],fft_image_channel)
        fft_image_filtered_OUT.append(temp) 
    fft_image_filtered_OUT = np.array(fft_image_filtered_OUT)
    fft_image_filtered_OUT = np.transpose(fft_image_filtered_OUT,(1,2,0))
    return fft_image_filtered_OUT

abs_fft_img              = np.abs(Image_transform(img))
abs_fft_img_filtered_IN  = np.abs(low_pass_filter(img))
abs_fft_img_filtered_OUT = np.abs(High_pass_filter(img))

def imshow_fft(absfft):
    magnitude_spectrum = 20*np.log(absfft)
    return(ax.imshow(magnitude_spectrum,cmap="gray"))

fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(15,10))
fontsize = 15 

for ichannel, color in enumerate(["R","G","B"]):
    ax = axs[0,ichannel]
    ax.set_title(color)
    im = imshow_fft(abs_fft_img[:,:,ichannel])
    if ichannel == 0:
        ax.set_ylabel("original DFT",fontsize=fontsize)
    fig.colorbar(im,ax=ax)
    
    
    ax = axs[1,ichannel]
    im = imshow_fft(abs_fft_img_filtered_IN[:,:,ichannel])
    if ichannel == 0:
        ax.set_ylabel("DFT + low pass filter",fontsize=fontsize)
    fig.colorbar(im,ax=ax)
    
    ax = axs[2,ichannel]
    im = imshow_fft(abs_fft_img_filtered_OUT[:,:,ichannel])
    if ichannel == 0:
        ax.set_ylabel("DFT + high pass filter",fontsize=fontsize)   
    fig.colorbar(im,ax=ax)
    



def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)


img_reco              = inv_FFT_all_channel(Image_transform(img))
img_reco_filtered_IN  = inv_FFT_all_channel(low_pass_filter(img))
img_reco_filtered_OUT = inv_FFT_all_channel(High_pass_filter(img))

fig = plt.figure(figsize=(25,18))
ax  = fig.add_subplot(1,3,1)
ax.imshow(np.abs(img_reco))
ax.set_title("original image")

ax  = fig.add_subplot(1,3,2)
ax.imshow(np.abs(img_reco_filtered_IN))
ax.set_title("low pass filter image")


ax  = fig.add_subplot(1,3,3)
ax.imshow(np.abs(img_reco_filtered_OUT))
ax.set_title("high pass filtered image")
plt.show()


import cv2
# import math
# import keras.backend as K
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr

def measures( file1, file2):
    img1 = cv2.imread(file1)
    # y_true = cv2.imread(file1)
    img2 = cv2.imread(file2)
    # y_pred = cv2.imread(file2)
    # img1  = img[:,320:640,:]
    # img2 = img[:,640:,:]
    psnr_val = psnr(img1, img2)
    # max_pixel = 1.0
    # psnr_val= 10.0 * math.log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
    # psnr_perc=(psnr_val/60)*100
    # print("psnr:", psnr_val)
    ssim_val = ssim(img1, img2, gaussian_weights=True, multichannel=True)
    # print("ssim:", ssim_val)
    # psnr_list.append(psnr_val)
    # ssim_list.append(ssim_val)
    # print psnr_val, ssim_val
    return  psnr_val,ssim_val




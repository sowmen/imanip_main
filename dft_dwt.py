import numpy as np
import cv2
import pywt

def rgb2ycbcr(image):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image.dot(xform.T)
    ycbcr[:, :, [1,2]] += 128
    return np.uint8(ycbcr)


def DFT(image):
  fft = cv2.dft(np.float32(image),flags=cv2.DFT_COMPLEX_OUTPUT)
  return np.array([fft[:,:,0], fft[:,:,1]])

def blockDFT(image):
  real = np.full_like(image, 0, dtype='float32')
  imag = np.full_like(image, 0, dtype='float32')
  for i in range(0, image.shape[0], 8):
    for j in range(0, image.shape[1], 8):
      x = i + 8
      y = j + 8
      pat = image[i : x, j : y]
      dft = DFT(pat)
      real_dft = dft[0]
      imag_dft = dft[1]
      row = 0
      col = 0
      for k in range(i, x):
        col = 0
        for l in range(j, y):
          real[k][l] = real_dft[row][col]
          imag[k][l] = imag_dft[row][col]
          col += 1
        row += 1
  ret = np.array([real, imag])
  return ret


def waveletTransform(image):
  output = pywt.dwt2(image, 'haar', mode = 'periodization')
  LL, (LH, HL, HH) = output
  LL = cv2.resize(LL, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
  LH = cv2.resize(LH, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
  HL = cv2.resize(HL, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
  HH = cv2.resize(HH, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
  
  return np.array([HH, HL, LH, LL])


def generate_dft_dwt_vector(image): 
  yCbCr = rgb2ycbcr(image)
  r = yCbCr[:,:,0]
  g = yCbCr[:,:,1]
  b = yCbCr[:,:,2]
  dft_r = blockDFT(r)
  dwt_r = waveletTransform(r)
  dft_g = blockDFT(g)
  dwt_g = waveletTransform(g)
  dft_b = blockDFT(b)
  dwt_b = waveletTransform(b)
  ans = np.concatenate([dft_r, dwt_r, dft_g, dwt_g, dft_b, dwt_b], axis = 0)
  return ans

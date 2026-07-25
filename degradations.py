"""

degradations.py
=================================================================

Description : Module pour ajouter un flou, du bruit, une compression JPEG,
              un filtre 2D sinc, une réduction d'échelle à un lot d'images.

Auteur : Diarimandimby Riantsoa Kanto

Pour le calcul du noyau et l'ajout du filtre 2D sinc, nous utilisons l'implémentation dans ce lien : 
https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

Pour la compression JPEG, nous utilisons l'implémentation suivante :
https://github.com/mlomnitz/DiffJPEG

Date de création : 24/07/2026

Licence : MIT

"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import special
import random
import math
from DiffJPEG import DiffJPEG

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def generalGaussianKernel(kernel_size, sigma1, sigma2, theta, beta=1):
  """ Calcul d'un noyau gaussien généralisé. """
  R = torch.tensor(
      [[math.cos(theta), -math.sin(theta)],
       [math.sin(theta), math.cos(theta)]]
      ).to(device)

  covariance_matrix = R @ torch.tensor([[sigma1**2.0, 0], [0, sigma2**2.0]]).to(device) @ R.t()

  kernel = torch.zeros(torch.Size([kernel_size, kernel_size])).to(device)

  center = (kernel_size - 1) / 2.0

  for i in range(0, kernel_size):
    for j in range(0, kernel_size):
      C = torch.tensor([i - center, j - center]).float().t().to(device)
      kernel[i, j] = math.exp(-(C.t() @ torch.linalg.inv(covariance_matrix) @ C)**beta/2)

  return kernel / kernel.sum()

def plateauShapedKernel(kernel_size, sigma1, sigma2, theta, beta):
  """ Calcul d'un noyau plateau-shaped. """
  R = torch.tensor(
      [[math.cos(theta), -math.sin(theta)],
       [math.sin(theta), math.cos(theta)]]
      ).to(device)
  covariance_matrix = R @ torch.tensor([[sigma1**2.0, 0], [0, sigma2**2.0]]).to(device) @ R.t()

  kernel = torch.zeros(torch.Size([kernel_size, kernel_size])).to(device)

  center = (kernel_size - 1) / 2.0

  for i in range(0, kernel_size):
    for j in range(0, kernel_size):
      C = torch.tensor([i - center, j - center]).float().t().to(device)
      kernel[i, j] = 1/(1 + (C.t() @ torch.linalg.inv(covariance_matrix) @ C)**beta)

  return kernel / kernel.sum()

def addRandomBlur(tensor_image, sigma_range):
  """ Applique un flou sur une image. """
  blur = random.randint(1, 100)

  kernel_size = random.randint(7, 21)
  if(kernel_size % 2 == 0):
    kernel_size += 1
  pad_size = (kernel_size - 1)//2

  sigma1 = random.uniform(sigma_range[0], sigma_range[1])
  sigma2 = random.uniform(sigma_range[0], sigma_range[1])

  if(blur <= 70):
    kernel = generalGaussianKernel(kernel_size, sigma1, sigma2, 0)
  elif(70 < blur <= 85):
    beta = random.uniform(0.5, 4)
    kernel = generalGaussianKernel(kernel_size, sigma1, sigma2, 0, beta)
  else:
    beta = random.uniform(1, 2)
    kernel = plateauShapedKernel(kernel_size, sigma1, sigma2, 0, beta)

  tensor_image = F.pad(tensor_image, (pad_size, pad_size, pad_size, pad_size), mode='reflect').to(device)

  return F.conv2d(tensor_image, kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1), groups=3, padding=0)

def addGaussianNoise(tensor, sigma, noise = 'color'):
  """ Ajoute du bruit gaussien à une image. """
  if noise == 'color':
    noise = torch.randn_like(tensor)*sigma
  elif noise == 'gray':
    noise = torch.randn(1, tensor.shape[2], tensor.shape[3])*sigma
    noise = noise.repeat(tensor.shape[0], 3, 1, 1)

  return tensor + noise.to(device)

def addPoissonNoise(tensor, scale):
  """ Ajoute du bruit de Poisson à une image. """
  min = tensor.min()
  if(min < 0):
    return torch.poisson((tensor -  min) * scale).to(device) / scale + min
  else:
    return torch.poisson(tensor * scale).to(device) / scale

def addRandomNoise(tensor_image, noise_sigma_range, noise_scale_range):
  """ Ajoute du bruit à une image. """
  is_gaussian = random.randint(0, 1)
  if(is_gaussian):
    gray = random.randint(1, 10)
    sigma = random.uniform(noise_sigma_range[0], noise_sigma_range[1])
    if(gray >= 4):
      t = addGaussianNoise(tensor_image, sigma, noise = 'gray')
    else:
      t = addGaussianNoise(tensor_image, sigma, noise = 'color')
  else:
    scale = random.uniform(noise_scale_range[0], noise_scale_range[1])
    t = addPoissonNoise(tensor_image, scale)

  return t

def resize(tensor_image, mode, r):
  """ Redimensionne une image. """
  return F.interpolate(tensor_image, scale_factor=r, mode=mode)

def randomDownscaling(tensor_image, r):
  """ Réduit l'échelle d'une image. """
  n = random.randint(1, 3)
  if(n == 1):
    mode = 'bicubic'
  elif(n == 2):
    mode = 'bilinear'
  else:
    mode ='area'

  t = resize(tensor_image, mode, 1/r)
  return t

def circularLowpassKernel(omega_c, N):
  """ Calcul d'un noyau 2D sinc.
  Args :
    omega_c (float) : fréquence de coupure en radians.
    N (int) : taille horizontale et verticale du noyau.
  """
  with np.errstate(divide='ignore',invalid='ignore'):
    kernel = np.fromfunction(lambda x, y: omega_c*special.j1(omega_c*np.sqrt((x - (N -1)/2)**2 + (y - (N - 1)/2)**2))/(2*np.pi*np.sqrt((x - (N - 1)/2)**2 + (y - (N -1)/2)**2)), [N, N])
  if N % 2:
    kernel[(N - 1)//2, (N - 1)//2] = omega_c**2/(4*np.pi)
  return kernel

def add2DSincFilter(tensor_image, omega_c, kernel_size=21):
  """ Applique un filtre 2D sinc à une image. """
  kernel = circularLowpassKernel(omega_c, kernel_size)
  pad_size = (kernel_size - 1)//2
  tensor_image = F.pad(tensor_image, (pad_size , pad_size , pad_size , pad_size), mode='reflect')
  t = F.conv2d(tensor_image, torch.tensor(kernel).float().unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).to(device), groups=3, padding=0)
  return t

def jpegCompression(tensor_image, quality):
  """ Applique une compression JPEG à une image. """
  jpeg = DiffJPEG(height=tensor_image.shape[2], width=tensor_image.shape[3], differentiable=True, quality=quality).to(device)
  t = jpeg(tensor_image)
  return t

def create_LR_tensor_image(HR_tensor_image, blur_sigma_range, noise_sigma_range, noise_scale_range, omega_c, resize_factor, d=2):
  """ Crée une image de basse résolution par une dégradation d'ordre supérieur.

  Args :
    HR_tensor_image (tensor) : lot d'image à haute résolution.
    blur_sigma_range (list) : liste de d intervalles.
    noise_sigma_range (list) : liste de d intervalles.
    noise_scale_range (list) : liste de d intervalles.
    omega_c : paramètre omega du filtre 2D sinc.
    resize_factor (list) : liste de taille d qui contient le facteur de redimensionnement à chaque ordre de dégradation
    d (int) : ordre de dégradation. Par défaut, d=2.
    
  Returns :
    LR_tensor_image (tensor) : lot d'image à basse résolution.
    
  """
  LR_tensor_image = HR_tensor_image
  last_jpeg = random.randint(0, 1)

  for i in range(0, d):
    is_blur = False
    if(i > 0):
      blur = random.randint(1, 10)
      if(blur >= 2):
        is_blur = True
    if(i == 0 or is_blur):
      LR_tensor_image = addRandomBlur(LR_tensor_image, blur_sigma_range[i])

      sinc = random.randint(1, 100)
      if(sinc <= 10):
        LR_tensor_image = add2DSincFilter(LR_tensor_image, omega_c)

    LR_tensor_image = randomDownsampling(LR_tensor_image, resize_factor[i])

    LR_tensor_image = addRandomNoise(LR_tensor_image, noise_sigma_range[i], noise_scale_range[i])

    if(i != d - 1 or not last_jpeg):
      quality = random.randint(30, 95)
      LR_tensor_image = jpegCompression(LR_tensor_image, quality)

  final_sinc = random.randint(1, 10)

  if(final_sinc <= 8):
    LR_tensor_image = add2DSincFilter(LR_tensor_image, omega_c)

  if(last_jpeg):
    quality = random.randint(30, 95)
    LR_tensor_image = jpegCompression(LR_tensor_image, quality)

  return LR_tensor_image


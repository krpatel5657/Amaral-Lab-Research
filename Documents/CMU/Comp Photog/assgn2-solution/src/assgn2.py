import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io
from skimage import color
from scipy.interpolate import *
from matplotlib import pyplot as plt
from cp_hw2 import *
import cv2
import os
import pickle
import imageio
from scipy import stats

# exposures and log exposures
exposure_times = [np.power(2, k)/2048.0 for k in range(16)]
exposure_logs = [np.log(i) for i in exposure_times]

# weighting functions
z_min = 0.06
z_max = 0.95
def w_uniform(z):
    return 1 if z_min <= z <= z_max else 0
def w_tent(z):
    return min(z, 1-z) if z_min <= z <= z_max else 0
def w_gaussian(z):
    return np.exp(-16 * np.square(z-0.5)) if z_min <= z <= z_max else 0
def w_photon(z, k):
    return exposure_logs[k] if z_min <= z <= z_max else 0
def w_optimal(z, k, g, sigma):
    return (exposure_logs[k]**2)/(g*z + sigma**2) if z_min <= z <= z_max else 0

def linearize(Z, l, w_type):
    if w_type == "uniform":
        w = w_uniform
    elif w_type == "tent":
        w = w_tent
    elif w_type == "gaussian":
        w = w_gaussian
    else:
        w = lambda z,k : w_photon(z,k)

    n = 256
    num_pixels, num_images = Z.shape
    A = np.zeros((num_images * num_pixels + n + 1, n + num_pixels))
    b = np.zeros((A.shape[0], 1))

    # Include the data-fitting equations
    k = 0
    for i in range(num_pixels):
        for j in range(num_images):
            if w_type == "photon":
                wij = w(Z[i,j]/255.0, j)
            else:
                wij = w(Z[i,j]/255.0)
            A[k, Z[i,j]] = wij
            A[k, n + i] = -wij 
            b[k] = wij * exposure_logs[j]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1.0
    k += 1

    # Include the smoothness equations
    for i in range(n - 2):
        if w_type == "photon":
                weight = 1
        else:
            weight = w((i + 1)/255.0)
        A[k, i] = l * weight
        A[k, i + 1] = -2 * l * weight
        A[k, i + 2] = l * weight
        k += 1
    # Solve the system using least squares (lstsq)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:n] # 256 + size of image originally
    lE = x[n:]

    return g, lE

# when calling merge functions, make sure to normalize!! different normalization schemes for jpg vs tiff
def merge_linear(imgs, im_lin, w_type):
    num_imgs = imgs.shape[0]
    if w_type == "uniform":
        w = np.vectorize(w_uniform)
    elif w_type == "tent":
        w = np.vectorize(w_tent)
    elif w_type == "gaussian":
        w = np.vectorize(w_gaussian)
    else:
        w = np.vectorize(lambda z,k : w_photon(z,k))
    num, den = np.zeros(imgs[0].shape), np.zeros(imgs[0].shape)
    for k in range(num_imgs):
        print("On image",k,"of merging")
        if w_type == "photon":
            weights = w(imgs[k], k)
        else:
            weights = w(imgs[k])
        num += weights * im_lin[k] / exposure_times[k]
        den += weights
    can_use = den != 0
    res = num
    if len(res[can_use].flatten()) == 0:
        return res.reshape(num.shape)
    res[can_use] /= den[can_use]

    # fix underexposed or overexposed pixels
    res = res.flatten()
    invalid = (den.flatten() == 0)
    flat_img = imgs[0].flatten()
    under = np.all(np.stack((invalid, (flat_img < z_min)), axis=0), axis=0)
    over = np.all(np.stack((invalid, (flat_img > z_min)), axis=0), axis=0)
    res[under] = np.min(res[can_use.flatten()])
    res[over] = np.max(res[can_use.flatten()])

    return res.reshape(num.shape)

def merge_log(imgs, im_lin, w_type):
    e = 0.001 # epsilon param for logarithmic merging

    num_imgs = imgs.shape[0]
    if w_type == "uniform":
        w = np.vectorize(w_uniform)
    elif w_type == "tent":
        w = np.vectorize(w_tent)
    elif w_type == "gaussian":
        w = np.vectorize(w_gaussian)
    else:
        w = np.vectorize(lambda z, k: w_photon(z, k))
    num, den = np.zeros(imgs[0].shape), np.zeros(imgs[0].shape)
    for k in range(num_imgs):
        print("On image", k, "of merging")
        if w_type == "photon":
            weights = w(imgs[k], k)
        else:
            weights = w(imgs[k])
        num += weights * (np.log(im_lin[k] + e) - exposure_logs[k])
        den += weights
    can_use = den != 0
    res = num
    if len(res[can_use].flatten()) == 0:
        return res.reshape(num.shape)
    res[can_use] /= den[can_use]

    # fix underexposed or overexposed pixels
    res = res.flatten()
    invalid = (den.flatten() == 0)
    flat_img = imgs[0].flatten()
    under = np.all(np.stack((invalid, (flat_img < z_min)), axis=0), axis=0)
    over = np.all(np.stack((invalid, (flat_img > z_min)), axis=0), axis=0)
    res[under] = np.min(res[can_use.flatten()])
    res[over] = np.max(res[can_use.flatten()])
    return np.exp(res.reshape(num.shape))

def homog(vec):
		x, y = vec
		return [x, y, 1]

def crop_color_checker_patches(image_path):
    # Load HDR image
    hdr_image = readHDR(image_path)

    # Use matplotlib's ginput to interactively select patch coordinates
    patch_coordinates = [[3350, 1481], [3352, 1317], [3350, 1160], [3350, 1020], [3350, 853], [3350, 688], [3538, 1475], [3538, 1317], [3538,1178], [3538, 1015], [3538, 870], [3538, 688], [3665, 1493], [3665, 1329], [3665, 1160], [3665, 1003], [3665, 882], [3665, 682], [3850, 1511], [3846, 1341], [3840, 1160], [3852, 997], [3840, 852], [3840, 700]]
    cropped_patches = []
    for coords in patch_coordinates:
        # Assuming patch size is fixed, e.g., 20x20 pixels
        patch_size = 20
        x, y = int(coords[0]), int(coords[1])
        patch = hdr_image[y:y + patch_size, x:x + patch_size, :]
        cropped_patches.append(patch)

    average_rgbs = []
    for patch in cropped_patches:
        average_rgb = np.mean(patch, axis=(0, 1))
        average_rgbs.append(average_rgb)
    average_rgbs = np.array(average_rgbs)
    
    A = np.zeros((72, 12))
    r, g, b = read_colorchecker_gm()
    rgb_array = np.stack([r, g, b], axis=2)

    for i in range(24):
        A[3*i] = [average_rgbs[i][0], average_rgbs[i][1], average_rgbs[i][2], 1, 0, 0, 0, 0, 0, 0, 0, 0]
        A[3*i+1] = [0, 0, 0, 0, average_rgbs[i][0], average_rgbs[i][1],
                average_rgbs[i][2], 1, 0, 0, 0, 0]
        A[3*i+2] = [0, 0, 0, 0, 0, 0, 0, 0, average_rgbs[i][0], average_rgbs[i][1],
                  average_rgbs[i][2], 1]
    rgb_array = rgb_array.reshape(72, 1)
    transformation_matrix, _, _, _ = np.linalg.lstsq(
        A, rgb_array, rcond=None)

    transformation_matrix = np.reshape(transformation_matrix, (3, 4))

    height, width, channels = hdr_image.shape
    hdr_image = np.reshape(np.append(hdr_image.reshape(height*width, channels), np.ones(
        (height*width, 1)), axis=1), (height, width, channels+1))
    
    def applyTransform(vec):
          return np.matmul(transformation_matrix, vec)
    
    transformed_image = np.apply_along_axis(
        applyTransform, 2, hdr_image)
    transformed_image[transformed_image < 0] = 0 # clip negative
    rgb_array = rgb_array.reshape(24, 3)
    return white_balance(transformed_image.reshape((height, width, channels)), [rgb_array[-6][0], rgb_array[-6][1], rgb_array[-6][2]])
    # scale so that white balancing works

def gamma_correction(channel):
    linear_scale = 12.92 * channel
    nonlinear_scale = (1 + 0.055) * (channel ** (1/2.4)) - 0.055
    return np.where(channel <= 0.0031308, linear_scale, nonlinear_scale)

def white_balance(image, ground_white):
    r,g,b = ground_white
    r_channel = image[:, :, 0] / r
    g_channel = image[:, :, 1] / g
    b_channel = image[:, :, 2] / b
    return np.stack([r_channel, g_channel, b_channel], axis=-1)

def tonemapping(image_path):
    hdr_image = readHDR(image_path)
    height, width, channels = hdr_image.shape
    # parameters to change
    k = 1000
    b = 0.06
    e = 0.01

    imhdr = np.exp((1/(height*width))* np.sum(np.log(hdr_image + e)))
    iijhdr = (k/imhdr) * hdr_image
    iwhite = b * np.max(iijhdr)
    iijtm = iijhdr*(1 + (iijhdr/(iwhite**2)))/(1 + iijhdr)
    
    
    return np.stack([gamma_correction(iijtm[:, :, 0]), gamma_correction(iijtm[:, :, 1]), gamma_correction(iijtm[:, :, 2])], axis=-1)

def tonemapping_luminence(image_path):
    hdr_image = readHDR(image_path)

    # parameters to change
    k = 0.3
    b = 0.1
    e = 0.01

    height, width, channels = hdr_image.shape
    xyz = lRGB2XYZ(hdr_image)
    X = xyz[:,:,0]
    Y = xyz[:,:,1]
    Z = xyz[:,:,2]

    imhdr = np.exp((1/(height*width)) * np.sum(np.log(Y + e)))
    iijhdr = (k/imhdr) * Y
    iwhite = b * np.max(iijhdr)
    iijtm = iijhdr*(1 + (iijhdr/(iwhite**2)))/(1 + iijhdr)

    # convert xyY to XYZ again
    X, Y, Z = xyY_to_XYZ(X/(X+Y+Z), Y/(X+Y+Z), iijtm)
    rev_img = XYZ2lRGB(np.stack((X, Y, Z), axis=-1))
    return np.stack([gamma_correction(rev_img[:, :, 0]), gamma_correction(rev_img[:, :, 1]), gamma_correction(rev_img[:, :, 2])], axis=-1)

# Merge using optimal weighting scheme
def merge_optimal(imgs, im_lin, g, sigma, idark):
    num_imgs = imgs.shape[0]
    w = np.vectorize(lambda z, k, g, sigma: w_optimal(z, k, g, sigma))
    num, den = np.zeros(imgs[0].shape), np.zeros(imgs[0].shape)
    for k in range(num_imgs):
        print("On image", k, "of merging")
        #tnc = 1/400
        imgs[k] -= (exposure_times[k]/(1./400))*idark
        weights = w(imgs[k], k, g, sigma)
        num += weights * im_lin[k] / exposure_times[k]
        den += weights
    can_use = den != 0
    res = num
    if len(res[can_use].flatten()) == 0:
        return res.reshape(num.shape)
    res[can_use] /= den[can_use]

    # fix underexposed or overexposed pixels
    res = res.flatten()
    invalid = (den.flatten() == 0)
    flat_img = imgs[0].flatten()
    under = np.all(np.stack((invalid, (flat_img < z_min)), axis=0), axis=0)
    over = np.all(np.stack((invalid, (flat_img > z_min)), axis=0), axis=0)
    res[under] = np.min(res[can_use.flatten()])
    res[over] = np.max(res[can_use.flatten()])

    return res.reshape(num.shape)

def generate_ramp():
    ramp = np.tile(np.linspace(0, 1, 255), (255, 1))
    io.imsave('ramp.jpg', ramp)

# Create my own hdr
def create_hdr(images):

    # parameters to change
    
    weight_type = "uniform"
    print("Finding g...")

    downsample_image = np.transpose(np.array(
        [np.ndarray.flatten(img[::200, ::200]) for img in images]))
    
    l = 100
    #g, lE = linearize(downsample_image, l, weight_type)
    #with open('myimage_values.pkl', 'wb') as f:
    #    pickle.dump(g, f)
    #with open('myimage_values.pkl', 'rb') as f:
    #    g = pickle.load(f)
    #plt.plot(g)
    #plt.show()

    # use g_linear to apply g to an image
    def g_linearize(pixel):
        return g[pixel]
    g_linear = np.vectorize(g_linearize)

    print("Linearizing...")
    images=np.array(images)
    # Save linearized images in pickle file
    #linear_images = [np.exp(g_linear(img)) for img in images]
    #with open('linearMyImageJPG.pkl', 'wb') as f:
    #    pickle.dump(linear_images, f)
    with open('linearMyImageJPG.pkl', 'rb') as f:
        linear_images = np.array(pickle.load(f))
    # merge normalized original with linearized jpg
    print("Merging...")
    merged_jpg_linear_tent = merge_linear(
        images/255.0, linear_images, weight_type).astype("float32")
    
    writeHDR("merged_linear_my_image_" + weight_type + ".hdr", merged_jpg_linear_tent)
    
    # Don't color map because no color checker

    #regular tonemap
    print("Tonemapping...")
    tonemapped = tonemapping("merged_linear_my_image_uniform.hdr")
    writeHDR("myimage_tonemapped_with_tiff.hdr", tonemapped)
    #io.imsave('myimage_tonemapped_1000_0.06.jpg', tonemapped)

# Take mean of all images to compute dark frame
def compute_dark_frame(images):
    return np.mean(images, axis = 0)

# Perform noise calibration
def noise_calibration(imgs, dark_frame):
    N = imgs.shape[0]
    for i in range(len(imgs)):
        imgs[i] -= dark_frame

    # compute mean and variance
    channel = 0
    means = np.round(np.sum(imgs[:,:, channel], axis=0)/N)
    variances = np.sum(np.power(imgs[:, :, channel] - means, 2), axis=0)/(N - 1)

    # find unique means
    unique_means = np.unique(means)
    # take average variances for unique means
    average_variances = [np.mean(np.array(variances)[np.where(
        np.array(means) == m)]) for m in unique_means]
    
    # display scatterplot of data and plot line of best fit
    plt.figure(figsize=(8, 6)) 
    plt.scatter(unique_means, average_variances, label='Data', color='red')
    slope, intercept, _, _, _ = stats.linregress(
        unique_means, average_variances)
    y_fit = slope * unique_means + intercept
    plt.plot(
        unique_means, y_fit, label=f'Line of Best Fit (y = {slope:.2f}x + {intercept:.2f})', color='black')
    plt.show()

    # slope and intercept are gain and sigma_add respectively
    return slope, intercept

def create_optimal(images, gain, sigma, dark_frame):
    weight_type = "uniform"
    images = np.array(images)
    
    # Use optimal merging scheme
    print("Merging...")
    merged_jpg_linear_tent = merge_optimal(
        images/(2**16 - 1), images/(2**16 - 1), gain, sigma, dark_frame).astype("float32")

    writeHDR("merged_optimal_my_image_" + weight_type +
             ".hdr", merged_jpg_linear_tent)

    # Don't color map because no color checker

    # RGB tonemap
    print("Tonemapping...")
    tonemapped = tonemapping("merged_optimal_my_image_uniform.hdr")
    writeHDR("optimal_tonemapped.hdr", tonemapped)
    io.imsave('myimage_optimal_tonemapped.jpg', tonemapped)

# Uncomment below to use provided image
"""
# get all images
images = []
for i in range(1, 17):
    url = './data/door_stack/exposure' + str(i) + '.jpg'
    #url = './data/door_stack/exposure' + str(i) + '.tiff'
    images.append(io.imread(url))

# parameters to change
weight_type = "uniform"

print("Finding g...")
# Save g in a pickle file
downsample_image = np.transpose(np.array(
    [np.ndarray.flatten(img[::200, ::200]) for img in images]))
l = 100
g, lE = linearize(downsample_image, l, weight_type)
with open('values.pkl', 'wb') as f:
    pickle.dump(g, f)
with open('values.pkl', 'rb') as f:
    g = pickle.load(f)
#plt.plot(g)
#plt.show()

# Save linearized images in a pickle file
def g_linearize(pixel):
    return g[pixel]
g_linear = np.vectorize(g_linearize)

print("Linearizing...")
images=np.array(images)
linear_images = [np.exp(g_linear(img)) for img in images]
with open('linearJPG.pkl', 'wb') as f:
   pickle.dump(linear_images, f)
with open('linearJPG.pkl', 'rb') as f:
   linear_images = np.array(pickle.load(f))
# merge normalized original with linearized jpg
print("Merging...")
merged_jpg_linear_tent = merge_linear(
    images/255.0, linear_images, weight_type).astype("float32")
writeHDR("merged_linear_" + weight_type + _0.001.hdr", merged_jpg_linear_tent)

# Color map
print("Colormapping...)
writeHDR("color_mapped.hdr", crop_color_checker_patches(
    "./data/merged_linear.hdr"))

#regular tonemap
tonemapped = tonemapping("./data/color_mapped.hdr")
#writeHDR("tonemapped.hdr", tonemapped)
io.imsave('result0.15_0.1.jpg', tonemapped)

# luminence tonemap
tonemapped = tonemapping_luminence("./data/color_mapped.hdr")
io.imsave('result_lum_0.3_0.1.jpg', tonemapped)

"""

# Uncomment below to use my captured image

"""
images = []
for i in range(16, 25):
    url = './data/my_image/DSC_01' + str(i) + '.jpg'
    #url = './data/my_image/DSC_01' + str(i) + '.tiff'
    images.append(io.imread(url))
create_hdr(images)
"""

# Uncomment below to use dark frame and ramp images (perform noise calibration)
# Here is where I think the issues arise - my captured images for dark frame don't seem
# to reflect accurate values for an image with the lens cap on
"""
images = []
N = 50
for i in range(128, 128+N):
    url = './krishna_raw/DSC_0' + str(i) + '.tiff'
    #url = './data/door_stack/exposure' + str(i) + '.tiff'
    images.append(io.imread(url).astype('float32'))
with open('dark_frame.pkl', 'rb') as f:
   dark_frame = np.array(pickle.load(f))
gain, sigma = noise_calibration(np.array(images), dark_frame)
images = []
for i in range(16, 24):
    url = './krishna_raw/DSC_01' + str(i) + '.tiff'
    images.append(io.imread(url).astype('float32'))
create_optimal(images, gain, sigma, dark_frame)
"""

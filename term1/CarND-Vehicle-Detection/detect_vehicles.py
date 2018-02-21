#!/usr/bin/python

import os
import cv2
import numpy as np
import pickle
import glob
import time
import sys
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

TEST_IMAGE_DIR = './test_images/'
recent_bboxes = []

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in imgs:
        file_features = []
        image = cv2.imread(file)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features                        

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, is_video=False):
    
    draw_img = img
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bbox_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rect = ((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw+win_draw + ystart))
                #cv2.rectangle(img, rect[0], rect[1], (0,0,255), 5)
                bbox_list.append(rect)
    
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap_threshold = 1

    if is_video:
        heatmap_threshold = 10
        if len(recent_bboxes) >= 18:
            del recent_bboxes[0]
            del recent_bboxes[0]
            del recent_bboxes[0]
    else:
        if len(recent_bboxes) >= 3:
            del recent_bboxes[:]
    
    recent_bboxes.append(bbox_list)

    # heatmap
    for i in recent_bboxes:
        for box in i:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # apply heatmap threshold
    heatmap[heatmap <= heatmap_threshold] = 0

    # remove quantity
    heatmap[heatmap >= 1] = 1

    labels = label(heatmap)
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0,255,0), 6)

#    if not is_video:
#        cv2.imwrite('./output_images/' + file_path.replace('.jpg', '') + '_heatmap.jpg', labels[0])

    return draw_img

def find_cars_video(img):
    ystart = 400
    ystop = 656
    scale = 1.5

    find_cars(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, is_video=True)
    find_cars(img, ystart, ystop, 3.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, is_video=True)
    return find_cars(img, ystart, ystop, 2.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, is_video=True)

# configuration
color_space     = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient          = 9
pix_per_cell    = 8
cell_per_block  = 2
hog_channel     = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size    = (32, 32)
hist_bins       = 48
spatial_feat    = True # Spatial features on or off
hist_feat       = True # Histogram features on or off
hog_feat        = True # HOG features on or off


# train data
if not os.path.isfile('svc.p') or len(sys.argv) == 2 and sys.argv[1] == "train":
    cars = glob.glob('data/vehicles/*/*.png')
    notcars = glob.glob('data/non-vehicles/*/*.png')

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
    else: 
        print('Fatal warning: your function only returns empty feature vectors...')

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()
    time_start = time.time()
    svc.fit(X_train, y_train)
    time_end = time.time()
    print(round(time_end - time_start, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    with open('svc.p', 'wb') as f:
        pickle.dump([svc, X_scaler], f)
else:
    print("Loading pre-fitted SVM...", end=' ')
    svc, X_scaler = pickle.load( open("svc.p", "rb" ) )
    print("OK")

print("Starting")

if len(sys.argv) == 2 and sys.argv[1] == "images": 
    for file_path in os.listdir(TEST_IMAGE_DIR):
        print("Running pipeline for '" + file_path + "'...", end=' ')
        if not file_path.endswith('.jpg'):
            print("Skip")
            continue
        img = cv2.imread(TEST_IMAGE_DIR + file_path)

        # debug output for writeup
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        cv2.imwrite('./output_images/' + file_path.replace('.jpg', '') + '_ycrcb.jpg', ycrcb_img)
        #features, visualization = get_hog_features(img[:,:,3], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
        #cv2.imwrite('./output_images/' + file_path.replace('.jpg', '') + '_hog.jpg', visualization)
        
        ystart = 400
        ystop = 656

        out_img = find_cars(img, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img = find_cars(img, ystart, ystop, 2.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        out_img = find_cars(img, ystart, ystop, 3.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        cv2.imwrite('./output_images/' + file_path, out_img)
        print("OK")

if len(sys.argv) == 2 and sys.argv[1] == "video":
    video_input = VideoFileClip("./project_video.mp4")
    processed_video = video_input.fl_image(find_cars_video)
    processed_video.write_videofile("./output_videos/project_video.mp4")

print("End")

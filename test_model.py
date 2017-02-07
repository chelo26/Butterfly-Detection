# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# Image importing:
from skimage import io
import scipy.io as sio
from sklearn.externals import joblib
import scipy.linalg as linalg

# HOG and histogram equalization:
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
from skimage import feature

# Region Proposals::
import matplotlib.patches as mpatches
import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb

# Transformations:
from skimage.transform import resize
from skimage.filters import gabor
from skimage import transform

# Global variables
BOX_SHAPE=4
RESIZE_VALUE=60
BBOX_NUM_FEATURES=900
THRESH_RANGE=np.arange(0.3,0.7,0.02)

# 1.1 Get Predictions
def get_image_prediction(image, model):  # -----> $$$
    # Extracting features:

    image_pix_list = get_pixel_list(image)  # -----> GO TO !!!

    # Predict:
    #print image_pix_list.shape
    y_test = model.predict_proba(image_pix_list)[:, 1]
    return np.array(y_test).reshape(image.shape[0], image.shape[1])


# 1.1.1 Converts an image into a list of pixels:
def get_pixel_list(image):  # -----> !!!
    return image.reshape(image.shape[0] * image.shape[1], image.shape[2])


# 1.2 Get the regions proposals and their performance measured in IoU
def get_multiple_boxes_iou(predicted_image, true_crop):  # -----> @@@

    total_bbox = np.array([]).reshape(0, BOX_SHAPE)
    total_iou = []
    for i_thresh in THRESH_RANGE:
        # Getting the bbox for the largest binary post:
        posterior_i = get_largest_post(predicted_image, i_thresh)
        posterior_bbox = mask_to_bbox(posterior_i)

        # Concatenate:
        total_bbox = np.concatenate((total_bbox, np.array([posterior_bbox])), axis=0)
        iou_measure = bb_iou(posterior_bbox, true_crop)
        total_iou.append(iou_measure)

    return total_bbox, total_iou


# 1.2.1 # Get the largest binary region with probability higher than thersh:

def get_largest_post(posterior_im, thresh):
    # threshold the posterior
    #    if no_thresh is True:
    posterior = np.where(posterior_im > thresh, 1, 0)
    # else:
    #    posterior=posterior_im

    labeled_posterior = label(posterior)
    # take the largest region from the image
    list_posterior = labeled_posterior.flatten()
    accum = np.bincount(list_posterior)
    accum[0] = 1
    final_im = np.where(labeled_posterior == np.argmax(accum), 1,0)
    return final_im


# 1.2.2
def mask_to_bbox(binary_max_region):
    # converts a binary mask to a bounding box
    im_non_zero = np.argwhere(binary_max_region)
    (ystart, xstart), (ystop, xstop) = im_non_zero.min(0), im_non_zero.max(0) + 1

    return np.array([xstart, xstop, ystart, ystop])


# 1.3  Getting the features for several boxes
def get_multiple_bbox_features(original_image, mult_bbox):  # ----> ;;;
    total_bbox_features = np.array([]).reshape(0, BBOX_NUM_FEATURES)
    for bbox in mult_bbox:
        # Getting the features for this bbox:
        feats_bbox = get_sample_bbox_features(original_image, bbox)  # -----> GO TO  &&&
        total_bbox_features = np.concatenate((total_bbox_features, np.array([feats_bbox])), axis=0)
    return total_bbox_features


# 1.3.1 Preprocessing the features for a given bbox:
def get_sample_bbox_features(original_image, bbox):  # ----->  &&&
    # Extract the original image:
    raw_image = get_bbox_image(original_image, bbox)  # ----->  GO TO eee
    # Resize to 60 x 60 and convert to grayscale:
    trans_image = resize(color.rgb2gray(raw_image), (RESIZE_VALUE, RESIZE_VALUE))
    # Histogram Equalization
    image_eq = exposure.equalize_hist(trans_image)
    # Gabor_filter:
    filt_real, _ = gabor(image_eq, frequency=0.6)
    # Sampling:
    sampled_image = transform.downscale_local_mean(filt_real, (2, 2)).flatten()
    return sampled_image


# 1.3.1.1 Get the image pixels for tha corresponding bbox:
def get_bbox_image(image, box, plot_box=False):  # -----> eee
    if plot_box == True:
        plt.imshow(image[int(box[2]):int(box[3]), int(box[0]):int(box[1])])
    # print "box: "+str(box.shape)
    return image[int(box[2]):int(box[3]), int(box[0]):int(box[1])]


# FUNCTION FOR TESTING THE NEW DATA:

#  1 Get the best Score and its performance IoU:

def get_bboxes_scores(test_images, test_crops, model_pixels, region_model):
    # Initialize the bbox and the iou
    total_bbox = np.array([]).reshape(0, test_crops[0].shape[0])
    total_iou = []

    for i in range(len(test_images)):
        # Get Best Box:
        best_bbox, _ = get_predicted_bbox(test_images[i], model_pixels, region_model)
        total_bbox = np.concatenate((total_bbox, np.array([best_bbox])), axis=0)

        # Calculate IoU:
        bbox_iou = bb_iou(best_bbox, test_crops[i])
        total_iou.append(bbox_iou)

        print "Box %d computed with IoU of %.2f" % (i, bbox_iou)
    return total_bbox, total_iou


# 1.1 Getting the best bbox according to the model:
def get_predicted_bbox(test_image, pixel_model, region_model):
    # Making a prediction:
    predictions = get_image_prediction(test_image, pixel_model)

    # Calculating the features of the Proposed Boxes :
    potential_boxes, boxes_features = get_best_bboxes(predictions, test_image)

    # Calculate the best box:
    # Add previous information:
    result = region_model.predict_proba(boxes_features)
    result = result.T[1, :]
    predicted_bbox = potential_boxes[np.argmax(result)]

    return predicted_bbox, potential_boxes


# 1.2 Get the best boxes according to the pixel model:
def get_best_bboxes(predictions, test_image):
    # Initialize the bboxes array
    total_bbox = np.array([]).reshape(0, BOX_SHAPE)

    for i_thresh in THRESH_RANGE:
        # Getting the bbox for the largest binary post:
        posterior_i = get_largest_post(predictions, i_thresh)
        posterior_bbox = mask_to_bbox(posterior_i)

        # Add the proposed bbox to the list:
        total_bbox = np.concatenate((total_bbox, np.array([posterior_bbox])), axis=0)

    # Calculate the features for the proposed bboxes
    bboxes_features = get_multiple_bbox_features(test_image, total_bbox)
    return total_bbox, bboxes_features

# Testing:

def bb_iou(bbox1, bbox2):
    # first compute the left, right, top, bottom of the intersection rectangle
    intersection = np.zeros(4);
    intersection[0] = max(bbox1[0], bbox2[0])
    intersection[1] = min(bbox1[1], bbox2[1])
    intersection[2] = max(bbox1[2], bbox2[2])
    intersection[3] = min(bbox1[3], bbox2[3])

    # Area of intersection
    intersection_area = box_area(intersection)

    # now find area of each bounding box
    area1 = box_area(bbox1);
    area2 = box_area(bbox2);

    # now use all this to compute the union
    union_area = area1 + area2 - intersection_area;

    # now finally the iou ratio
    iou_ratio = intersection_area / union_area;

    return iou_ratio


def box_area(bbox):
    # computes the area of a bounding box defined by (left, right, top, bottom)
    # (This will be zero if the coordinates are invalid)

    left = bbox[0];
    right = bbox[1];
    top = bbox[2];
    bottom = bbox[3];

    # If left > right or top > bottom, then set area to zero,
    # don't let area be negative!

    if left > right:
        return 0
    elif top > bottom:
        return 0
    else:
        return (right - left) * (bottom - top)



def box_area(bbox):
    # computes the area of a bounding box defined by (left, right, top, bottom)
    # (This will be zero if the coordinates are invalid)

    left = bbox[0];
    right = bbox[1];
    top = bbox[2];
    bottom = bbox[3];

    # If left > right or top > bottom, then set area to zero,
    # don't let area be negative!

    if left > right:
        return 0
    elif top > bottom:
        return 0
    else:
        return (right - left) * (bottom - top)

# Plotting:
def plot_results(original_image, true_bbox, model_pixel, model_bbox, name):
    # Get the predictions:
    posterior_im = get_image_prediction(original_image, model_pixel)

    # Getting diferent labels

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 14))
    plt.axis('off')

    ax1.imshow(original_image)

    bin_posterior = get_largest_post(posterior_im, 0.25)
    bbox_posterior = mask_to_bbox(bin_posterior)
    patch_posterior = box_to_patch(bbox_posterior, 'red')

    true_patch = box_to_patch(true_bbox, 'blue')

    ax2.imshow(posterior_im, cmap=plt.cm.gray)
    ax2.add_patch(patch_posterior)
    ax2.add_patch(true_patch)

    ax3.imshow(posterior_im, cmap=plt.cm.gray)

    best_bbox, _ = get_predicted_bbox(original_image, model_pixel, model_bbox)

    best_patch = box_to_patch(best_bbox, 'red')
    true_patch = box_to_patch(true_bbox, 'blue')

    ax3.add_patch(best_patch)
    ax3.add_patch(true_patch)

    iou_post = bb_iou(bbox_posterior, true_bbox)
    iou_bbox = bb_iou(best_bbox, true_bbox)

    # Titles:
    ax1.set_title("Original Image")
    ax2.set_title("Detection after Model 1, thresh=0.5, IoU = %.2f" % (iou_post))
    ax3.set_title("Final Detection after adding Box Model, IOU = %.2f" % (iou_bbox))
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    fig.savefig(name)

# PLOTING AND ANALIZING:
def box_to_patch(bbox,color):
    x_start=bbox[0]
    x_stop=bbox[1]
    y_start=bbox[2]
    y_stop=bbox[3]
    # Patch Coordinates:
    a=[x_start,y_start],x_stop-x_start,y_stop-y_start
    return mpatches.Rectangle(a[0],a[1],a[2],fill = False, edgecolor=color, linewidth=2)



if __name__=="__main__":
    # Loding Test Images
    test_filename = "../data/test.mat"   ## Define the location of the data
    loadTest = sio.loadmat(test_filename)
    # Importing crops:
    test_crops = loadTest['crops']
    test_names_array = loadTest['image_names'][0]

    # Importing Image_names:
    image_names = []
    for i in range(len(test_names_array)):
        image_names.append(str(test_names_array[i][0]))

    # Importing all Test Images:
    test_set = []
    for name_test_im in image_names:
        path_to_image = "../data/test/" + name_test_im
        test_set.append(io.imread(path_to_image))
    test_set = np.array(test_set)
    print "test set imported"

    # Loading the models:
    pixel_model = joblib.load('pixel_model.pkl')  ### ----> Need to be updloaded from the link in the README file
    box_model = joblib.load('box_model.pkl')


    # Test model:
    init_img=0
    last_img=100

    list_bbox,list_iou_test=get_bboxes_scores(test_set[init_img:last_img],test_crops[init_img:last_img],pixel_model,box_model)

    if False:   ##---> change to true in order to get the images plots
        for i in range(last_img):
            path = "./results/"+image_names[i]
            plot_results(test_set[i], test_crops[i], pixel_model, box_model, path)
            print "image processed " +str(i)


    print "average iou test: %.2f" %(np.mean(list_iou_test))

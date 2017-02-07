import matplotlib.pyplot as plt
import numpy as np

# Image importing:
from skimage import io
import scipy.io as sio
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

# Transformations:
from skimage.transform import resize
from skimage.filters import gabor
from skimage import transform

# Save the model:
from sklearn.externals import joblib

# Global variables
BOX_SHAPE = 4
RESIZE_VALUE = 60
BBOX_NUM_FEATURES = 900
ACCEPTANCE_THRESH = 0.85
THRESH_RANGE = np.arange(0.3, 0.7, 0.02)

# Getting the images:




# Function to label and image with a a bounding box
def create_label(image, bbox):
    #Initializing label image
    mask = np.array([[0] * image.shape[1] for i in range(image.shape[0])])

    # Iterating:
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if bbox[0] < j < bbox[1] and bbox[2] < i < bbox[3]:
                mask[i][j] = 1
    return mask


# FUNCTIONS FOR EXTRACTING FEATURES FOR THE FIRST MODEL:

# MAIN FUNCTION:
def extract_training_features(training_set, mask_set, number_samples):
    # Initialize Using the first img:
    num_dim, _ = extract_pos_neg_feats(training_set[0], mask_set[0])
    print "num_dim: " + str(num_dim.shape)
    num_dim = num_dim.shape[1]

    # Initialize array:
    total_pos = np.array([]).reshape(0, num_dim)
    total_neg = np.array([]).reshape(0, num_dim)
    # getting features:
    for num_sam in range(number_samples):
        X_pos, X_neg = extract_pos_neg_feats(training_set[num_sam], mask_set[num_sam])
        # print "total_feat_set: "+str(total_feat_set.shape)
        total_pos = np.concatenate((total_pos, X_pos), axis=0)
        total_neg = np.concatenate((total_neg, X_neg), axis=0)
        print "Extraction %d of %d completed. Positive samples: %d. Negative samples: \
         %d" % (num_sam + 1, number_samples, total_pos.shape[0], total_neg.shape[0])
    return total_pos, total_neg


# EXTRACTING FOR 1 IMAGE:

def extract_pos_neg_feats(image, mask):
    # Adding Features:
    # hog_image=add_hog(image)
    # edges=add_edges(image)

    # image=np.dstack((image, hog_image))
    # image=np.dstack((image, edges))

    #    rows=image.shape[0]
    #    cols=image.shape[1]
    # print "image_shape: "+str(image.shape)
    # Separate:
    pos_image = image[mask > 0]
    neg_image = image[mask == 0]
    # print "pos_image_shape: "+str(pos_image.shape)
    return pos_image, neg_image


# FUNCTIONS FOR EXTRACTION AND SAMPLING:

def add_hog(image):
    _, hog_image = hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualise=True)
    return hog_image


def add_edges(image):
    # Generate noisy image of a square
    edges = color.rgb2gray(image)
    # Compute the Canny filter for two values of sigma
    return edges


def get_sample_pixels(feats_array, sampling_ratio):
    # Get the sample of a list of pictures:
    number_samples = int(len(range(feats_array.shape[0])) * sampling_ratio)
    idx_sample = np.random.choice(range(feats_array.shape[0]), number_samples)
    sampled_array = feats_array[idx_sample]
    return sampled_array


# FUNCTIONS FOR EXTRACTING THE REGION PROPOSALS and their performance measured in IoU:

# 1. Extracting Regions Features and IOU:
def get_set_features_iou(set_images, set_true_crops, model):
    total_bboxes = np.array([]).reshape(0, BOX_SHAPE)
    total_ious = []
    total_bboxes_features = np.array([]).reshape(0, BBOX_NUM_FEATURES)
    for i in range(len(set_images)):
        # Getting a prediction :
        predicted_image = get_image_prediction(set_images[i], model)  # -----> Go to $$$

        # Getting different bboxes and ious:
        bboxes, ious = get_multiple_boxes_iou(predicted_image, set_true_crops[i])  # ----> Go to @@@

        # Adding the IoUs and the BBoxes:
        total_bboxes = np.concatenate((total_bboxes, bboxes), axis=0)
        total_ious = np.concatenate((total_ious, ious), axis=0)

        # Getting the features for each box:
        bboxes_features = get_multiple_bbox_features(set_images[i], bboxes)  # ----->Go to ;;;

        total_bboxes_features = np.concatenate((total_bboxes_features, bboxes_features), axis=0)
        print "image %d completed, number of boxes: %d" % (i + 1, total_bboxes.shape[0])
    return total_bboxes, total_ious, total_bboxes_features


# 1.1 Get Predictions
def get_image_prediction(image, model):  # -----> $$$
    # Extracting features:
    image_pix_list = get_pixel_list(image)  # -----> GO TO !!!

    # Predict:
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
    final_im = np.where(labeled_posterior == np.argmax(accum), 1,
                        0)  #########---------> check list_posterior or labeled post!!!
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


# 1.3.1.1 Get the image pixels for tha correspodant bbox:
def get_bbox_image(image, box, plot_box=False):  # -----> eee
    if plot_box == True:
        plt.imshow(image[int(box[2]):int(box[3]), int(box[0]):int(box[1])])
    # print "box: "+str(box.shape)
    return image[int(box[2]):int(box[3]), int(box[0]):int(box[1])]

#FUNCTION FOR TESTING THE NEW DATA:

# Get the best Score and its performance IoU:

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


# FUNCTIONS TO EVALUATE FIRST MODEL:

def get_first_model_error(set_images, set_crops, model_1):
    error = []
    for i in range(len(set_images)):
        # Making the prediction:
        prediction = get_image_class(set_images[i], model_1)
        # Getting the largest image:
        largest_region = get_largest_post(prediction, 0.5, False)
        posterior_bbox = mask_to_bbox(largest_region)
        error_iou = bb_iou(posterior_bbox, set_crops[i])
        print "error of image %d is %.2f" % (i, error_iou)
        error.append(error_iou)
    return error


def get_image_class(image, model):
    # get the list of pixels:
    image_pix_list = get_pixel_list(image)
    y_test = model.predict(image_pix_list)
    return np.array(y_test).reshape(image.shape[0], image.shape[1])


if __name__ == "__main__":
    # MAIN:
    # Loding Training Images
    filename = "../data/train.mat"
    loadTrain = sio.loadmat(filename)

    # Importing crops:
    training_crops = loadTrain['crops']
    image_names_array = loadTrain['image_names'][0]

    # Importing Image_names:
    image_names = []
    for i in range(len(image_names_array)):
        image_names.append(str(image_names_array[i][0]))

    # Importing all Training Images:
    training_set = []
    for name_train_im in image_names:
        path_to_image = "../data/train/" + name_train_im
        training_set.append(io.imread(path_to_image))
    training_set = np.array(training_set)

    # Importing and converting Masks:
    mask_set = []
    for i in range(len(training_set)):
        mask_set.append(create_label(training_set[i], training_crops[i]))
    mask_set = np.array(mask_set)

    print "training set imported"

    # Loding Test Images
    test_filename = "../data/test.mat"
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

    # Importing pixels for all images:
    number_training = 200
    # Extract features:
    X_pos, X_neg = extract_training_features(training_set, mask_set, number_training)

    # Sampling for efficience
    sampling_ratio = 0.005
    X_pos_sample = get_sample_pixels(X_pos, sampling_ratio)
    X_neg_sample = get_sample_pixels(X_neg, sampling_ratio)

    # Create training and label arrays:
    X = np.concatenate((X_pos_sample, X_neg_sample), axis=0)
    Y = np.concatenate((np.ones(X_pos_sample.shape[0]), np.zeros(X_neg_sample.shape[0])), axis=0)
    print "number of samples: " + str(X.shape[0])

    # Training the model:
    from sklearn.ensemble import RandomForestClassifier

    pixel_rf = RandomForestClassifier(n_estimators=25)
    pixel_rf = pixel_rf.fit(X, Y)

    # Extracting BBoxes Features and Labels to train Bbox model :
    num_train = 200
    bboxes, list_iou, bboxes_features = get_set_features_iou(training_set[0:num_train],
                                                             training_crops[0:num_train], pixel_rf)
    # Training boxes model :
    Y_iou = list_iou > ACCEPTANCE_THRESH
    sam_weight = len(Y_iou) / len(Y_iou[Y_iou])

    # Add a threshold
    rp_model = RandomForestClassifier(n_estimators=25)
    rp_model = rp_model.fit(bboxes_features, Y_iou)

    joblib.dump(pixel_rf, 'new_pixel_model.pkl')
    joblib.dump(rp_model, 'new_box_model.pkl')

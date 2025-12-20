#https://www.geeksforgeeks.org/computer-vision/histogram-of-oriented-gradients/
#Pip install numpy
#Pip install Pillow
#Pip install xgboost

import numpy as np
from PIL import Image
import math
from xgboost import XGBClassifier
import random
import time
import os

def extract_hog_features(img_path):
    """Extract HOG features from a single image"""
    # Declare Sobel Arrays
    GxSobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    GySobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Load and preprocess image
    img = Image.open(img_path).convert("L")
    img_resized = img.resize((512, 512)) 
    img_np = np.array(img_resized)
    imgArr = np.pad(img_np, 1)
    
    arrSize = img_np.shape
    
    # Initialize arrays
    Gx = np.zeros((arrSize[0], arrSize[1]))
    Gy = np.zeros((arrSize[0], arrSize[1]))
    magnitude = np.zeros((arrSize[0], arrSize[1]))
    orientation = np.zeros((arrSize[0], arrSize[1]))
    
    # Compute gradients
    for i in range(1, arrSize[0]+1):
        for j in range(1, arrSize[1]+1):
            Gx[i-1][j-1] = (imgArr[i-1][j-1] * GxSobel[0][0]) + (imgArr[i-1][j] * GxSobel[0][1]) + (imgArr[i-1][j+1] * GxSobel[0][2]) + \
                       (imgArr[i][j-1] * GxSobel[1][0]) + (imgArr[i][j] * GxSobel[1][1]) + (imgArr[i][j+1] * GxSobel[1][2]) + \
                       (imgArr[i+1][j-1] * GxSobel[2][0]) + (imgArr[i+1][j] * GxSobel[2][1]) + (imgArr[i+1][j+1] * GxSobel[2][2])
             
            Gy[i-1][j-1] = (imgArr[i-1][j-1] * GySobel[0][0]) + (imgArr[i-1][j] * GySobel[0][1]) + (imgArr[i-1][j+1] * GySobel[0][2]) + \
                       (imgArr[i][j-1] * GySobel[1][0]) + (imgArr[i][j] * GySobel[1][1]) + (imgArr[i][j+1] * GySobel[1][2]) + \
                       (imgArr[i+1][j-1] * GySobel[2][0]) + (imgArr[i+1][j] * GySobel[2][1]) + (imgArr[i+1][j+1] * GySobel[2][2])
             
            magnitude[i-1][j-1] = np.sqrt(Gx[i-1][j-1]**2 + Gy[i-1][j-1]**2)
            orientation[i-1][j-1] = np.arctan2(Gy[i-1][j-1], Gx[i-1][j-1])
    
    orientation = (np.degrees(orientation) + 180) % 180
    
    # Create histograms
    histArr = np.zeros(((arrSize[0] // 8), (arrSize[1] // 8), 9))
    blocksArr = np.zeros(((arrSize[0] // 8)-1, (arrSize[1] // 8)-1, 36))
    
    for i in range(0, arrSize[0], 8):
        for j in range(0, arrSize[1], 8):
            cellArrMag = magnitude[i:i+8, j:j+8]
            cellArrOri = orientation[i:i+8, j:j+8]
    
            for a in range(8):
                for b in range(8):
                    binVal = cellArrOri[a][b] / 20
                    lowVal = math.floor(binVal)
                    binLow = lowVal % 9
                    binHigh = (binLow + 1) % 9
                    histArr[i // 8][j // 8][binHigh] += cellArrMag[a][b] * (binVal - lowVal)
                    histArr[i // 8][j // 8][binLow] += cellArrMag[a][b] * (1 - (binVal - lowVal))
    
    # Block normalization
    for i in range(arrSize[0]//8-1):
        for j in range(arrSize[1]//8-1):
            blocksArr[i, j] = np.concatenate([histArr[i,j], histArr[i+1, j], histArr[i, j+1], histArr[i+1, j+1]])
            histRoot = np.sqrt(np.sum(blocksArr[i][j]**2)) + 1e-9
            blocksArr[i, j] = blocksArr[i][j] / histRoot
    
    return blocksArr.flatten()

Dataset = 3

""" First Dataset Used
base_path = "Bone Fractures Detection"

#Load Train Data
train_image_paths = []
train_labels = []
for filename in sorted(os.listdir(f"{base_path}/train/images")):
    if filename.endswith(('.jpg')):
        label_file = os.path.join(base_path, "train", "labels", os.path.splitext(filename)[0] + '.txt')
        if os.path.getsize(label_file) == 0:
            continue  
        train_image_paths.append(os.path.join(base_path, "train", "images", filename))
        with open(label_file, 'r') as f:
            class_id = int(f.read().strip().split()[0])
            if class_id == 2:
                train_labels.append(0)
                twoFound = 1
            else:
                train_labels.append(1)
        

# Load Test Data
test_image_paths = []
test_labels = []
count = 0
for filename in sorted(os.listdir(f"{base_path}/test/images")):
    if filename.endswith(('.jpg')):
        label_file = os.path.join(base_path, "test", "labels", os.path.splitext(filename)[0] + '.txt')
        if os.path.getsize(label_file) == 0:
            continue  
        test_image_paths.append(os.path.join(base_path, "test", "images", filename))
        with open(label_file, 'r') as f:
            class_id = int(f.read().strip().split()[0])
            if class_id == 2:
                test_labels.append(0)
                count += 1
            else:
                test_labels.append(1)
print(count)
print(len(train_image_paths))
print(len(train_labels))
print(len(test_image_paths))
print(len(test_labels))
"""

if Dataset == 2:
    base_path = "Dataset"
    healthyCount = 0
    unhealthyCount = 0

    #Load Train Data
    train_image_paths_healthy = []
    train_labels_healthy = []
    train_image_paths_unhealthy = []
    train_labels_unhealthy = []
    train_image_paths = []
    train_labels = []
    for filename in sorted(os.listdir(f"{base_path}/images/train")):
        if filename.endswith(('.png')):
            label_file = os.path.join(base_path, "labels", "train", os.path.splitext(filename)[0] + '.txt')
            img_path = os.path.join(base_path, "images", "train", filename)
            if not os.path.exists(label_file):
                continue
            if os.path.getsize(label_file) == 0:
                continue  
            path = os.path.join(base_path, "images", "train", filename)
            with open(label_file, 'r') as f:
                class_id = int(f.read().strip().split()[0])
                if class_id >= 0 and class_id <= 4: #Bone Fracture
                    train_image_paths_unhealthy.append(path)
                    train_labels_unhealthy.append(1)
                    unhealthyCount += 1
                else:
                    train_image_paths_healthy.append(path)
                    train_labels_healthy.append(0)
                    healthyCount += 1

    if len(train_image_paths_unhealthy) > len(train_image_paths_healthy):
        train_image_paths_unhealthy = train_image_paths_unhealthy[:len(train_image_paths_healthy)]
        train_labels_unhealthy = train_labels_unhealthy[:len(train_labels_healthy)]
    else:
        train_image_paths_healthy = train_image_paths_healthy[:len(train_image_paths_unhealthy)]
        train_labels_healthy = train_labels_healthy[:len(train_labels_unhealthy)]

    # Load Test Data
    test_image_paths = []
    test_labels = []
    count = 0
    for filename in sorted(os.listdir(f"{base_path}/images/val")):
        if filename.endswith(('.png')):
            label_file = os.path.join(base_path, "labels", "val", os.path.splitext(filename)[0] + '.txt')
            img_path = os.path.join(base_path, "images", "val", filename)
            if not os.path.exists(label_file):
                continue
            if os.path.getsize(label_file) == 0:
                continue  
            test_image_paths.append(os.path.join(base_path, "images", "val", filename))
            with open(label_file, 'r') as f:
                class_id = int(f.read().strip().split()[0])
                if class_id >= 0 and class_id <= 4: #Bone Fracture
                    test_labels.append(1)
                else:
                    test_labels.append(0)

if (Dataset == 3):
    base_path = "rawdataset"
    healthyCount = 0
    unhealthyCount = 0

    # Load Train Data
    train_image_paths_healthy = []
    train_labels_healthy = []
    train_image_paths_unhealthy = []
    train_labels_unhealthy = []
    train_image_paths = []
    train_labels = []

    for filename in sorted(os.listdir(f"{base_path}/train/images")):
        if filename.endswith('.jpg'):
            label_file = os.path.join(base_path, "train", "labels", os.path.splitext(filename)[0] + '.txt')
            img_path = os.path.join(base_path, "train", "images", filename)

            if not os.path.exists(label_file):
                continue
            if os.path.getsize(label_file) == 0:
                continue  

            with open(label_file, 'r') as f:
                class_id = int(f.read().strip().split()[0])
                if class_id == 0:  # Bone Fracture
                    train_image_paths_unhealthy.append(img_path)
                    train_labels_unhealthy.append(1)
                    unhealthyCount += 1
                else:
                    train_image_paths_healthy.append(img_path)
                    train_labels_healthy.append(0)
                    healthyCount += 1

    # Balance healthy/unhealthy
    if len(train_image_paths_unhealthy) > len(train_image_paths_healthy):
        train_image_paths_unhealthy = train_image_paths_unhealthy[:len(train_image_paths_healthy)]
        train_labels_unhealthy = train_labels_unhealthy[:len(train_labels_healthy)]
    else:
        train_image_paths_healthy = train_image_paths_healthy[:len(train_image_paths_unhealthy)]
        train_labels_healthy = train_labels_healthy[:len(train_labels_unhealthy)]

    train_image_paths = train_image_paths_healthy + train_image_paths_unhealthy
    train_labels = train_labels_healthy + train_labels_unhealthy

    # Load Test Data
    test_image_paths = []
    test_labels = []

    for filename in sorted(os.listdir(f"{base_path}/test/images")):
        if filename.endswith('.jpg'):
            label_file = os.path.join(base_path, "test", "labels", os.path.splitext(filename)[0] + '.txt')
            img_path = os.path.join(base_path, "test", "images", filename)

            if not os.path.exists(label_file):
                continue
            if os.path.getsize(label_file) == 0:
                continue  
            test_image_paths.append(img_path)
            with open(label_file, 'r') as f:
                class_id = int(f.read().strip().split()[0])
                if class_id == 0:  # Bone Fracture
                    test_labels.append(1)
                else:
                    test_labels.append(0)

#Select Random Samples
randomTrainSamples = random.sample(range(len(train_image_paths_healthy)), 500) #len(train_image_paths_healthy))
train_image_paths = [train_image_paths_healthy[i] for i in randomTrainSamples] + [train_image_paths_unhealthy[i] for i in randomTrainSamples]
train_labels = [train_labels_healthy[i] for i in randomTrainSamples] + [train_labels_unhealthy[i] for i in randomTrainSamples]

healthyCount = train_labels.count(0)
unhealthyCount = train_labels.count(1)

#Shuffle Samples
combined = list(zip(train_image_paths, train_labels))
random.shuffle(combined)
train_image_paths, train_labels = zip(*combined)
train_image_paths = list(train_image_paths)
train_labels = list(train_labels)

randomTestSamples = random.sample(range(len(test_image_paths)), len(test_image_paths))
test_image_paths = [test_image_paths[i] for i in randomTestSamples]
test_labels = [test_labels[i] for i in randomTestSamples]

print(f"Training Image Count: {len(train_image_paths)}")
print(f"Test Image Count: {len(test_image_paths)}")
print(f"Unhealthy Count: {unhealthyCount}")
print(f"Healthy Count: {healthyCount}")

print(test_labels.count(1))



start_wall_time = time.time()

# Extract HOG features for all images
print("Extracting HOG features...")
x = 1
all_features_train = []
for img_path in train_image_paths:
    features = extract_hog_features(img_path)
    all_features_train.append(features)
    print(f"image {x} done")
    x += 1
    

all_features_test = []
for img_path in test_image_paths:
    features = extract_hog_features(img_path)
    all_features_test.append(features)
    print(f"image {x} done")
    x += 1

end_wall_time = time.time()
elapsed_time = end_wall_time - start_wall_time
print(f"Elapsed wall clock time: {elapsed_time} seconds")

X_train = np.array(all_features_train)
X_test = np.array(all_features_test)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

weight = unhealthyCount/healthyCount

bst = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.01, objective='binary:logistic', tree_method='hist')

bst.fit(X_train, y_train)
preds = bst.predict(X_test)

correct = np.sum(preds == y_test)
total = len(y_test)
accuracy = correct / total

output_path = "HOG_XGBoost_testing_results.txt"

probabilities = bst.predict_proba(X_test)
with open(output_path, "w") as f:
    f.write(f"\nAccuracy: {accuracy:.4f}\n")
    for i in range(len(preds)):
        confidence = probabilities[i][preds[i]]
        f.write(f"Image {i}: Predicted={preds[i]}, Actual={y_test[i]}, Confidence={confidence:.2%}\n")
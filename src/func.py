import numpy as np

NUM_CLASS = 19


# --------------- Load Data --------------------


# --------------- Accuracy ---------------------
# calculate accuracy for total classes
def calculate_accuracy(ground_truth_label_list, predict_label_list):
    count_class_sample = np.zeros(NUM_CLASS)
    positive_class_sample = np.zeros(NUM_CLASS)
    for id,(ground_truth_label, predict_label) in enumerate(zip(ground_truth_label_list, predict_label_list)):
        count_class_sample[ground_truth_label-1] += 1
        if ground_truth_label == predict_label:
            positive_class_sample[ground_truth_label-1] += 1
    accuracy_array = positive_class_sample/count_class_sample
    accuracy = sum(positive_class_sample)/sum(count_class_sample)
    return accuracy, accuracy_array.tolist(), count_class_sample.tolist()


# --------------- F1 score ---------------------
# calculate F1 for each class
def calculate_F1_class(class_label, ground_truth_label_list, predict_label_list):
    count = len(ground_truth_label_list)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(count):
        ground_truth_label = ground_truth_label_list[i]
        predict_label = predict_label_list[i]
        if ground_truth_label == class_label:
            if predict_label == class_label:
                TP += 1
            else:
                FN += 1
        else:
            if predict_label == class_label:
                FP += 1
            else:
                TN += 1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    return F1

# calculate F1 for total classes, use average
def calculate_F1(ground_truth_label_list, predict_label_list):
    F1_list = []
    for label in range(1,NUM_CLASS+1):
        F1 = calculate_F1_class(label, ground_truth_label_list, predict_label_list)
        F1_list.append(F1)
    F1_array = np.array(F1_list)
    F1 = np.mean(F1_array)
    return F1,F1_list


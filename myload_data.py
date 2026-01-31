import PIL.Image
import numpy as np
import os
import random
import csv


#import multi_tasks.image_utils as image_utils

#根据csv读图片
def read_datalist():
    images_paths = list()
    labels = list()

    f = csv.reader(open(r"data\data_csv\train_data.csv", 'r'))
    for line in f:
        tmpimagefile = line[0]
        tmplabel = int(line[1])
        tmpimagefile = tmpimagefile.replace("/home/a610/Project/Ou/multi_tasks/Data_no_multi_roi","data\AllPhoto")
        # tmpimagefile = tmpimagefile.replace(" C:/Paper Reproduction/Co_teaching_test/data/AllPhoto","C:/Paper Reproduction/Co_teaching_test2/data/AllPhoto")

        assert os.path.exists(tmpimagefile), "dataset root: {} does not exist.".format(tmpimagefile)
        images_paths.append(tmpimagefile)
        labels.append(tmplabel)

    return images_paths, labels


#写csv
def write_csv(name, data, ilabels):
    with open(name, 'a', newline='') as f:
        csv_writer = csv.writer(f, dialect='excel')
        for i in range(len(data)):
            csv_writer.writerow([data[i], ilabels[i]])

    pass


#划分数据
def shuffle_split_filelist(path_c, training_rate, validataion_rate, testing_rate):
    imageist = []
    filelist = os.listdir(path_c,)
    for tmpfile in filelist:
        if os.path.splitext(tmpfile)[1] == '.bmp':
            imageist.append(os.path.join(path_c, tmpfile))
    random.shuffle(imageist)
    num_images = len(imageist)

    num_train = int(training_rate*num_images)
    num_val = int(validataion_rate*num_images)
    num_test = int(testing_rate*num_images)

    training_list = imageist[0:num_train]
    validation_list = imageist[num_train:(num_train + num_val)]
    testing_list = imageist[(num_train + num_val): ]
    return training_list, validation_list, testing_list


#生成三个数据集
def generate_datalist(path):

    path_c1 = os.path.join(path, 'high')
    training_rate = 0.5
    validataion_rate = 0.2
    testing_rate = 0.3

    path_c2 = os.path.join(path, 'low')
    path_c3 = os.path.join(path, 'mix')


    training_list1, validation_list1, testing_list1= \
        shuffle_split_filelist(path_c1, training_rate, validataion_rate, testing_rate)

    training_list2, validation_list2, testing_list2= \
        shuffle_split_filelist(path_c2, training_rate, validataion_rate, testing_rate)

    training_list3, validation_list3, testing_list3= \
        shuffle_split_filelist(path_c3, training_rate, validataion_rate, testing_rate)


    training_labels = np.append(np.zeros((1, len(training_list1)), np.uint8),
                                np.ones((1, len(training_list2)), np.uint8))
    training_labels = np.append(training_labels,
                                2 * np.ones((1, len(training_list3)), np.uint8))
    training_list = training_list1 + training_list2 + training_list3

    val_labels = np.append((np.zeros((1, len(validation_list1)), np.uint8)),
                           (np.ones((1, len(validation_list2)), np.uint8)))
    val_labels = np.append(val_labels, (2 * np.ones((1, len(validation_list3)), np.uint8)))
    validation_list = validation_list1 + validation_list2 + validation_list3

    testing_labels = np.append((np.zeros((1, len(testing_list1)), np.uint8)),
                               (np.ones((1, len(testing_list2)), np.uint8)))
    testing_labels = np.append(testing_labels, (2 * np.ones((1, len(testing_list3)), np.uint8)))
    testing_list = testing_list1 + testing_list2 + testing_list3


    write_csv('train_list.csv', training_list, training_labels)
    write_csv('val_list.csv', validation_list, val_labels)
    write_csv('test_list.csv', testing_list, testing_labels)

    print('finishing')

    pass

def load_per_data(image_path):
    img = PIL.Image.open(image_path)
    img_array = np.array(img)
    txtfile =image_path.replace('bmp', 'txt')
    pointlist = list()
    random.seed(0)

    if os.path.isfile(txtfile):
        with open(txtfile, "r") as f:
            for eachline in f:
                tmp = eachline.split()
                tmppoint = np.array([int(tmp[0]), int(tmp[1])])

                if 0==tmppoint[0] & 0==tmppoint[1]:
                    if len(pointlist) == 0:
                        continue
                    points_arrary = np.array(pointlist)
                    mins = np.min(points_arrary, axis=0)
                    maxs = np.max(points_arrary, axis=0)
                    height_length = np.abs(maxs[0] - mins[0])
                    width_length = np.abs(maxs[1] - mins[1])
                    longer_length = height_length if height_length > width_length else width_length
                    shorter_length = height_length if height_length <= width_length else width_length

                    base_proportion = 144.0/96.0

                    temp_length = longer_length
                    base_per = 1.1
                    base_per = base_per + random.uniform(0, 0.1)
                    temp_length = temp_length * base_per
                    temp_ano_length = temp_length / base_proportion

                    difference_length = (temp_length - longer_length) / 2
                    difference_ano_length = (temp_ano_length - shorter_length) / 2

                    if longer_length == height_length:
                        mins0_length_to_zero = int(mins[0] - difference_length)
                        mins1_length_to_zero = int(mins[1] - difference_ano_length)
                        maxs0_length_to_most = int(maxs[0] + difference_length)
                        maxs1_length_to_most = int(maxs[1] + difference_ano_length)

                        if temp_length > img_array.shape[0]:
                            mins0_length_to_zero = 0
                            maxs0_length_to_most = img_array.shape[0] - 1
                        elif mins0_length_to_zero < 0:
                            maxs0_length_to_most = maxs0_length_to_most - mins0_length_to_zero
                            mins0_length_to_zero = 0
                        elif maxs0_length_to_most > img_array.shape[0]:
                            mins0_length_to_zero = mins0_length_to_zero - (maxs0_length_to_most - img_array.shape[0])
                            maxs0_length_to_most = img_array.shape[0] - 1

                        if temp_ano_length > img_array.shape[1]:
                            mins1_length_to_zero = 0
                            maxs1_length_to_most = img_array.shape[1] - 1
                        elif mins1_length_to_zero < 0:
                            maxs1_length_to_most = maxs1_length_to_most - mins1_length_to_zero
                            mins1_length_to_zero = 0
                        elif maxs1_length_to_most > img_array.shape[1]:
                            mins1_length_to_zero = mins1_length_to_zero - (maxs1_length_to_most - img_array.shape[1])
                            maxs1_length_to_most = img_array.shape[1] - 1

                        mins = [mins0_length_to_zero, mins1_length_to_zero]
                        maxs = [maxs0_length_to_most, maxs1_length_to_most]
                    elif longer_length == width_length:
                        mins1_length_to_zero = int(mins[1] - difference_length)
                        mins0_length_to_zero = int(mins[0] - difference_ano_length)
                        maxs1_length_to_most = int(maxs[1] + difference_length)
                        maxs0_length_to_most = int(maxs[0] + difference_ano_length)

                        if temp_length > img_array.shape[1]:
                            mins1_length_to_zero = 0
                            maxs1_length_to_most = img_array.shape[1] - 1
                        elif mins1_length_to_zero < 0:
                            maxs1_length_to_most = maxs1_length_to_most - mins1_length_to_zero
                            mins1_length_to_zero = 0
                        elif maxs1_length_to_most > img_array.shape[1]:
                            mins1_length_to_zero = mins1_length_to_zero - (maxs1_length_to_most - im.shape[1])
                            maxs1_length_to_most = img_array.shape[1] - 1

                        if temp_ano_length > img_array.shape[0]:
                            mins0_length_to_zero = 0
                            maxs0_length_to_most = img_array.shape[0] - 1
                        elif mins0_length_to_zero < 0:
                            maxs0_length_to_most = maxs0_length_to_most - mins0_length_to_zero
                            mins0_length_to_zero = 0
                        elif maxs0_length_to_most > img_array.shape[0]:
                            mins0_length_to_zero = mins0_length_to_zero - (maxs0_length_to_most - img_array.shape[0])
                            maxs0_length_to_most = img_array.shape[0] - 1

                        mins = [mins0_length_to_zero, mins1_length_to_zero]
                        maxs = [maxs0_length_to_most, maxs1_length_to_most]

                    imROI = img_array[mins[1]:maxs[1], mins[0]:maxs[0],:]

                    im = PIL.Image.fromarray(imROI)

                    continue

                # print(tmppoint)
                pointlist.append(tmppoint)
    return im


#
if __name__ == '__main__':
    # generate_datalist('/home/ran/MyProj/Data/Zhongnan_plaqueClassification')
    imROI_list = load_per_data(
        'data\\AllPhoto\\Zhongnan_plaqueClassification\\mix\\1142_3.bmp',
    )
    imROI_list.show()

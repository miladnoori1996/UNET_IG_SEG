import os
import numpy as np
# import cv2
from tensorflow.python.keras.preprocessing import image
from tensorflow import keras


cats = {
 'void': [0, 1, 2, 3, 4, 5],
 'road': [7],
 'sidewalk': [8],
 'flat': [9, 10, 6],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
}

def get_data(mode):
    """
    Function get_data(mode) accepts a string mode (train, val, test) and groups the paths of
    the images from the img_folder together, while also doing the same task with the
    images from the gtimg_folder. this function return 2 lists of these groups.

    :param mode: (type : string): "train, "val", "test"
    :return: a list of ids of images and masks
    """
    if mode == 'train' or mode == 'val' or mode == 'test':
        img_folder_path = "cityscapes/leftImg8bit"
        gtimg_folder_path = "cityscapes/gtFine"
        x_paths = []
        y_paths = []
        tmp_img_folder_path = os.path.join(img_folder_path, mode)

        # walk helps finding all files in a directory
        # saving all the images in the img_folder
        for (path, _, files) in os.walk(tmp_img_folder_path):
            for file_name in files:
                if file_name.endswith('.png'):
                    x_paths.append(os.path.join(path, file_name))
        # saving all the images in the gtimg_folder
        idx = len(tmp_img_folder_path)
        for x_path in x_paths:
            y_paths.append(gtimg_folder_path + '/{}'.format(mode)+ x_path[idx:-15] + 'gtFine_labelIds.png')
        assert len(y_paths) == len(x_paths)
        return x_paths, y_paths
    else:
        print("please choose the right mode (train, val, test)")


class DataGen(keras.utils.Sequence):
    """
    this class is made for Data generation to map all the masks to the data we are going to use
    you need to define a mode for the class ('train', 'val', 'test') to generate data
    ex:
    data_gen = DataGen(mode='train')
    """
    def __init__(self, mode, batch_size=8, image_height=1024, image_width=2048, image_depth=3, split=False, amount=1):
        self.train_ids, self.mask_train_ids = get_data('train')
        self.val_ids, self.mask_val_ids = get_data('val')
        self.test_ids, self.mask_test_ids = get_data('test')
        self.split = split
        self.amount = amount
        if self.split:
            self.train_ids, self.mask_train_ids = self.train_ids[:amount], self.mask_train_ids[:amount]
            self.val_ids, self.mask_val_ids = self.val_ids[:amount], self.mask_val_ids[:amount]
            self.test_ids, self.mask_test_ids = self.test_ids[:amount], self.mask_test_ids[:amount]
        self.mode = mode
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.on_epoch_end()

    def __load__(self, id_num):
        """
         A function to load the corresponding both the image and the mask

        :param id_num: (type : int) the index of the ID in the list
        :return: 2 np.array, one for image, and one for masks
        """
        mode = self.mode
        if mode =='train':
            image_path = self.train_ids[id_num]
            mask_path = self.mask_train_ids[id_num]
        elif mode == 'val':
            image_path = self.val_ids[id_num]
            mask_path = self.mask_val_ids[id_num]
        elif mode == 'test':
            image_path = self.test_ids[id_num]
            mask_path = self.mask_test_ids[id_num]
        else:
            print("mode does not exist")
            return

        # print(self.mode, " img path : ", image_path)
        # print(self.mode, " mask path : ", mask_path)
        _image = image.img_to_array(image.load_img(image_path, target_size=(self.image_height, self.image_width)))/255.
        _mask = image.img_to_array(image.load_img(mask_path, color_mode="grayscale", target_size=(self.image_height, self.image_width)))
        mask = np.zeros((_mask.shape[0], _mask.shape[1], 10))
        for i in range(-1, 34):
            if i in cats['void']:
                mask[:,:,0] = np.logical_or(mask[:,:,0],(_mask[:,:,0]==i))
            elif i in cats['road']:
                mask[:,:,1] = np.logical_or(mask[:,:,1],(_mask[:,:,0]==i))
            elif i in cats['sidewalk']:
                mask[:,:,2] = np.logical_or(mask[:,:,2],(_mask[:,:,0]==i))
            elif i in cats['flat']:
                mask[:,:,3] = np.logical_or(mask[:,:,3],(_mask[:,:,0]==i))
            elif i in cats['construction']:
                mask[:,:,4] = np.logical_or(mask[:,:,4],(_mask[:,:,0]==i))
            elif i in cats['object']:
                mask[:,:,5] = np.logical_or(mask[:,:,5],(_mask[:,:,0]==i))
            elif i in cats['nature']:
                mask[:,:,6] = np.logical_or(mask[:,:,6], _mask[:,:,0]==i)
            elif i in cats['sky']:
                mask[:,:,7] = np.logical_or(mask[:,:,7],(_mask[:,:,0]==i))
            elif i in cats['human']:
                mask[:,:,8] = np.logical_or(mask[:,:,8],(_mask[:,:,0]==i))
            elif i in cats['vehicle']:
                mask[:,:,9] = np.logical_or(mask[:,:,9],(_mask[:,:,0]==i))
        return _image, mask


    def __getitem__(self, index):
        dic = {
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids
        }
        ids = dic[self.mode]
        if (index+1)*self.batch_size > len(ids):
            self.batch_size = len(ids) - index*self.batch_size
        files_batch = ids[index*self.batch_size : (index+1)*self.batch_size]
        image = np.zeros((self.batch_size, self.image_height, self.image_width, 3))
        mask = np.zeros((self.batch_size, self.image_height, self.image_width, 10))
        count = 0
        for id_name in files_batch:
            _img, _mask = self.__load__(ids.index(id_name))
            image[count] = _img
            mask[count] = _mask
            count += 1
        return image, mask


    def on_epoch_end(self):
        pass


    def __len__(self):
        mode = self.mode
        if mode =='train':
            return int(np.ceil(len(self.train_ids)/float(self.batch_size)))
        elif mode == 'val':
            return int(np.ceil(len(self.val_ids)/float(self.batch_size)))
        elif mode == 'test':
            return int(np.ceil(len(self.test_ids)/float(self.batch_size)))
        else:
            print("mode does not exist")

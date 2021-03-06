import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

INPUT_DATA='./flower_photos'
OUTPUT_FILE='./flower_processed_data.npy'

VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10

def create_image_lists(sess,testing_percentage,validation_percentage):
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]   #https://www.jianshu.com/p/bbad16822eab
                                                    #os.walk讲解，是一个迭代器,每次返回的是一个元组（[当前文件夹名],[文件夹下文件夹],[文件夹下文件]），然后迭代每一个文件夹
    is_root_dir=True
    training_images=[]
    training_labels=[]
    testing_images =[]
    testing_labels =[]
    validation_images=[]
    validation_labels=[]
    current_label=0

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        extensions=['jgp','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir)
        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))       #append是追加一个元素，extend是追加一个列表
        if not file_list:continue

    for file_name in file_list:
        image_raw_data=gfile.FastGFile(file_name,'rb').read()
        image=tf.image.decode_jpeg(image_raw_data)
        if image.dtype!=tf.float32:
            image=tf.image.convert_image_dtype(image,dtype=tf.float32)
        image=tf.image.resize_images(image,[299,299])
        image_value=sess.run(image)

        #随机划分数据集
        chance=np.random.randint(100)
        if chance<validation_percentage:
            validation_images.append(image_value)
            validation_labels.append(current_label)
        elif chance<(testing_percentage+validation_percentage):
            testing_images.append(image_value)
            testing_labels.append(current_label)
        else:
            training_images.append(image_value)
            training_labels.append(current_label)
    current_label+=1


    state=np.random.get_state()
    np.random.shuffle(training_images)                #保证打乱后的图和标签仍然是一一对应状态
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return  np.asarray([training_images,training_labels,validation_images,validation_labels,testing_images,testing_labels])


def main():
    with tf.Session() as sess:
        process_data=create_image_lists(sess,TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
        np.save(OUTPUT_FILE,process_data)

if __name__=='__main__': #在一个程序运行时，其__name__就变成了“__main__”这时这样的原理
    main()
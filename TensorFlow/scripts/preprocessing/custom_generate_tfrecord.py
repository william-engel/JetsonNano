import os
import xml.etree.ElementTree as ET #Efficient API for parsing and creating XML data
import pandas as pd
from collections import namedtuple
import tensorflow
from PIL import Image
import io
import tensorflow as tf

from object_detection.utils import dataset_util, label_map_util


labels_path = './TensorFlow/workspace/training_demo/annotations/label_map.pbtxt'
xml_path = './TensorFlow/workspace/training_demo/images/train/'

label_map = label_map_util.load_labelmap(labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map) # {'class1': 1, 'class2': 2, ...}


def class_text_to_int(classname):
    '''takes a classname and returns the classindex'''
    return label_map_dict[classname]


def xml_to_csv(xml_path):
    '''loops through all xml files and combines them into a csv file'''
    xml_list = []
    for xml_file in os.listdir(xml_path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(xml_path + xml_file) # read xml file
            root = tree.getroot() # the root element is the parent of all other elements
            for member in root.findall('object'): # loop through all objects marked with bbox
                value = (root.find('filename').text,    # filename
                        int(root.find('size')[0].text), # height
                        int(root.find('size')[1].text), # width
                        member[0].text,                 # class name
                        int(member[4][0].text),         # xmin
                        int(member[4][1].text),         # ymin
                        int(member[4][2].text),         # xmax
                        int(member[4][3].text)          # ymax
                        )
                xml_list.append(value)

            column_name = ['filename', 'width', 'height', 
                           'class', 'xmin', 'ymin', 'xmax', 'ymax'] # define headers
    xml_df = pd.DataFrame(xml_list, columns=column_name) # convert to pandas dataframe
    return xml_df

def split(df, group = 'filename'):
    '''
    each entry in the dataframe(df) represents a bbox
    df.groupby('filename') will group entrys by their filename
    so all bbox'es present in an image are grouped together

    since we grouped by filename gb.groups.keys() contains a list of unique filenames

    gb.groups contains the data for each bbox in the acording group
    with gb.get_group() we acess the data per group
       e.g.: group with two bbox'es
       filename             width  height  class       xmin   ymin xmax  ymax
    0  img_00_reshaped.jpg   4000    3000  Chartreuse  1514   352  2240  2921
    1  img_00_reshaped.jpg   4000    3000  Chartreuse   514   352   240   921
       group with one bbox
       filename             width  height  class       xmin   ymin xmax  ymax
    2  img_01_reshaped.jpg   4000    3000  Chartreuse  1087     1  1498  1297

    namedtuple('data', ['filename', 'object']) is an tprl with the name 'data'
    the first tupel index in named 'filename' and will store the filename
    the second tupel index is named 'object' and will store the csv data for one imagefile
    '''
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, image_path):

    encoded_jpg = open(os.path.join(image_path, group.filename), 'rb').read() # rb: read binary
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows(): # loop through all bbox entrys
        xmins.append(row['xmin'] / width)  # relativ position
        xmaxs.append(row['xmax'] / width) 
        ymins.append(row['ymin'] / height) 
        ymaxs.append(row['ymax'] / height) 
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example.SerializeToString()

xml_df = xml_to_csv(xml_path)
print(xml_df.head())

grouped = split(xml_df, 'filename')
print(grouped[0].filename)
print(grouped[0].object)

tfrecord_file = './TensorFlow/workspace/training_demo/annotations/train.record'
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for group in grouped:
        tf_example = create_tf_example(group, xml_path)
        writer.write(tf_example)
    
    


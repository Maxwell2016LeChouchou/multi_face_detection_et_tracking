'''
Implement and test  detection (localization)
'''

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))
import cv2

sys.path.append("/home/max/Downloads/MTCNN/models/research")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Uncomment the following two lines if need to use the Tensorflow visualization_unitls
#os.chdir(cwd+'/models')
#from object_detection.utils import visualization_utils as vis_util

class face_detection(object):
    def __init__(self):

        self.face_boxes = []
        
        #self.face_boxes = []

        os.chdir(cwd)
        
        #Tensorflow localization/detection model
        # Single-shot-dectection with mobile net architecture trained on COCO dataset
        
        #detect_model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
        detect_model_name = 'ckpt_data_ssd_mobilenet_v1_coco_FDDB'
        #detect_model_name = '/home/max/Desktop/files/ckpt_data_ssd_inception_v2_coco'
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
        
        # PATH_TO_LABELS = '/home/maxwell/Downloads/tf-face-detector-master/data/fddb_label_map.pbtxt'
        # NUM_CLASSES = 2

        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')
               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

        # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)
    
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)       
        
    def get_localization(self, image, visual=False):  
        
        """Determines the locations of the faces in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """

        #category_index={1: {'id': 1, 'name': u'face'}} # WIDERFACE
        category_index={1: {'id': 1, 'name': u'face'},     #FDDB
                         2: {'id': 2, 'name': u'eye'}}      #FDDB       
        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            # print('num_detections:{}'.format(num_detections))
            # print(scores.min(), scores.max())
            # print(classes.min(), classes.max())
            # visual = True
            if visual == True:
                #image.flags.writeable = True
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,min_score_thresh=.4,
                    line_thickness=3)

                plt.figure(figsize=(9,6))
                plt.imshow(image)
                plt.show()  
              
            boxes=np.squeeze(boxes)
            classes =np.squeeze(classes)
            scores = np.squeeze(scores)
    
            cls = classes.tolist()
              
            idx_vec = [i for i, v in enumerate(cls) if ((v==1) and (scores[i]>0.3))]
              
            if len(idx_vec) ==0:
                print('no detection!')
                self.face_boxes = []  
            else:
                tmp_face_boxes=[]
                for idx in idx_vec:
                    dim = image.shape[0:2]
                    box = self.box_normal_to_pixel(boxes[idx], dim)
                    box_h = box[2] - box[0]
                    box_w = box[3] - box[1]
                    ratio = box_h/(box_w + 0.01)
                      
                    if ((ratio > 1.0) and (box_h>20) and (box_w>20)):
                        tmp_face_boxes.append(box)
                        print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                         
                    else:
                        print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)
                          
            # idx_vec = [i for i, v in enumerate(cls) if ((v==1) and (scores[i]>0.3))]
            # if len(idx_vec) == 0:
            #     print('No detection')
            #     self.face_boxes = []
            # else: 
            #     tmp_face_boxes=[]
            #     for idx in idx_vec:
            #         dim = image.shape[0:2]
            #         box = self.box_normal_to_pixel(boxes[idx],dim)
            #         box_h = box[2] - box[0]
            #         box_w = box[3] - box[1]
            #         ratio = box_h/(box_w + 0.01)
                      
            #         if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
            #             tmp_face_boxes.append(box)
            #             print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                         
            #         else:
            #             print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)                    

                  
                self.face_boxes = tmp_face_boxes
             
        return self.face_boxes
        
if __name__ == '__main__':
        # Test the performance of the detector
        det =face_detection()
        #os.chdir(cwd)
        #TEST_IMAGE_PATHS= glob(os.path.join('test_face/', '*.jpg'))


        path_to_test_images_dir = '/home/max/Downloads/MTCNN/multi_face_detection_et_tracking/2/'
        for image_file in sorted(os.listdir(path_to_test_images_dir)):
            images = os.path.join(path_to_test_images_dir,image_file)
            print(images)

        
        #for i, image_path in enumerate(TEST_IMAGE_PATHS[0:22]):
         
            print('')
            print('*************************************************')
            plt.ion()

            img_full = Image.open(images)
            image = det.load_image_into_numpy_array(img_full)
            img_full_np_copy = np.copy(image)
            start = time.time()
            b = det.get_localization(image, visual=False)
            end = time.time()
            # cv2.imshow("image", image)
            # cv2.waitKey(50)
            plt.imshow(image)
            plt.show()
            plt.pause(0.5)
            plt.close()
            print('Localization time: ', end-start)
           
            

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from datetime import datetime
import pymysql.cursors
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.48, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

import socket
import time

# 서버 주소 설정
# HOST = '192.168.0.114'
# PORT = 6666

# 데이터베이스 설정
conn = pymysql.connect(
    host = 'localhost',
    user = 'root',
    passwd = 'smartlab',
    db = 'shop_floor',
    charset='utf8'
)
cursor = conn.cursor()

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # 비디오 해상도 설정
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # start line
    start1 = (1713,385)
    start2 = (1813,385)
    # end line
    end1 = (915,800)
    end2 = (915,900)

    # dictionary
    dictionary = {}

    # cycletime
    cyclelist = []
    basket_cyclelist = []
    soccer_cyclelist = []
    global total_cycletime, basket_cycletime, soccer_cycletime
    total_cycletime = 0
    basket_cycletime = 0
    soccer_cycletime = 0

    # WIP list
    wiplist = []

    # 지표 변수 선언
    global in_count, out_count, warning1, warning2, th, basket_in, basket_out, soccer_in, soccer_out
    in_count = 0
    out_count = 0
    th = 0
    warning1 = 0
    warning2 = 0
    basket_in = 0
    basket_out = 0
    soccer_in = 0
    soccer_out = 0

############# client setting ###########
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((HOST, PORT))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        ## 로그 출력할 때
        # with open('Tracker_log.txt', 'a') as f:
        #     f.write('Frame : {}'.format(frame_num)+"\n")
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['B','S']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        basketball = []
        soccerball = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name == "B":
                basketball.append(class_name)
            elif class_name == "S":
                soccerball.append(class_name)

            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        # WIP 공별 구분
        basketball_count = len(np.array(basketball))
        soccerball_count = len(np.array(soccerball))
        count = len(names)

        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # draw start line
        cv2.line(frame,start1,start2,(0,255,0),3)
        # draw end line
        cv2.line(frame,end1,end2,(0,255,0),3)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            # 공에 따라 색깔 구분
            if class_name == "B":
                color = (97,237,124)
            else:
                color = (90, 160, 230)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*21, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.70, (255,255,255),2)
            # 공의 중심 계산
            center_x = int((int(bbox[0]) + int(bbox[2])) / 2)
            center_y = int((int(bbox[1]) + int(bbox[3])) / 2)

            # DB insert
            tnow = datetime.today().strftime("%Y%m%d%H%M%S")
            sql = "insert into ball(ID,x_coordinate,y_coordinate,frame,Date) values({0},{1},{2},{3},{4});".format(track.track_id, center_x, center_y, frame_num, tnow)
            cursor.execute(sql)
            conn.commit()

            # 중심좌표로 start line부터 추적
            if (start1[0] <= center_x <= start2[0]) and (start1[1] - 3 <= center_y <= start2[1] + 3):
                if (class_name + "-" + str(track.track_id)) not in wiplist:
                    wiplist.append(class_name + "-" + str(track.track_id))
                    cv2.line(frame,start1,start2,(0,0,0),3)
                    dictionary[class_name + "-" + str(track.track_id)] = time.time()
                    if class_name == "B":
                        basket_in += 1
                    else:
                        soccer_in += 1
                    in_count += 1


            # end line 추적

            if (end1[0] - 7 <= center_x <= end1[0] + 5) and (end1[1] <= center_y <= end2[1]):
                if (class_name + "-" + str(track.track_id)) in wiplist:
                    wiplist.remove(class_name + "-" + str(track.track_id))
                    # donelist.append(class_name + "-" + str(track.track_id))
                    cv2.line(frame,end1,end2,(0,0,0),3)
                    cycletime = time.time() - dictionary.get(class_name + "-" + str(track.track_id))
                    cyclelist.append(cycletime)

                    if class_name == "B":
                        basket_cyclelist.append(cycletime)
                        basket_out += 1
                    else:
                        soccer_cyclelist.append(cycletime)
                        soccer_out += 1

                    del dictionary[class_name + "-" + str(track.track_id)]
                    out_count += 1

            # calculate cycletime
            if len(cyclelist) == 0:
                total_cycletime = 0

            else:
                total_cycletime = round((sum(cyclelist) / len(cyclelist)), 3)

            if len(basket_cyclelist) == 0:
                basket_cycletime = 0

            else:
                basket_cycletime = round((sum(basket_cyclelist) / len(basket_cyclelist)), 3)

            if len(soccer_cyclelist) == 0:
                soccer_cycletime = 0

            else:
                soccer_cycletime = round((sum(soccer_cyclelist) / len(soccer_cyclelist)), 3)


            # error zone 1
            if (1235 < center_x < 1355) and (505 < center_y < 615):
                cv2.rectangle(frame, (1235, 505), (1355, 615), (255, 0, 0), 3)
                cv2.putText(frame, "ERROR", (1240, 635), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 0, 0), 2)
                warning1 = 1
            else:
                warning1 = 0

            # error zone 2
            if (1260 < center_x < 1380) and (810 < center_y < 920):
                cv2.rectangle(frame, (1260, 810), (1380, 920), (255,0,0), 3)
                cv2.putText(frame, "ERROR", (1265, 940), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255,0,0), 2)
                warning2 = 1
            else:
                warning2 = 0


            # # Write txt file(로그 출력)
            # with open('Tracker_log.txt', 'a') as f:
            #     f.write("Tracker ID: {}, {}".format(str(track.track_id), (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))+"\n")

        #지표 그리기(테스트)
        cv2.putText(frame, "Cycletime: Total {} (sec)".format(total_cycletime),(20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "B: Cycletime: {} (sec)".format(basket_cycletime), (150, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0, 255, 0), 2)
        cv2.putText(frame, "S: Cycletime: {} (sec)".format(soccer_cycletime), (150, 140),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "WIP: Total {} (ea)".format(count), (25, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "B: {} (ea)".format(basketball_count), (147,250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0), 2)
        cv2.putText(frame, "S: {} (ea)".format(soccerball_count), (147, 290), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "IN:  Total {} (ea)".format(in_count), (25, 360), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "B: {} (ea)".format(basket_in), (142, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "S: {} (ea)".format(soccer_in), (142, 440), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
        cv2.putText(frame, "OUT: Total {} (ea)".format(out_count), (16, 510), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.putText(frame, "B: {} (ea)".format(basket_out), (150, 550), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
        cv2.putText(frame, "S: {} (ea)".format(soccer_out), (150, 590), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)



       # send message to server
        #message = "{} {} {} {} ".format(in_count, out_count, total_cycletime, count) # 공 구분하지 않을 때

        #message = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} ".format(in_count, out_count, total_cycletime, count, warning1, warning2, basket_in, basket_out, basket_cycletime, basketball_count,soccer_in, soccer_out, soccer_cycletime, soccerball_count)

        #client_socket.send(message.encode())

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        # if FLAGS.output:
        #     out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

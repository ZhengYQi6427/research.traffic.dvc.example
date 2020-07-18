import cv2
import os
import glob
'''
Description: this script is used to create test dataset for darknet
'''
def split_frames(video):
    try: 
        video_name = os.path.basename(video)
        os.mkdir(video_path+'frame_dir_'+video_name)
    except FileExistsError:
        print('dir existed')
    vc=cv2.VideoCapture(video_path+video_name)
    c=0
    digits_str = ''
    result_dir = video_path+'frame_dir_'+video_name
    list_file = open('test_'+video_name+'.txt', 'w')
    if vc.isOpened():
        rval,frame=vc.read()
        print('hh')
    else:
        rval=False

    while rval:
        rval,frame=vc.read()
        if c<10:
            digits_str = '0000'
        elif c<100:
            digits_str = '000'
        elif c<1000:
            digits_str = '00'
        else:
            digits_str = '0'

        try :
            cv2.imwrite(result_dir+'/frame_'+digits_str+str(c)+'.jpg',frame)
        except cv2.error:
            print('cv2 error ignored')
            continue
        getpath = os.path.abspath(result_dir+'/frame_'+digits_str+str(c)+'.jpg')
        list_file.write(getpath+'\n')
        c=c+1
        cv2.waitKey(1)
    list_file.close()
    vc.release()
    
#video_path = './2020summer_data_inference/video_source/'
video_path = '/home/ty2417/videos/'
list_video = glob.glob(video_path+'*.mp4')
for i in list_video:
    print(i)
    split_frames(i)
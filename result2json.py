import os
import json
import numpy as np
import glob
import re

def file_init(file_path):
    result_list = glob.glob(file_path)
    result_list.sort()
    result_dic = {}
    return result_list,result_dic
def voc_maskrcnn(voc_result_dir,maskrcnn_result_dir,video_name,frame_name_path):
    '''
    Description: This is the format transform script(from voc output to maskrcnn output)
    arg video_name: video name
    arg voc_result_dir: directory containg all the .txt detection results
    arg maskrcnn_result_dir: directory to save the transformed result
    arg: frame_name_path: the txt file containing all the frame names
    '''
    file_path = voc_result_dir+'/*.txt'
    print(glob.glob(file_path))
    filename = os.path.join(maskrcnn_result_dir,video_name+'.json')
    print('the filename is: ', filename)
    result_list,result_dic = file_init(file_path)
    if not result_list:
        print('empty list, please check your file path')

    with open(frame_name_path) as f:
        for line in f:
            line_info = line.replace('\n',' ').split(' ')
            path = os.path.basename(line_info[0])
            temp = re.findall(r'\d+', path) 
            res = list(map(int, temp)) 
            result_dic[res[0]] = {'class_ids': [],'rois':[],'scores':[]}  #create place holder for the result dict
    for i in range(len(result_list)): 
        if 'bg' in result_list[i]:
            print('skip bg class:',result_list[i])
#             break#different tag
        with open(result_list[i]) as f:
    #         result_dic[i] = {'class_ids': [],'rois':[],'scores':[]}
            for line in f:
                line_info = line.replace('\n',' ').split(' ') #remove /n and space
                temp = re.findall(r'\d+', line_info[0]) 
                res = list(map(int, temp)) # fetch frame id
                if float(line_info[1]) < 0.25:
                    continue
                if i ==2:
                     result_dic[res[0]]['class_ids'].append(0)
                else:
                    result_dic[res[0]]['class_ids'].append(i+1)
                    result_dic[res[0]]['scores'].append(line_info[1])
                    result_dic[res[0]]['rois'].append([float(line_info[3]),float(line_info[2]),float(line_info[5]),float(line_info[4])])
    if filename:
            # Writing JSON data
        with open(filename, 'w') as f:
            json.dump(result_dic, f,indent=4)
            print('create output %s '%(filename))

vvoc_result_dir = './darknet/results' #dir that containing all the txt results
maskrcnn_result_dir = './result_in_json/' #dir that you want to save the result

with open("./config/traffic.data", "r") as f:
    for line in f:
        word = line.split()
        #print('this is line:', word)
        if word[0]=='valid':
            frame_name_path = word[2]

# frame_name_path = '~/test_traffic_video_HIKL1D190911T153514_20190920_0700_0830_90sec_calibration.mp4.txt' #txt file contains all the frames
video_name = 'res_for_eval' # inference video name or anything you want to name the json file
voc_maskrcnn(vvoc_result_dir,maskrcnn_result_dir,video_name,frame_name_path)

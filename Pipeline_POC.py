# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:52:34 2020

@author: oscar.frausto.perez
"""

import cv2
import numpy as np
from darkflow.net.build import TFNet
import sys
import os

sys.path.insert(0, "../utils")
from postprocessing import draw_bb,save_current_products_count,draw_count
from timestamp import get_time
from name import get_name
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation #animation 

from lowpass_filter import butter_lowpass_filter
from level_detection import pipeline_level_detection

def animate(j,dictionary):
    ''' create animation from plt.figure
    Parameters
    ----------
        j:            Actual frame
        dictionary:   Dictionary with the label and the count until the j-th frame
    '''
    for i, (key,value) in enumerate(dictionary.items()):  
        ax[i].clear()
        ax[i].plot(value[:j],color=colors_ax_dic[key])
        ax[i].legend([key + '  : {}'.format(value[j])],loc="upper right")        
        ax[i].set_ylim([0,25])
        ax[i].set_yticklabels([])      
        ax[i].set_axisbelow(True)
        ax[i].minorticks_on()
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        #ax[i].yaxis.set_minor_locator(AutoMinorLocator(20))
    plt.xlabel('Frames')
    ax[2].set(ylabel='Cantidad de Productos')
    

# filter order        
order = 2
# sample rate, Hz    
fs = 20  
# desired cutoff frequency of the filter, Hz    
cutoff = 0.5  

labels = ['cocaregular25','fantaregular25','fantasinazucar30','spriteregular15','powerade','altodelcarmen1'] 
colors = [(255,0,0),(0,255,0),(255,250,205),(255,255,0),(0,255,255),(255,0,255)]
colors_dic = {l:c for l,c in zip(labels,colors) }

# confidence 
threshold = 0.35 #0.3
model_name = "yolov2-tiny-voc-6c-ds3-4-original-da-2"

options = {"model": "cfg/yolov2-tiny-voc-6c.cfg", 
            "pbLoad":  "built_graph/{}.pb".format(model_name ),
            "metaLoad": "built_graph/{}.meta".format(model_name),
            "threshold" : threshold ,
            "gpu": 0.5}


# model_name = 2628  # ds3 y ds4
# #model_name = 3031  # con imagenes del video con data augmentation
# #model_name = 3511  # con imagenes del video con data augmentation y threshold = 0.5 en entrenamiento
# #model_name  = 3217  # ds3 y ds4 y treshold = 0.5
# #model_name  = 3375 # ds3 y ds4 y treshold = 0.5 con imagenes del video con data augmentation
# options = {"model": "cfg/yolov2-tiny-voc-6c.cfg", 
#             "load": model_name,
#             "threshold" : threshold ,
#             "gpu": 0.5}


# load options
tfnet = TFNet(options)
# intersection over union threshold
iou_threshold = 0.3
scale_boundingbox = 0.9

videos_path = 'C:/Users/oscar.frausto.perez/Accenture/Reyna, Daniel - POC_Picking/videos/test/17-06-dataset4/videos/'
#videos_path = 'C:/Users/oscar.frausto.perez/Accenture/Reyna, Daniel - POC_Picking/videos/test/10-06-dataset3/videos/'
video_name = videos_path + 'video1.mp4'

if os.path.isfile(video_name ) is False:
    print("The video does not exist")
    sys.exit(2)
        
# source from video    
capture = cv2.VideoCapture(video_name)
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
# to scale video output
factor = 0.5

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
out_video_name = video_name[:-4]+'_'+str(model_name)+'_' + str(threshold).replace('.', '_')+'_'+ str(iou_threshold).replace('.', '_')+'_'+ get_time()+ '.mp4'
fps=10

# do you want to save the video
save_video = False
if save_video:    
    out = cv2.VideoWriter(out_video_name, fourcc,fps,(frame_width,frame_height)) # Image resolution
# dou you want to plot and save the monitoring counting (image)
plot_monitoring = False
# dou you want to animate and save the monitoring counting (video)
animate_monitoring = False



# the frames from the video must be scaled
frame_width_net = 980
frame_height_net = 980
# area of interest
#area = [180,60,800,900] # xmin, ymin, xmax, ymax

area = [800,180,900,60] #xmin,ymin, xmax,ymax

frame_video =  np.zeros((frame_height,frame_width,3),dtype=np.uint8)


# store the time line by level
objects_count_frame = defaultdict(list)
# store all the time line
objects_count_frame_aux = defaultdict(list)
# store values (int) for each level
objects_count_total = dict.fromkeys(colors_dic.keys(), 0)
    
# show the current count of products
x_y= (10,540)
frame_video = draw_count(frame_video, objects_count_total,colors_dic,'Total products',x_y ) 

# show the video?
show_video = True
results_name = get_name(video_name) + '_results_'+get_time()  

# store the total count of producta of the current frame 
no_objects_list = []

#how many frames without any object have to happen to detect the new level
frames_no_objects = 40 # frames sin objetcos 80
level_count = 0
f=0 # frame

# while thera are new frames from the video
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()        
    if frame is None:    
        break
    #resize the frame to 980x980
    frame = cv2.resize(frame,(frame_width_net,frame_height_net ))
    #select region of interest
    #frame_roi = frame[60:900,180:800,:]    
    frame_roi = frame[area[3]:area[2],area[1]:area[0],:]
    frame = frame_roi.copy()
    #resize the frame to 980x980
    frame = cv2.resize(frame,(980,980))
    # get prediction
    results = tfnet.return_predict(frame)

    # draw bounding boxes and return current objects count
    frame_1,objects_count = draw_bb(frame.copy(), results,colors_dic,iou_threshold = iou_threshold, scale_boundingbox=scale_boundingbox) 
                      
    frame_minus_1 = frame_1.copy()
    
    no_objects_frame = 0
    # iterate over key-value pair (label-number of prodcuts)
    for key, value in objects_count.items():
        objects_count_frame[key].append(value)
        objects_count_frame_aux[key].append(value)
        no_objects_frame += value 
        

    no_objects_list.append(no_objects_frame)
    # verify if there is a new level
    level_state = pipeline_level_detection(frame.copy()[:,:,::-1])
    
    #  number of objects in the last frames 
    no_objects_list_last = no_objects_list[-frames_no_objects:]
    #print(level_state)
    #print(no_objects_list_last)
    #print(len(no_objects_list_last))
    
    # if there is a level detecte and in the last "frames_no_objects" frames dont there were not any products
    if level_state is True and no_objects_list_last.count(0) == len(no_objects_list_last) and len(no_objects_list_last) >= frames_no_objects:
        print('New level')
        # get the current number of objects
        objects_count_total = save_current_products_count(objects_count_total,objects_count_frame, cutoff, fs, order)      
        objects_count_frame = defaultdict(list)
        del no_objects_list[:]
        level_count = frames_no_objects
      

    if level_count > 0 and len(results) == 0:    
        frame_1 = cv2.putText(frame_1,'Level detected',(int(frame_width_net/2)-200,int(frame_height_net/2)-50 ),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        level_count-=1
    else :
        level_count = 0
    
    # size of area to show the video proccesed 
    w,h= int(frame_width * 0.625),int(frame_height*0.75)#     1200,810
    frame_1 = cv2.resize(frame_1,(w,h))
    col=int((frame_width - w)/2)
    fil = int((frame_height - h)/2)
    frame_video[fil:-fil,col :-col,:] = frame_1 
    
    # draw legend
    x_y= (10,40)
    frame_video = draw_count(frame_video, objects_count,colors_dic,'Current Products',x_y ) 
    
    # draw legend
    x_y= (10,540)
    frame_video = draw_count(frame_video, objects_count_total,colors_dic,'Total products',x_y ) 
       
    if show_video is True:
          frame_resized = cv2.resize(frame_video, (int(frame_width*factor), int(frame_height*factor) ))  
          cv2.imshow('frame',frame_resized)
          key = cv2.waitKey(1)
          if  key == ord('q'):
              capture.release()
              if save_video: 
                  out.release()
              break

    if save_video:     
        out.write(cv2.resize(frame_video,(frame_width,frame_height)))
    time_processing = time.time()-stime
    print('{:f}'.format(time_processing))
    #print('FPS {:.1f}'.format(1/(time.time()-stime)))    
    frame_video[:,:,:] = 0
    f+=1

# get the current number of objects
objects_count_total = save_current_products_count(objects_count_total,objects_count_frame, cutoff, fs, order )

w,h=1200,810
frame_1 = cv2.resize(frame_minus_1,(w,h))
col=int((frame_width - w)/2)
fil = int((frame_height - h)/2)
frame_video[fil:-fil,col :-col,:] = frame_1 

# draw legend
x_y= (10,40)
frame_video = draw_count(frame_video, objects_count,colors_dic,'Current Products',x_y ) 
x_y= (10,540)
frame_video = draw_count(frame_video, objects_count_total,colors_dic,'Total products',x_y )       

if show_video is True:
    frame_resized = cv2.resize(frame_video, (int(frame_width*factor), int(frame_height*factor) ))  
    cv2.imshow('frame',frame_resized)

if save_video: 
    out.write(cv2.resize(frame_video,(frame_width,frame_height)))

total=''.join(f'{key} : {value}\n' for  key, value in objects_count_total.items())
#cv2.waitKey(0)
capture.release()
if save_video: 
    out.release()
cv2.destroyAllWindows()
plt.close('all')



# #********************************* Plot counting monitoring ******************************
if plot_monitoring:
    color_ax=['b','g','k','c','y','m']
    colors_ax_dic = {l:c for l,c in zip(labels,color_ax) }
    
    fig, ax = plt.subplots(len(objects_count_frame_aux), sharex=True, sharey=True)
    ax = ax.ravel()
    ax[0].set_title('Products count/frame ')
    for i, (key, value) in enumerate(objects_count_frame_aux.items()):  
        ax[i].plot(value,color=colors_ax_dic[key])
        ax[i].legend([key])
        ax[i].grid()
    plt.savefig(out_video_name[:-4]+'_original_image.png')

    fig, ax = plt.subplots(len(objects_count_frame_aux), sharex=True, sharey=True)
    ax = ax.ravel()
    ax[0].set_title('Products count/frame filtered')
    for i, (key, value) in enumerate(objects_count_frame_aux.items()):  
        ax[i].plot(butter_lowpass_filter(value, cutoff, fs, order),color=colors_ax_dic[key])
        ax[i].legend([key])
        ax[i].grid(axis='both')
        #print(np.max(butter_lowpass_filter(value, cutoff, fs, order)))
    plt.savefig(out_video_name[:-4]+'_filtered_image.png')
    plt.close('all')




#******************************** Show and save counting monitoring animation **************************************
if animate_monitoring:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    color_ax=['b','g','k','c','y','m']
    colors_ax_dic = {l:c for l,c in zip(labels,color_ax) }

    fig1, ax = plt.subplots(len(colors_ax_dic),sharex=True)
    fig1.suptitle('Picking Monitoring')
    ani = animation.FuncAnimation(fig1, animate,fargs = (objects_count_frame_aux,),  frames=len (objects_count_frame_aux['cocaregular25']))
    #plt.show()
    out_video_gif_name = out_video_name[:-4]+'_plot_'+'.mp4'
    ani.save(out_video_gif_name, writer=writer)  

    objects_count_frame_aux_filter = defaultdict(list)
    for i, (key, value) in enumerate(objects_count_frame_aux.items()):  
        objects_count_frame_aux_filter[key] = np.round(butter_lowpass_filter(value, cutoff, fs, order),0).astype(int)
    fig1, ax = plt.subplots(len(colors_ax_dic),sharex=True)
    fig1.suptitle('Picking Monitoring (Filter)')
    ani = animation.FuncAnimation(fig1, animate,fargs = (objects_count_frame_aux_filter,) ,frames=len (objects_count_frame_aux_filter['cocaregular25']))
    #plt.show()
    out_video_gif_name = out_video_name[:-4]+'_plot_filter'+'.mp4'
    ani.save(out_video_gif_name, writer=writer)  


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:12:43 2020

@author: oscar.frausto.perez
"""
import cv2
import numpy as np

def filter_bb(bounding_boxes,iou_threshold):
    
    ''' Eliminate bounding boxes inside other bouding boxes
    ----------
        bounding_boxes:   array the bounding boxes coordinates
    Returns
    -------
        ind :     index of  bounding boxes that are not inside other bouding boxes
    '''
    
    # calculate area per bounding boxes
    x2_x1 = bounding_boxes[:,2]-bounding_boxes[:,0]+1
    y2_y1 = bounding_boxes[:,3]-bounding_boxes[:,1] +1 
    b = np.maximum(0, x2_x1)
    h = np.maximum(0, y2_y1) 
    areaBoundingBoxes = b * h
    
    # create an empty list where indexes suppressed will be stored
    supress = []
    
    # iterate over each boudnind box area and its correspondent coordinates
    # compare each bounding box with the rest of bounding boxes 
    for i, (area_bb, bb)  in enumerate (zip(areaBoundingBoxes,bounding_boxes)):
        # assuming that all bounding boxes are not inside other bouding boxes
        mask = np.ones(len(bounding_boxes),dtype=bool)
        
        # calculate intersection area between bounding boxes 
        xA = np.maximum(bb[0], bounding_boxes[:,0])
        yA = np.maximum(bb[1], bounding_boxes[:,1])
        xB = np.minimum(bb[2], bounding_boxes[:,2])
        yB = np.minimum(bb[3], bounding_boxes[:,3])
        xA_xB = xB - xA + 1
        yA_yB = yB - yA + 1
        b = np.maximum(0, xA_xB)
        h = np.maximum(0, yA_yB) 
        interArea = b * h        
        
        # calculate union area between current box and the rest
        unionArea = area_bb + areaBoundingBoxes - interArea
        # get IoU
        iou = interArea / unionArea
        
        # Do not compare with itself
        mask[i] = 0
       
        # if the iou is greater than the threshold at leat two cases
        if (iou[mask] >= iou_threshold).sum() >=2 :
            supress.append(i)         
        
        # if the bounding box area is equal to intersection area means that bounding box  is  inside the another bouding box
        # and needs to be supressed
        if (area_bb in  interArea[mask]):
            supress.append(i) 
    # create index and eliminate and apply suppres value
    ind = np.ones(len(bounding_boxes),dtype=bool)   
    ind[supress] = 0
    return ind
    
    

def draw_bb(img, results,colors,iou_threshold = 0.20):
    ''' 
    Draw bounding boxes on the image
    ----------
        img :         image
        results :     model output containing each prediction (label, confidence and bounding box coordinates)
        colors:       dicctionary of colors
    Returns
    -------
        img :         image with bounding boxes drawn 
    '''
    objects_count = dict.fromkeys(colors.keys(), 0)
    
    
    
    # if there are at least 2 bounding boxes apply eliminate_inner_bb fucntion 
    if  len(results) >= 2 :
        bounding_boxes = np.array([ [r['topleft']['x'] ,r['topleft']['y'],r['bottomright']['x'],r['bottomright']['y'] ] for r in results ] )         
        bounding_boxes_indices = filter_bb(bounding_boxes,iou_threshold)  
        if False in bounding_boxes_indices:
            results = [r for r,bbi in zip(results, bounding_boxes_indices) if bbi]
    # iterate over each results

    
    for r in results:
        # get confidence
        confidence = r['confidence']*100
        #if confidence < 15:
        #    continue
        xmin_ymin = (r['topleft']['x'],r['topleft']['y'])
        xmax_ymax = (r['bottomright']['x'],r['bottomright']['y'])
        label = r['label'] 
        
        text_label = '{} '.format(label)
        conf_label = '{:.0f}%'.format(confidence)
        img = cv2.rectangle(img,xmin_ymin,xmax_ymax,colors[label],3)
        img = cv2.putText(img,text_label,xmin_ymin,cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        xmin_ymin = (r['topleft']['x']+120,r['topleft']['y'] + 60)
        #img = cv2.putText(img,conf_label,xmin_ymin,cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        
        objects_count[label] += 1
    
        
        
    img = cv2.putText(img,'Current Products',(0,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),3)
    x,y = 0,80
    for key, value in objects_count.items():
        img = cv2.putText(img,'{} : {}'.format(key,value),(x,y),cv2.FONT_HERSHEY_COMPLEX,1,colors[key],3)
        y += 40
        
    return img#cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
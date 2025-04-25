'''
Implementation of Energy-based Pointing Game proposed in Score-CAM.
Adjusted by Marcel Kleinmann
'''

import torch

'''
bbox (list): upper left and lower right coordinates of object bounding box
saliency_map (array): explanation map, ignore the channel
'''
def energy_point_game(bbox, saliency_map, threshold=None):
  
  x1, y1, x2, y2 = bbox
  w, h = saliency_map.shape
  
  if threshold is not None:
    max_val = saliency_map.max()
    thresh_val = threshold * max_val
    saliency_map = torch.where(saliency_map >= thresh_val, 
                              saliency_map, 
                              torch.zeros_like(saliency_map))

  
  empty = torch.zeros((w, h))
  empty[y1:y2, x1:x2] = 1 
  mask_bbox = saliency_map * empty  
  
  energy_bbox =  mask_bbox.sum()
  energy_whole = saliency_map.sum()
  
  proportion = energy_bbox / energy_whole
  
  return proportion



def energy_point_game_recall(bbox, saliency_map, threshold=0):
  
  x1, y1, x2, y2 = bbox
  w, h = saliency_map.shape
  
  bounding_box_map = saliency_map[y1:y2,x1:x2]
  
  if threshold is not None:
    max_val = saliency_map.max()
    thresh_val = threshold * max_val
    bounding_box_map = torch.where(bounding_box_map >= thresh_val, 
                              bounding_box_map, 
                              torch.zeros_like(bounding_box_map))
  
  full_bbox_energy = torch.zeros((w, h))
  full_bbox_energy[y1:y2, x1:x2] = 1 
  mask_bbox = saliency_map * full_bbox_energy  
  
  energy_bbox =  mask_bbox.abs().sum()
  
  #print(f"Mask_Bbox = {energy_bbox} and bounding_box_map = {bounding_box_map.sum()}")

  proportion = bounding_box_map.sum() / energy_bbox
  #print(proportion)
  
  return proportion

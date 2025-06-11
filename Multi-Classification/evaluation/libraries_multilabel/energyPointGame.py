'''
Implementation of Energy-based Pointing Game proposed in Score-CAM.
'''

import torch

'''
bbox (list): upper left and lower right coordinates of object bounding box
saliency_map (array): explanation map, ignore the channel
'''
def energy_point_game(bbox, saliency_map):
  
  x1, y1, x2, y2 = bbox
  w, h = saliency_map.shape
  
  empty = torch.zeros((w, h))
  empty[y1:y2, x1:x2] = 1 # changed?

  #empty[x1:x2, y1:y2] = 1
  mask_bbox = saliency_map * empty  
  
  energy_bbox =  mask_bbox.sum()
  energy_whole = saliency_map.sum()
  #print(f"BBox: {energy_bbox}, Whole: {energy_whole}")
  
  proportion = energy_bbox / energy_whole
  
  return proportion

def energy_point_game_mask(mask, saliency_map, threshold=0):
    """
    mask: Precomputed mask with 1s in ALL target regions
    saliency_map: Explanation heatmap
    """
    assert mask.shape == saliency_map.shape, "Mask/saliency shape mismatch"
    if threshold is not None:
      max_val = saliency_map.max()
      thresh_val = threshold * max_val
      saliency_map = torch.where(saliency_map >= thresh_val, 
                                saliency_map, 
                                torch.zeros_like(saliency_map))

    
    masked_energy = (saliency_map * mask).sum()
    total_energy = saliency_map.sum()
    
    return masked_energy.item() / total_energy.item()

def energy_point_game_recall(bbox, saliency_map, threshold=0):
  bounding_box_map = bbox * saliency_map 
  energy_bbox =  bounding_box_map.abs().sum() # calculate before removal of negative values
  
  if threshold is not None:
    max_val = saliency_map.max()
    thresh_val = threshold * max_val
    bounding_box_map = torch.where(bounding_box_map >= thresh_val, 
                              bounding_box_map, 
                              torch.zeros_like(bounding_box_map))
  
  
  
  proportion = bounding_box_map.sum() / energy_bbox  
  return proportion.item()

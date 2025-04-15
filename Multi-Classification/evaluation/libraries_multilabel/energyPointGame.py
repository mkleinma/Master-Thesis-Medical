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

def energy_point_game_mask(mask, saliency_map):
    """
    mask: Precomputed mask with 1s in ALL target regions
    saliency_map: Explanation heatmap
    """
    assert mask.shape == saliency_map.shape, "Mask/saliency shape mismatch"
    
    masked_energy = (saliency_map * mask).sum()
    total_energy = saliency_map.sum()
    
    return masked_energy.item() / total_energy.item()

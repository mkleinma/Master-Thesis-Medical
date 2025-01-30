import os

def find_original_image_name(dicom_folder, my_image):
    parts = my_image.split("_")
    
    filename = os.path.basename(my_image)
    
    # Split the filename by "_" and take the first part
    first_part = filename.split("_")[0]
    original_image_name = os.path.join(dicom_folder, first_part)
    print(original_image_name)
    return original_image_name

    
    
def get_expl_img_bbox_coordinates():
    print()
    
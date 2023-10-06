import os
from PIL import Image

# Define the cut rate
rate = 2
data_dir = "lab/original_process/original_images"
# Get the current directory
current_dir = os.getcwd()
current_dir = os.path.join(current_dir,data_dir)
# Get a list of all files in the current directory
files = os.listdir(current_dir)
output_dir = os.path.join(current_dir, "../../extract_images_200fps")
os.makedirs(output_dir, exist_ok=True)

# Filter out only the JPG image files
image_files = [file for file in files if file.endswith("png")]
image_files = sorted(image_files)
image_number = len(image_files)
j=0
for i in range(0,image_number,rate):
    file = image_files[i]
    j += 1
    image = Image.open(os.path.join(current_dir,file))
    file_name = "%05d" % j + ".png"
    image.save(os.path.join(output_dir,file_name))
    print(os.path.join(output_dir,file_name),"Has benn saved")

import os
from PIL import Image
import tqdm

# Get the current directory
current_dir = os.getcwd()
image_dir = os.path.join(current_dir,"parking_lot/raw_frames")
# Get a list of all files in the current directory
files = os.listdir(image_dir)

# Filter out only the JPG image files
jpg_files = [file for file in files if file.endswith("")]

output_dir = os.path.join(image_dir,"../images")
os.makedirs(output_dir,exist_ok=True)
# Iterate over the JPG image files and convert them to PNG
for jpg_file in jpg_files:
    # Construct the full file paths
    jpg_path = os.path.join(image_dir, jpg_file)

    # num = int(jpg_file[:5].lstrip("0"))
    # num = num - 51
    # new_str = "%05d" % num

    # Get the last 5 characters of the input image name
    output_name = jpg_file[1:6] + ".png"

    # Change the extension to .png
    output_name = output_name

    # Construct the output file path
    png_path = os.path.join(output_dir, output_name)

    try:
        # Open the JPG image using PIL
        image = Image.open(jpg_path)

        # Convert the image to PNG format and save with the new name
        image.save(png_path, "PNG")

        # Close the image after processing
        image.close()

        # Remove the original JPG file if needed
        os.remove(jpg_path)

        print(f"Converted {jpg_path} to {png_path}")
    except IOError:
        print(f"Failed to convert {jpg_path}")

import os
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract images from a directory with a specified rate')
    parser.add_argument('--rate', type=int, default=1, help='Rate for extracting images')
    parser.add_argument('--data_dir', default='lab/original_process/original_images', help='Directory containing images')
    parser.add_argument('--output_dir', default='../../extract_images_200fps', help='Output directory for extracted images')

    args = parser.parse_args()

    rate = args.rate
    data_dir = args.data_dir
    output_dir = args.output_dir

    current_dir = os.getcwd()
    current_dir = os.path.join(current_dir, data_dir)

    files = os.listdir(current_dir)
    output_dir = os.path.join(current_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [file for file in files if file.endswith("png")]
    image_files = sorted(image_files)
    image_number = len(image_files)
    j = 0

    for i in range(0, image_number, rate):
        file = image_files[i]
        j += 1
        image = Image.open(os.path.join(current_dir, file))
        file_name = "%05d" % j + ".png"
        image.save(os.path.join(output_dir, file_name))
        print(os.path.join(output_dir, file_name), "Has been saved")

if __name__ == '__main__':
    main()

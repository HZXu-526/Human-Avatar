import os
from PIL import Image
import argparse

def convert_images(input_dir, output_dir):
    files = os.listdir(input_dir)
    jpg_files = [file for file in files if file.endswith('.jpg')]

    os.makedirs(output_dir, exist_ok=True)

    for jpg_file in jpg_files:
        jpg_path = os.path.join(input_dir, jpg_file)
        output_name = jpg_file[1:6] + ".png"
        png_path = os.path.join(output_dir, output_name)

        try:
            image = Image.open(jpg_path)
            image.save(png_path, "PNG")
            image.close()

            os.remove(jpg_path)

            print(f"Converted {jpg_path} to {png_path}")
        except IOError:
            print(f"Failed to convert {jpg_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert JPG images to PNG format.')
    parser.add_argument('input_dir', help='Input directory containing JPG images')
    parser.add_argument('output_dir', help='Output directory for PNG images')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    convert_images(input_dir, output_dir)

if __name__ == '__main__':
    main()

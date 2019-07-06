import os, argparse, cv2
from textract.extractor import TextExtractor


# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, help='test image folder')
parser.add_argument('--out_dir', type=str, help='ocr text output folder')
args = parser.parse_args()

# output folder
output_path = args.out_dir
if not os.path.exists(output_path):
    os.mkdir(output_path)

# load images
image_path = args.img_dir
image_folder = os.listdir(image_path)

# use textract
extractor = TextExtractor()

# ocr each image
for file in image_folder:

    # check available format
    if file.split('.')[-1] not in ['png','jpg','tiff']:
        continue

    # read image
    image = cv2.imread(os.path.join(image_path, file))

    # extract ocr text
    ocr_text = ''.join(extractor.extract(image))

    # write output txt file
    output_file = open(os.path.join(output_path, file+'.txt'), 'w')
    output_file.write(ocr_text)
    output_file.close()

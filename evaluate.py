'''
do the similarity evaluation
usage:
    > python evaluate.py --img_dir path/to/image/folder --gd_dir path/to/groundtruth/folder --out_dir path/to/output/folder
'''
import argparse, os, cv2
from similarity.normalized_levenshtein import NormalizedLevenshtein
from textract.extractor import TextExtractor

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, help='image folder')
parser.add_argument('--gd_dir', type=str, help='ground truth folder')
parser.add_argument('--out_dir', type=str, help='ocr text output folder')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

normalized_levenshtein = NormalizedLevenshtein()
similarities = {}

# use textract
extractor = TextExtractor()

folder = os.listdir(args.img_dir)
for article in folder:
    article_path = os.path.join(args.img_dir, article)
    if not os.path.isdir(article_path):
        continue

    groudtruth_path = os.path.join(args.gd_dir, article+'.txt')

    if not os.path.exists(groudtruth_path):
        continue

    print('Current Process: ' + article)

    pages = os.listdir(article_path)
    pages.sort()# sort by page number

    result_path = os.path.join(args.out_dir, article + '.txt')
    result_file = open(result_path, 'a', encoding='utf8')

    for page in pages:
        if page.split('.')[-1] not in ['png','jpg','tiff']:
            continue

        print(page)
        page_img = cv2.imread(os.path.join(article_path, page))
        text = ''.join(extractor.extract(page_img))
        result_file.write(text)
    result_file.close()

    # read file
    result_text = open(result_path, 'r').read().replace('\n','').replace(' ','')
    groudtruth_text = open(groudtruth_path, 'r').read().replace('\n','').replace(' ','')

    # calculate the similarity
    similarity = normalized_levenshtein.similarity(result_text, groudtruth_text)
    similarities[article] = similarity

    print('similarity of {} is {}'.format(article, similarity))

print(similarities)


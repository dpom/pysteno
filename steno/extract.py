import cv2
import argparse


def extract_words(image_path, outdir):
    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("{}/word_{:0>3d}.png".format(outdir, image_number), ROI)
        image_number += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image_path", help="the path of the input image")
    parser.add_argument("output_directory", help="the output directory path")
    args = parser.parse_args()
    extract_words(args.input_image_path, args.output_directory)


if __name__ == "__main__":
    main()

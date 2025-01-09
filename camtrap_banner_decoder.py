"""
camtrap_banner_decoder
Copyright 2024 Olivier Friard


Rename file using date and time extracted from the camera-trap bottom video/picture banner

Require: tesseract program (see https://github.com/tesseract-ocr/tesseract)

Usage:
python camtrap_banner_decoder.py.py -d INPUT_DIRECTORY
    show extracted information

python camtrap_banner_decoder.py.py -d INPUT_DIRECTORY --debug
    show more information

python camtrap_banner_decoder.py.py -d INPUT_DIRECTORY --rename
    show extracted information and rename file like YYYY-MM-DD_hhmmss_CAMTRAP-ID_OLD-FILE-NAME


For moon phase see:
https://pyorbital.readthedocs.io/en/feature-moon-phase/moon_calculations.html


  camtrap_banner_decoder is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  any later version.

  camtrap_banner_decoder is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not see <http://www.gnu.org/licenses/>.


"""

import argparse
import sys
import cv2
import pytesseract
import re
from pathlib import Path

__version__ = "0.0.1"

EXTENSIONS = {".avi", ".mp4", ".jpg", ".jpeg"}


def banner_text_from_frame(frame, roi_height_fraction: float = 0.15, debug=False, file_path=""):
    """
    extract text from frame banner
    """

    extracted_text = None
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    if debug:
        print(f"image original dimention: {frame_height}x{frame_width}")

    if frame_width > 2592:
        aspect_ratio = frame_height / frame_width
        new_width = 1280
        new_height = int(new_width * aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))
        frame_height, frame_width, _ = frame.shape

        if debug:
            print(f"image resized dimention: {frame_height}x{frame_width}")

    # Define the region of interest (ROI)
    roi_height = int(frame_height * roi_height_fraction)
    roi = frame[frame_height - roi_height : frame_height, 0:frame_width]

    # Save the first sampled frame for inspection

    if file_path:
        cv2.imwrite(Path(file_path).with_suffix(".jpeg"), roi)

    # Convert ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply OCR to extract text
    try:
        extracted_text = pytesseract.image_to_string(roi_gray, config="--psm 6")
    except Exception:
        print("Tesseract error")
        sys.exit()

    if debug:
        print(f"{extracted_text=}")

    return extracted_text


def extract_banner_text_from_video(video_path, frame_interval=30, roi_height_fraction=0.15, debug=False):
    """
    Extracts text from the bottom banner of a video.

    Args:
        video_path (str): Path to the video file.
        output_frame_path (str): Path to save a sample frame for inspection.
        frame_interval (int): Interval at which frames are sampled.
        roi_height_fraction (float): Fraction of the frame height that contains the banner.

    Returns:
        str: Extracted date and time from the video banner.
    """

    extracted_text = None

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process every `frame_interval` frame
        if frame_count % frame_interval == 0:
            extracted_text = banner_text_from_frame(frame, roi_height_fraction, debug)
            break

        frame_count += 1

    # Release video capture
    video_capture.release()

    return extracted_text


def extract_banner_text_from_image(image_path, roi_height_fraction=0.15, debug=False):
    """
    extract text contained in the bottom banner of an image
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return "Error: Unable to load the image. Check the file path."

    extracted_text = banner_text_from_frame(frame, roi_height_fraction, debug=debug, file_path=image_path)

    return extracted_text


def extract_date_time(path_file, debug=False):
    """
    extract info from the picture/video banner
    """

    banner_text = None

    if Path(path_file).suffix.lower() in (".avi", ".mp4"):
        banner_text = extract_banner_text_from_video(path_file, debug=debug)

    if Path(path_file).suffix.lower() in (".jpg", ".jpeg"):
        banner_text = extract_banner_text_from_image(path_file, debug=debug)

    if banner_text is None:
        return {"error": ""}

    for text in banner_text.split("\n"):
        # The input string
        # text = "@ FOSA_01 73F 23C @ 06-09-2023 13:41:51"

        flag_info = False

        # Extract date (MM-DD-YYYY)
        date_match = re.search(r"\d{2}-\d{2}-\d{4}", text)
        if date_match:
            raw_date = date_match.group(0)
            date = f"{raw_date.split('-')[2]}-{raw_date.split('-')[0]}-{raw_date.split('-')[1]}"
            flag_info = True
        else:
            continue

        # Extract time (HH:MM:SS)
        time_match = re.search(r"\d{2}:\d{2}:\d{2}", text)
        if time_match:
            raw_time = time_match.group(0)
            hhmmss = raw_time.replace(":", "")
            flag_info = True
        else:
            continue

        # Extract temperature in Fahrenheit (e.g., 73F)
        temperature_f = None
        temp_f_match = re.search(r" \d+F ", text)
        if temp_f_match:
            raw_temperature_f = temp_f_match.group(0)
            temperature_f = raw_temperature_f.strip()
            flag_info = True

        # Extract temperature in Celsius (e.g., 23C)
        temperature_c = None
        temp_c_match = re.search(r" \d+C ", text)
        if temp_c_match:
            raw_temperature_c = temp_c_match.group(0)
            temperature_c = raw_temperature_c.strip()
            flag_info = True

        if flag_info:
            # check for camera ID
            text2 = text
            if date:
                text2 = text2.replace(raw_date, "")
            if hhmmss:
                text2 = text2.replace(raw_time, "")
            if temperature_c:
                text2 = text2.replace(f"{temperature_c}", "")
            if temperature_f:
                text2 = text2.replace(f"{temperature_f}", "")

            while "  " in text2:
                text2 = text2.replace("  ", " ")

            if debug:
                print(f"{text=}")
                print(f"{text2=}")

            cam_id = None
            try:
                cam_id = sorted(text2.split(" "), key=len, reverse=True)[0]
            except Exception:
                pass

            if debug:
                print(f"{cam_id=}")

            return {
                "text": text,
                "cam_id": cam_id,
                "date": date,
                "time": hhmmss,
                "temperature_c": temperature_c,
                "temperature_f": temperature_f,
            }
        else:
            return {"error": ""}

    return {"error": ""}


def main():
    parser = argparse.ArgumentParser(description="Extract and rename picture and video files with date/time extracted from banner")

    parser.add_argument("-d", "--directory", action="store", dest="input_directory", default="", help="Directory with media files")
    parser.add_argument("--cam-id", action="store", dest="cam_id", default="", help="CAM_ID default")
    parser.add_argument("--rename", action="store_true", dest="rename", default="", help="Rename files")
    parser.add_argument("--debug", action="store_true", dest="debug", help="Enable debug mode")
    parser.add_argument("-v", "--version", action="store_true", dest="version", help="Display version")

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.version:
        print(f"camtrap_banner_decoder v. {__version__}\n")
        sys.exit()

    if Path(args.input_directory).is_dir():
        input_dir = args.input_directory
    else:
        print(f"Directory {args.input_directory} not found")
        sys.exit()

    if args.debug:
        print(f"{input_dir=}")

    files = sorted([file for file in Path(input_dir).glob("*") if file.suffix.lower() in EXTENSIONS])

    for file_path in files:
        if args.debug:
            print(f"{file_path=}")

        data = extract_date_time(str(file_path), debug=args.debug)
        if "error" in data:
            print(f"Date and time not found in {file_path}")
            print("-" * 30)
            continue

        if args.debug:
            print(f"{data['temperature_c']=}   {data['temperature_f']=}")

        if data["date"] and data["time"]:
            if data["cam_id"] is None:
                if args.cam_id:
                    data["cam_id"] = args.cam_id
                else:
                    data["cam_id"] = "CAM-ID"

            new_file_path = Path(file_path).parent / f"{data['date']}_{data['time']}_{data['cam_id']}_{file_path.name}"

            # check if file already renamed
            if str(Path(file_path).name).count("-") == 2:
                print(f"{Path(file_path).name} already renammed")
            else:
                if args.rename:
                    file_path.rename(new_file_path)
                    print(f"{Path(file_path).name} renamed to {Path(new_file_path).name}")
                else:
                    print(f"rename {Path(file_path).name} to {Path(new_file_path).name}")
        print("-" * 30)


if __name__ == "__main__":
    main()

# Accession Number Finder

A tool for extracting accession numbers from herbarium specimen images using a two-stage OCR approach.

## Overview

This project provides a Python-based solution for automatically extracting 6-7 digit accession numbers from herbarium specimen images. It uses a two-stage OCR process:

1. **EasyOCR** - Fast initial scan on a 3×3 grid of image tiles
2. **AWS Textract** - More comprehensive fallback scan when EasyOCR doesn't find a match

The tool downloads images from URLs provided in a text file, processes them to find accession numbers, and outputs the results to a CSV file.

## Features

- Downloads images from a list of URLs
- Extracts 6-7 digit accession numbers (no leading zeros)
- Uses a two-stage OCR approach for improved accuracy
- Configurable confidence threshold for EasyOCR
- Option to prefer longer numbers when multiple matches are found
- Results saved to CSV with automatic date stamping
- Progress is preserved as each image is processed

## Requirements

- Python 3.6+
- AWS account with Textract access
- AWS CLI configured with appropriate credentials
- Dependencies:
  - boto3
  - easyocr
  - opencv-python
  - numpy
  - requests

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/rherbst123/AccessionNumberFinder.git
   cd AccessionNumberFinder
   ```

2. Install required packages:
   ```
   pip install boto3 easyocr opencv-python numpy requests
   ```

3. Configure AWS credentials:
   ```
   aws configure
   ```


## Usage for OCR only system.

Basic usage:
```
python OCR_Extraction.py urls.txt [download_dir]
```

Where:
- `urls.txt` is a text file containing one image URL per line
- `download_dir` (optional) is the directory where images will be saved (defaults to "downloaded_images")

Example:
```
python OCR_Extraction.py specimen_urls.txt herbarium_images
```



## Usage for AWS system.

Basic usage:
```
python AWS_Extraction.py urls.txt [download_dir]
```

Where:
- `urls.txt` is a text file containing one image URL per line
- `download_dir` (optional) is the directory where images will be saved (defaults to "downloaded_images")

Example:
```
python AWS_Extraction.py specimen_urls.txt herbarium_images
```

## Output

The script generates a CSV file with the following columns:
- URL of the image
- Name of the downloaded image file
- Extracted accession number (or error message)

The CSV filename includes the input filename and current date (e.g., `specimen_urls_2023-06-15.csv`).

## Configuration

You can adjust the following parameters at the top of the script:

```python
MIN_LEN = 6             # Minimum length of accession numbers
MAX_LEN = 7             # Maximum length of accession numbers
EASY_CONF_THRESH = 0.80 # Confidence threshold for EasyOCR
PREFER_LONGEST = True   # Choose 7-digit over 6-digit when both found
```

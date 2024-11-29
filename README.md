# Day-evening-detector
## Video Day and Evening Analysis  

This project processes a video file that captures transitions from day to evening, analyzing the frames using **image processing** techniques in OpenCV. The program calculates and outputs the percentage of **day** and **evening** frames in the video, while providing intermediate visualizations to better understand the image processing workflow.

## Features
- Analyzes video frames to classify them as **day** or **evening** based on brightness thresholds.
- Calculates the percentage of **day** and **evening** in the video.
- Displays visualizations of:
  - **Original (BGR)** frames.
  - Intermediate steps such as HSV transformations and brightness analysis.
  - Combined views for easier interpretation.
- Outputs a final result summary showing the percentage of day and evening.


## Prerequisites
Ensure you have the following installed:
1. **Python 3.8+**
2. **OpenCV**
3. **numpy**
4. **seaborn**


   
## Project Workflow
1. **Input Video**
   - A video capturing transitions from day to evening is required.
   - The script reads the video file frame by frame using OpenCV.
2. **Frame Processing**
   - Each frame is analyzed to classify it as day or evening:
     - Day frames: High brightness and specific HSV characteristics.
     - Evening frames: Lower brightness levels and HSV shifts typical of dusk.
3. **Visualization**
   - The program displays intermediate steps, including:
     - Original (BGR) frame.
     - HSV-transformed frame.
     - Brightness and histogram analysis.
4. **Output**
   - The program calculates and outputs the percentage of day and evening in the video.

## Code Overview
- ### **Main Logic**
   - **Input:** Video file path.
   - **Processing:**
     - Frame-by-frame analysis using brightness and HSV thresholds.
     - Intermediate visualizations for better understanding.
   - **Output:**
     - Final percentages of day and evening frames.
     - Visual display of key processing steps.
- ### **Key Functions**
  - **Frame Classification:**
     - classify_frame(frame): Determines whether a frame is day or evening based on brightness and color thresholds.
  - **Visualization:**
     - Side-by-side and stacked visualizations of original and processed frames.
**Analysis:**
     - Brightness histograms and frame intensity graphs.

## Example Output
 **During Execution**
- **HSV Frame Conversion**: Example of a video frame converted to HSV color space.
![HSV Frame](interleved_color_intensity_frames.png)
- **Color intensite histogram for BGR and HSV**: representing color intensity histogram for both BGR and HSV plotted to understand the intensity level visually.
![Brightness Heatmap](ColorIntensityHistogram.png)
- **Brightness and frames**: Three different maps(line graph, colored line graph and a heatmap)  to representing brightness levels across frames, giving a visual summary of brightness variations throughout the video.
![Brightness Heatmap](BrightnessVsFrameNo.png)
- **Brightness Heatmap**: Three different maps(line graph, colored line graph and a heatmap)  to representing brightness levels across frames, giving a visual summary of brightness variations throughout the video.
![Brightness Heatmap](different_plots_represnting_brightnessvsFrameno.png)

## Final Output
At the end of processing:

 > Day Frames: 54.44%
   
 > Evening Frames: 45.56%

## Usage Scenarios
This project is ideal for:
   > Learning image processing techniques.
   
   > Understanding how brightness and color spaces (BGR, HSV) can be used for scene classification.
   
   > Automating video analysis tasks based on environmental conditions.

   

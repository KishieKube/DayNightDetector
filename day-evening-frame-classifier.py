import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_combined_histograms(ax1, ax2, frame, hsv_frame):
    """
    Plot BGR and HSV histograms together in a single figure using subplots.
    The histograms will update in the same window.
    """
    # Clear previous plots
    ax1.clear()
    ax2.clear()

    # Plot BGR histogram
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        ax1.plot(hist, color=col)
    ax1.set_title("Color Intensity Histogram (BGR)")
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # Plot HSV histogram
    channels = ('Hue', 'Saturation', 'Value')
    colors_hsv = ('m', 'c', 'y')  # Magenta, Cyan, Yellow
    for i, col in enumerate(colors_hsv):
        hist = cv2.calcHist([hsv_frame], [i], None, [256], [0, 256])
        ax2.plot(hist, color=col, label=channels[i])
    ax2.set_title("Color Intensity Histogram (HSV)")
    ax2.set_xlabel("Pixel Intensity")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    # Pause briefly to allow the plot to update
    plt.pause(0.001)

def classify_frame(frame, hsv_frame, ax1, ax2, brightness_values):
    """
    Classify a frame as 'day' or 'evening', plot histograms, and calculate average brightness.
    """
    # Plot histograms
    plot_combined_histograms(ax1, ax2, frame, hsv_frame)

    # Calculate the average brightness (Value channel in HSV)
    avg_brightness = np.mean(hsv_frame[:, :, 2])
    brightness_values.append(avg_brightness)

    # Classify based on brightness threshold
    if avg_brightness > 100:
        return "day"
    else:
        return "evening"

def plot_brightness_graphs(brightness_values, total_frames):
    """
    Plot various brightness visualization graphs:
    1. Line plot of brightness over frames.
    2. Color-coded scatter plot based on brightness intensity.
    3. Heatmap of brightness values.

    :param brightness_values: List of average brightness values.
    :param total_frames: Total number of frames.
    """
    frame_numbers = range(1, total_frames + 1)

    # Create a figure for multiple subplots
    plt.figure(figsize=(15, 8))

    # Line plot
    plt.subplot(3, 1, 1)
    plt.plot(frame_numbers, brightness_values, label="Brightness", color="blue", linewidth=1.5)
    plt.title("Line Plot: Brightness vs Frame Number")
    plt.xlabel("Frame Number")
    plt.ylabel("Average Brightness")
    plt.legend()

    # Color-coded scatter plot
    plt.subplot(3, 1, 2)
    scatter = plt.scatter(frame_numbers, brightness_values, c=brightness_values, cmap="viridis", s=10)
    plt.colorbar(scatter, label="Brightness Intensity")
    plt.title("Color-Coded Scatter Plot")
    plt.xlabel("Frame Number")
    plt.ylabel("Average Brightness")

    # Heatmap
    plt.subplot(3, 1, 3)
    heatmap_data = np.expand_dims(brightness_values, axis=0)  # Convert brightness values to 2D
    sns.heatmap(heatmap_data, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
    plt.title("Heatmap Representation of Brightness")
    plt.xlabel("Frame Number")

    # Display all graphs
    plt.tight_layout()
    plt.show()


def display_interleaved_hsv_frames(frames, interleave_factor, grid_cols=5, frame_width=160, frame_height=90):
    """
    Display interleaved HSV frames in a single OpenCV window with a grid layout.

    :param frames: List of frames to display.
    :param interleave_factor: Interval for selecting frames (e.g., every 10th frame).
    :param grid_cols: Number of columns in the grid.
    :param frame_width: Width of each frame in the grid.
    :param frame_height: Height of each frame in the grid.
    """
    interleaved_frames = frames[::interleave_factor]
    resized_frames = [cv2.resize(f, (frame_width, frame_height)) for f in interleaved_frames]

    grid_rows = (len(resized_frames) + grid_cols - 1) // grid_cols
    grid_height = frame_height * grid_rows
    grid_width = frame_width * grid_cols

    # Create a blank canvas to hold the grid
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(resized_frames):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width

        grid_image[y_start:y_end, x_start:x_end] = frame

    # Display the grid in a single window
    cv2.imshow('Interleaved HSV Frames Grid', grid_image)
    cv2.waitKey(0)  # Wait for user input to close the window
    cv2.destroyAllWindows()


def analyze_video(video_path, interleave_factor=10):
    """
    Analyze the video and calculate the percentage of 'day' and 'evening'.
    Also plot histograms and brightness graphs for further insights.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set up Matplotlib figure and axes for histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    total_frames = 0
    day_frames = 0
    evening_frames = 0
    brightness_values = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Convert the frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frames.append(hsv_frame)
        # Classify the current frame
        classification = classify_frame(frame, hsv_frame, ax1, ax2, brightness_values)

        if classification == "day":
            day_frames += 1
        else:
            evening_frames += 1

        # Display HSV video frame
        cv2.imshow("HSV Video Frame", hsv_frame)

        # Exit when 'q' is pressed or program finishes all frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # Calculate percentages
    day_percentage = (day_frames / total_frames) * 100
    evening_percentage = (evening_frames / total_frames) * 100

    # Print results
    print(f"Day Percentage: {day_percentage:.2f}%")
    print(f"Evening Percentage: {evening_percentage:.2f}%")

    ### Plot brightness graph
    #plt.figure(figsize=(10, 6))
    #plt.title("Brightness vs Frame Number")
    #plt.xlabel("Frame Number")
    #plt.ylabel("Average Brightness")
    #plt.plot(range(1, total_frames + 1), brightness_values, label="Brightness")
    #plt.legend()
    #plt.show()
    # Plot brightness graphs
    plot_brightness_graphs(brightness_values, total_frames)

    # Display interleaved HSV frames
    display_interleaved_hsv_frames(frames, interleave_factor)
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    #return day_percentage, evening_percentage

# Input video file (provide the correct path)
video_path = "inputfile/2.mp4"

# Run the analysis
analyze_video(video_path, interleave_factor=10)
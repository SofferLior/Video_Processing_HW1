"""Basic Video Processing methods."""
import os
import cv2

# Replace ID1 and ID2 with your IDs.
ID1 = '203135058'
ID2 = '203764170'

INPUT_VIDEO = 'atrium.avi'
GRAYSCALE_VIDEO = f'{ID1}_{ID2}_atrium_grayscale.avi'
BLACK_AND_WHITE_VIDEO = f'{ID1}_{ID2}_atrium_black_and_white.avi'
SOBEL_VIDEO = f'{ID1}_{ID2}_atrium_sobel.avi'


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object. The input video's VideoCapture.
    Returns:
        parameters: dict. A dictionary of parameters names to their values.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    parameters = {"fourcc": fourcc, "fps": fps, "height": height, "width": width}
    return parameters


def convert_video_to_grayscale(input_video_path: str,
                               output_video_path: str) -> None:
    """Convert the video in the input path to grayscale.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, convert it to gray scale,
    and save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """

    # read video
    capture = cv2.VideoCapture(input_video_path)

    # print get output - might need to do something else here
    capture_info = get_video_parameters(capture)

    # open a new video file
    new_video = cv2.VideoWriter(output_video_path, capture_info['fourcc'], capture_info['fps'], (capture_info['width'], capture_info['height']), 0)

    # read video frames and convert to grayscale
    success, image = capture.read()
    count = 0
    while success:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_video.write(grayscale_image)
        success, image = capture.read()
        count += 1

    # release the video and close
    capture.release()
    new_video.release()
    cv2.destroyAllWindows()


def convert_video_to_black_and_white(input_video_path: str,
                                     output_video_path: str) -> None:
    """Convert the video in the input path to black and white.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, first convert it to gray scale,
    then use OpenCV's THRESH_OTSU to slice the gray color values to
    black (0) and white (1) and finally convert the frame format back to RGB.
    Save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    # read video
    capture = cv2.VideoCapture(input_video_path)

    # print get output - might need to do something else here
    capture_info = {"framecount": capture.get(cv2.CAP_PROP_FRAME_COUNT),
                    "fps": capture.get(cv2.CAP_PROP_FPS),
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "codec": int(capture.get(cv2.CAP_PROP_FOURCC))}

    # open a new video file
    new_video = cv2.VideoWriter(output_video_path, capture_info['codec'], capture_info['fps'], (capture_info['width'], capture_info['height']))

    # read video frames and convert to black and white
    success, image = capture.read()
    count = 0
    while success:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        new_rgb_frame = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB)
        new_video.write(new_rgb_frame)
        success, image = capture.read()
        count += 1

    # release the video and close
    capture.release()
    new_video.release()
    cv2.destroyAllWindows()


def convert_video_to_sobel(input_video_path: str,
                           output_video_path: str) -> None:
    """Convert the video in the input path to sobel map.

    Use VideoCapture from OpenCV to open the video and read its
    parameters using the capture's get method.
    Open an output video using OpenCV's VideoWriter.
    Iterate over the frames. For each frame, first convert it to gray scale,
    then use OpenCV's THRESH_OTSU to slice the gray color values to
    black (0) and white (1) and finally convert the frame format back to RGB.
    Save the frame to the new video.
    Make sure to close all relevant captures and to destroy all windows.

    Args:
        input_video_path: str. Path to input video.
        output_video_path: str. Path to output video.

    Additional References:
    (1) What are fourcc parameters:
    https://docs.microsoft.com/en-us/windows/win32/medfound/video-fourccs

    """
    # read video
    capture = cv2.VideoCapture(input_video_path)

    # print get output - might need to do something else here
    capture_info = {"framecount": capture.get(cv2.CAP_PROP_FRAME_COUNT),
                    "fps": capture.get(cv2.CAP_PROP_FPS),
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "codec": int(capture.get(cv2.CAP_PROP_FOURCC))}

    # open a new video file
    new_video = cv2.VideoWriter(output_video_path, capture_info['codec'], capture_info['fps'], (capture_info['width'], capture_info['height']))

    # read video frames and apply sobel filter
    success, image = capture.read()
    count = 0
    while success:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_image = cv2.Sobel(grayscale_image, ddepth=-1, dx=1, dy=1, ksize=5)
        new_rgb_frame = cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2RGB)
        new_video.write(new_rgb_frame)
        success, image = capture.read()
        count += 1

    # release the video and close
    capture.release()
    new_video.release()
    cv2.destroyAllWindows()


def main():
    convert_video_to_grayscale(INPUT_VIDEO, GRAYSCALE_VIDEO)
    convert_video_to_black_and_white(INPUT_VIDEO, BLACK_AND_WHITE_VIDEO)
    convert_video_to_sobel(INPUT_VIDEO, SOBEL_VIDEO)


if __name__ == "__main__":
    main()

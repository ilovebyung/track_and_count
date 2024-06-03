import cv2

def extract_first_frame(video_path, output_path):
  """
  Extracts the first frame of an MP4 video and saves it as an image.

  Args:
    video_path: Path to the MP4 video file.
    output_path: Path to save the extracted image.
  """
  cap = cv2.VideoCapture(video_path)
  ret, frame = cap.read()
  if ret:
    cv2.imwrite(output_path, frame)
  cap.release()

# Example usage
video_path = "HIGH_RES.mp4"
output_path = "output/frame.jpg"
extract_first_frame(video_path, output_path)



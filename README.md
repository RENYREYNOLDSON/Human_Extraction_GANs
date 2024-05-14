## Human Patch Extraction
This is a semantic segmentation problem. We use the
pre-trained Mask-RCNN model “COCO” to detect
humans in the input frames. The dataset has 91 distinct
object classes; however, we only extracted objects
belonging to the “person” class. We iterate through
each video frame using cv2, only applying object
detection on every 24th frame (~every second), if a
person is detected with a confidence score above the
threshold value, then 4 frames spread across the
previous second are also run through the detector.
Doing this makes the patch extraction lightweight as we
don’t need all frames. We made the patches square to
ensure consistent sizes throughout the project. We
gathered 48,900 image patches in total.

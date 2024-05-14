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

## Patch Classification
We used the pytorch
keypointrcnn_resnet50_fpn model (mask-RCNN) to
estimate the joint positions / key points. The 4 classes
given are:
FULL-BODY-FRONT-VIEW
HEAD-AND-SHOULDER-FRONT-VIEW
FULL-BODY-BACK-VIEW
HEAD-AND-SHOULDER-BACK-VIEW
For each output tensor we used the person with the
highest score (as some images contain multiple people).
In some cases, this results in the wrong person being
used, we tested with taking the person with the largest
average size instead, but this didn’t work as well. There
are 17 output classes, and we use these to classify each
image based on the confidence of features. For
example, high confidence in facial features shows us
that it is a front view. This method worked very well for
classification and it’s an intuitive design.

## Pair Selection
To improve human style transfer training, we require
paired images. The proposed method of finding such
pairs is to do the cosine similarity between movie-game
estimated key point positions. 
This gives us images that contain
humans with similar poses, which is very useful and
ignores irrelevant background information that would
come from doing cosine similarity on the image arrays
directly. The algorithm was altered to save all
estimated feature positions. Pairs were then saved into
CSVs for reference in style transfer. Note that pairs also had
to belong to the same classification group. Overall,
this method performed incredibly well on pairing the
images.










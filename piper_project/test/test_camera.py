import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    15
)

pipeline.start(config)

try:
    while True:

        frames = pipeline.wait_for_frames(10000)

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        cv2.imshow("RGB", color_image)

        key = cv2.waitKey(1)

        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

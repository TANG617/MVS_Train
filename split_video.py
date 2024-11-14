import cv2
import numpy as np
from PIL.ImageChops import overlay


def split_and_concatenate_frames(video_path, output_image_path,  points_src, points_dst, frame_height=240, frame_width=1080, gap = 60, overlay = 30, index = 0):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    matrix = cv2.getPerspectiveTransform(np.float32(points_src), np.float32(points_dst))

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return



    # 获取视频的帧宽度和高度
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 每一帧分割的行数
    rows = video_height // frame_height
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完成

        # 分割每一帧
        # for i in range(rows):
        #     start_y = i * frame_height
        #     end_y = (i + 1) * frame_height
            # 截取每一帧中的对应区域
        start_y = index * gap - overlay
        start_y = max(start_y, 0)
        end_y = (index + 1) * gap +  overlay
        end_y = min(end_y, 1080)

        corrected_frame = cv2.warpPerspective(frame, matrix, (1920, 1080))
        print(corrected_frame.shape)

        cropped_frame = corrected_frame[start_y:end_y, :]
        frames.append(cropped_frame)
        cv2.imwrite( "/Users/timli/DeepLearningProjects/MVS_Train/split/_" + str(index) + "_" + output_image_path, cropped_frame)

    # 拼接图像
    # concatenated_image = np.vstack(frames)  # 按行拼接

    # 保存拼接后的图像
    # cv2.imwrite(output_image_path, concatenated_image)

    cap.release()
    print(f"Image saved as {output_image_path}")


# 示例使用
video_path = '01.mp4'  # 输入视频路径
output_image_path = '.png'  # 输出图片路径
# points_src = [(283, 0), (1548, 0), (1707, 1080), (128, 1080)]  # 这四个点需要你根据视频内容手动选择
points_src = [(283-40, 0), (1548+40, 0), (1707+40, 1080), (128-40, 1080)]  # 这四个点需要你根据视频内容手动选择
# 目标点：矫正后的目标矩形区域
frame_width = 1920
frame_height = 1080
points_dst = [(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)]
gap = 180
overlay_gap = 20
for i in range(int(1080/gap)):
    split_and_concatenate_frames(video_path=video_path, output_image_path="i_" + str(i) +"_r_" + str(gap) + "_"+output_image_path, points_src=points_src, points_dst=points_dst , index = i, gap = gap, overlay = overlay_gap)
# for i in range(8):
#     split_and_concatenate_frames(video_path=video_path, output_image_path="range_"+str((i+1)*10)+output_image_path, points_src=points_src, points_dst=points_dst, gap = (i+1)*10 )
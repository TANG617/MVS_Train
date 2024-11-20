import cv2
import numpy as np
import os


def create_panorama(image_paths):
    # 读取图像
    images = [cv2.imread(image_path) for image_path in image_paths]

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()

    # 计算每张图片的特征和描述符
    keypoints_list = []
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    # 使用暴力匹配器（Brute Force Matcher）匹配特征点
    bf = cv2.BFMatcher()

    # 存储所有匹配结果
    matches = []
    for i in range(len(images) - 1):
        matches.append(bf.knnMatch(descriptors_list[i], descriptors_list[i + 1], k=2))

    # 进行特征点的比值检验（Lowe's ratio test）
    good_matches = []
    for match in matches:
        good = []
        for m, n in match:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        good_matches.append(good)

    # 初始化目标拼接图像
    result = images[0]
    result_offset_y = 0  # 用于纵向拼接时计算图像放置的纵向偏移量

    for i in range(1, len(images)):
        if len(good_matches[i - 1]) >= 3:  # 确保至少有3个匹配点（仿射变换需要3个点）
            src_pts = np.float32([keypoints_list[i - 1][m.queryIdx].pt for m in good_matches[i - 1]]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_list[i][m.trainIdx].pt for m in good_matches[i - 1]]).reshape(-1, 1, 2)

            # 计算仿射变换矩阵
            M = cv2.estimateAffine2D(src_pts, dst_pts)[0]  # 返回仿射变换矩阵

            # 确保矩阵有效
            if M is None:
                print("Affine matrix is invalid, skipping this match.")
                continue  # 如果仿射矩阵无效，则跳过这一对图像

            # 获取拼接后的结果图像的宽度和高度
            result_width = max(result.shape[1], images[i].shape[1])  # 保证拼接图像的宽度
            result_height = result.shape[0] + images[i].shape[0]  # 高度逐步增加

            # 创建新的画布（目标拼接图像）
            result_resized = np.zeros((result_height, result_width, 3), dtype=np.uint8)

            # 将当前拼接图像复制到目标画布中
            result_resized[0:result.shape[0], 0:result.shape[1]] = result

            # 将当前图像通过仿射变换加入到结果画布的上方
            result_resized[result_offset_y:result_offset_y + images[i].shape[0], 0:images[i].shape[1]] = images[i]

            result = result_resized  # 更新结果图像
            result_offset_y += images[i].shape[0]  # 更新纵向偏移量

            # 在重叠区域进行融合或裁剪处理
            for y in range(result_offset_y):
                for x in range(result.shape[1]):
                    if np.all(result[y, x] != 0):  # 检查图像是否有内容
                        # 使用加权叠加的方式，确保每个像素的RGB通道正确处理
                        weighted = cv2.addWeighted(result[y, x].astype(np.float32), 0.5,
                                                   images[i][y % images[i].shape[0], x].astype(np.float32), 0.5, 0)[0]

                        # 将加权叠加后的像素值限制在 0 到 255 之间，并转回 uint8 类型
                        result[y, x] = np.clip(weighted, 0, 255).astype(np.uint8)

        else:
            print(f"Not enough good matches for images {i - 1} and {i}. Skipping.")

    return result


def get_image_paths(folder_path, extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]):
    # 获取文件夹中的所有文件
    image_paths = []

    # 遍历文件夹
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否是图片（通过扩展名判断）
        if os.path.isfile(file_path) and any(file_path.endswith(ext) for ext in extensions):
            image_paths.append(file_path)

    return image_paths


# 示例使用
folder_path = "split"  # 图片文件夹路径
image_paths = get_image_paths(folder_path)
image_paths.sort()

print(image_paths)  # 打印所有图片路径
panorama = create_panorama(image_paths)

# 保存结果
cv2.imwrite('panorama.jpg', panorama)
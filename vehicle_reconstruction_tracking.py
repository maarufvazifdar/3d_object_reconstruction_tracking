import os
import cv2
import numpy as np
from pixellib.semantic import semantic_segmentation
import matplotlib.pyplot as plt
import open3d as o3d


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


dir1 = '/home/maaruf/cmsc733/project/kitti_selected/image_2/'
dir2 = '/home/maaruf/cmsc733/project/kitti_selected/image_3/'

files1 = os.listdir(dir1)
files1 = sorted(files1, key=lambda x: int(os.path.splitext(x)[0]))

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model(
    "/home/maaruf/cmsc733/project/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

cam_K = np.array([[721.5377, 0, 609.5593],
                  [0, 721.5377, 172.854],
                  [0, 0, 1]])

Tmat = np.array([0.54, 0., 0.])

rev_proj_matrix = np.zeros((4, 4))

for i, file in enumerate(files1[9:]):
    print(i, file)

    l_img = cv2.imread(dir1 + file)
    r_img = cv2.imread(dir2 + file)
    l_g = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    r_g = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

    segvalues, output = segment_image.segmentAsPascalvoc(dir1 + file)
    segvalues2, output2 = segment_image.segmentAsPascalvoc(dir2 + file)

    output_g = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output_g[output_g != 0] = 255
    kernel_1 = np.ones((11, 11), dtype='uint8')
    output_g = cv2.erode(output_g, kernel_1, iterations=1)
    op = cv2.bitwise_and(l_g, l_g, mask=output_g)

    output_g2 = cv2.cvtColor(output2, cv2.COLOR_BGR2GRAY)
    output_g2[output_g2 != 0] = 255
    output_g2 = cv2.erode(output_g2, kernel_1, iterations=1)
    op2 = cv2.bitwise_and(r_g, r_g, mask=output_g2)

    disparity = stereo.compute(op, op2)

    # Plots
    fig1, ax1 = plt.subplots(3, 2)
    plt.gcf().set_facecolor('white')
    ax1[0, 0].imshow(cv2.cvtColor(l_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    ax1[0, 0].set_title("Left Image")
    ax1[0, 0].axis('off')

    ax1[0, 1].imshow(cv2.cvtColor(r_img.astype('uint8'), cv2.COLOR_BGR2RGB))
    ax1[0, 1].set_title("Right Image")
    ax1[0, 1].axis('off')

    ax1[1, 0].imshow(op, cmap='gray')
    ax1[1, 0].set_title("Segmented Left Image")
    ax1[1, 0].axis('off')

    ax1[1, 1].imshow(op2, cmap='gray')
    ax1[1, 1].set_title("Segmented Right Image")
    ax1[1, 1].axis('off')

    ax1[2, 0].imshow(disparity, cmap='CMRmap')
    ax1[2, 0].set_title("Disparity")
    ax1[2, 0].axis('off')
    ax1[2, 1].axis('off')
    plt.tight_layout()
    plt.show()

    img = disparity.copy()

    cv2.stereoRectify(cameraMatrix1=cam_K, cameraMatrix2=cam_K,
                      distCoeffs1=0, distCoeffs2=0,
                      imageSize=l_g.shape,
                      R=np.identity(3), T=Tmat,
                      R1=None, R2=None,
                      P1=None, P2=None, Q=rev_proj_matrix)

    points = cv2.reprojectImageTo3D(img, rev_proj_matrix)

    # reflect on x axis
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.matmul(points, reflect_matrix)

    # extract colors from image
    colors = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)

    # filter by min disparity
    mask = img > img.min()
    out_points = points[mask]
    out_colors = colors[mask]

    # filter by dimension
    idx = np.fabs(out_points[:, 0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    ply_write = f'/home/maaruf/cmsc733/project/ply_files/{i}.ply'
    write_ply(ply_write, out_points, out_colors)

    pcd = o3d.io.read_point_cloud(
        f'/home/maaruf/cmsc733/project/ply_files/{i}.ply')
    o3d.visualization.draw_geometries([pcd])

    cl, ind = pcd.remove_radius_outlier(nb_points=300, radius=0.04)

    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud])

    aabb = inlier_cloud.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    print('\n\nEstimated Vehicle BoundingBox Pose:\n', aabb, '\n\n')
    o3d.visualization.draw_geometries([inlier_cloud, aabb])

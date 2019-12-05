import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import matplotlib.image as img
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import os


def isFloat(str_val):
    try:
        float(str_val)
        return True
    except ValueError:
        return False


# read sensor data file: radar,lidar
def read_file(filename, filetype):
    if filetype == 'radar':
        sensor_data_dir = radar_data_dir
    else:
        sensor_data_dir = lidar_data_dir

    p = []
    with open(sensor_data_dir + filename) as f:
        for line in f:
            line = line.rstrip()
            if line:
                line_str = line.split()
                if isFloat(line_str[0]):
                    line_float = [float(x) for x in line_str]
                    p.append(line_float)
    pa = np.array(p)
    return pa


def plot_3D_animation():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 45)
    frames = ax.scatter([], [], [], c='darkblue', alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-20, 20)

    def update(n):
        p = pcl_set[n]
        print(p.shape)
        frames._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        title.set_text('3D Visualization, time={}'.format(n))
        return frames

    ani = animation.FuncAnimation(fig, update, len(pcl_set), interval=2000, repeat=False)
    plt.show()
    return 0


# get the object bounding boxes list
# def get_objects(n):
#     read ground truth file
#     center
#     dimension
#     orientation
# convert orientation quaternion to rotation matrix
# get the eight points which describes a annotation bounding box


# transformation matrix from sensorA->sensorB to sensorB->sensorA
def invTrans(T):
    rotation = np.linalg.inv(T[0:3, 0:3])  # rotation matrix

    translation = T[0:3, 3]
    translation = -1 * np.dot(rotation, translation.T)
    translation = np.reshape(translation, (3, 1))
    Q = np.hstack((rotation, translation))

    # test if it is truly a roation matrix
    d = np.linalg.det(rotation)
    t = np.transpose(rotation)
    o = np.dot(rotation, t)
    return Q


def quat_to_rotation(quat):
    m = np.sum(np.multiply(quat, quat))
    q = quat.copy()
    q = np.array(q)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        rot_matrix = np.identity(4)
        return rot_matrix
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
        dtype=q.dtype)
    d = np.linalg.det(rot_matrix)
    t = np.transpose(rot_matrix)
    o = np.dot(rot_matrix, t)
    return rot_matrix


def plot_box_on_pcl(ax, points, classId):
    # assume the shape of points:(8,3)
    w = points[2, 0] - points[3, 0]
    h = points[0, 1] - points[3, 1]
    rect = patches.Rectangle((points[3, 0], points[3, 1]), width=w, height=h, linewidth=1, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    return 0


def plot_box_on_image(ax, points, classId):
    # assume the shape of points:(8,3)
    a = np.zeros((2, 5))
    a[0, 0:4] = points[0, 0:4]
    a[0, 4] = points[0, 0]
    a[1, 0:4] = points[1, 0:4]
    a[1, 4] = points[1, 0]
    ax.plot(a[0, :], a[1, :], color='darkblue')

    b = np.zeros((2, 5))
    b[0, 0:4] = points[0, 4:]
    b[0, 4] = points[0, 7]
    b[1, 0:4] = points[1, 4:]
    b[1, 4] = points[1, 4]
    ax.plot(b[0, :], b[1, :], color='darkblue')

    c = np.zeros((2, 2))
    c[0, 0] = points[0, 0]
    c[0, 1] = points[0, 4]
    c[1, 0] = points[1, 0]
    c[1, 1] = points[1, 4]
    ax.plot(c[0, :], c[1, :], color='darkblue')

    c[0, 0] = points[0, 1]
    c[0, 1] = points[0, 5]
    c[1, 0] = points[1, 1]
    c[1, 1] = points[1, 5]
    ax.plot(c[0, :], c[1, :], color='darkblue')

    c[0, 0] = points[0, 2]
    c[0, 1] = points[0, 6]
    c[1, 0] = points[1, 2]
    c[1, 1] = points[1, 6]
    ax.plot(c[0, :], c[1, :], color='darkblue')

    c[0, 0] = points[0, 3]
    c[0, 1] = points[0, 7]
    c[1, 0] = points[1, 3]
    c[1, 1] = points[1, 7]
    ax.plot(c[0, :], c[1, :], color='darkblue')

    # ax.txt(a[0,0], a[1,0], classId)

    return 0


def plot_2D_annotation(n):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()

    # plot radar pcl on x-y dimension
    ax = fig.add_subplot(gs[0, 0])
    p1 = radar_pcl_set[n]
    # p1 = np.flip(p1[:, 0:2], 1)
    frames_radar = ax.scatter(p1[:, 0], p1[:, 1], c='darkblue', s=1, alpha=0.5)
    title1 = ax.set_title('Radar Point Cloud 2D Visualization at time={}'.format(n))
    ax.set_xlim(0, 100)  # (-100, 100)
    ax.set_ylim(-100, 100)  # (0, 100)

    # objects = get_objects(n)
    groundtruth_path = groundtruth_data_dir + str(n).zfill(6) + '.json'
    with open(groundtruth_path, mode='r') as file:
        data = json.load(file)
    objects_info = data['objects']
    objects = []
    classids = []
    bbox = np.zeros((8, 3))
    for p in objects_info:
        center = np.array(p['center3d'])
        dimension = np.array(p['dimension3d'])
        orientation = np.array(p['orientation_quat'])
        classids.append(p['classname'])
        bbox[0, :] = np.array([center[0] - dimension[0] / 2, center[1] + dimension[1], center[2] + dimension[2]])
        bbox[1, :] = np.array([center[0] + dimension[0] / 2, center[1] + dimension[1], center[2] + dimension[2]])
        bbox[2, :] = np.array([center[0] + dimension[0] / 2, center[1] - dimension[1], center[2] + dimension[2]])
        bbox[3, :] = np.array([center[0] - dimension[0] / 2, center[1] - dimension[1], center[2] + dimension[2]])
        bbox[4, :] = np.array([center[0] - dimension[0] / 2, center[1] + dimension[1], center[2] - dimension[2]])
        bbox[5, :] = np.array([center[0] + dimension[0] / 2, center[1] + dimension[1], center[2] - dimension[2]])
        bbox[6, :] = np.array([center[0] + dimension[0] / 2, center[1] - dimension[1], center[2] - dimension[2]])
        bbox[7, :] = np.array([center[0] - dimension[0] / 2, center[1] - dimension[1], center[2] - dimension[2]])
        orientation_matrix = quat_to_rotation(orientation)
        bbox = np.dot(orientation_matrix, np.transpose(bbox))
        bbox = np.transpose(bbox)
        objects.append(bbox)
    print(len(objects))

    # add annotations on radar pcl plot
    for obj, id in zip(objects, classids):
        plot_box_on_pcl(ax, obj, id)
    print('radar plot finished!')

    # read calibration file for tranformations of annotations
    calib_path = calib_dir + str(n).zfill(6) + '.json'
    with open(calib_path, mode='r') as file:
        data = json.load(file)
    T_fromLidar = np.array(data['sensors'][1]['calib_data']['T_to_ref_COS'])
    T_fromCamera = np.array(data['sensors'][2]['calib_data']['T_to_ref_COS'])
    K = np.array(data['sensors'][2]['calib_data']['K'])

    T_toLidar = invTrans(T_fromLidar)
    T_toCamera = invTrans(T_fromCamera)

    # plot lidar pcl on x-y dimension
    ax = fig.add_subplot(gs[0, 1])
    p2 = lidar_pcl_set[n]
    # p2 = np.flip(p2[:, 0:2], 1)
    frames_lidar = ax.scatter(p2[:, 0], p2[:, 1], c='darkblue', s=1, alpha=0.5)
    title2 = ax.set_title('Lidar Point Cloud 2D Visualization at time={}'.format(n))
    ax.set_xlim(-10, 100)
    ax.set_ylim(-50, 50)

    # add annotations on lidar pcl plot
    for obj, id in zip(objects, classids):
        obj = np.dot(T_toLidar[0:3, 0:3], np.transpose(obj))
        T = T_toLidar[0:3, 3]
        obj = obj + T[:, np.newaxis]
        obj = np.transpose(obj)
        # obj[:,[0, 1]] = obj[:,[1, 0]]
        plot_box_on_pcl(ax, obj, id)
    print('lidar plot finished!')

    # plot camera image
    ax = fig.add_subplot(gs[1, :])
    image_path = camera_data_dir + str(n).zfill(6) + '.jpg'
    print(image_path)
    image = img.imread(image_path)
    # frames_camera = ax.imshow(image)
    title3 = ax.set_title('Camera Image at time={}'.format(n))

    # add annotations on camera image
    for obj2, id2 in zip(objects, classids):
        tmp2 = np.dot(T_toCamera[0:3, 0:3], np.transpose(obj2))
        T = T_toCamera[0:3, 3]
        tmp2 = tmp2 + T[:, np.newaxis]
        tmp2 = np.dot(K, tmp2)
        tmp2 = np.delete(tmp2, 2, 0)
        plot_box_on_image(ax, tmp2, id2)

    plt.show()
    return 0


root_dir = os.environ['AOD_HOME']
groundtruth_data_dir = root_dir + 'groundtruth_obj3d/'
calib_dir = root_dir + 'calibration/'
radar_data_dir = root_dir + 'radar_6455/'
lidar_data_dir = root_dir + 'lidar_vlp16/'
camera_data_dir = root_dir + 'camera_front/'

files = os.listdir(radar_data_dir)
files.sort()
n = 0
files = files[n:n + 5]
radar_pcl_set = []
lidar_pcl_set = []

for file in files:
    if not os.path.isdir(file):
        radar_pcl = read_file(file, 'radar')
        radar_pcl_set.append(radar_pcl)

        lidar_pcl = read_file(file, 'lidar')
        lidar_pcl_set.append(lidar_pcl)

        # plot_3D_animation()
        plot_2D_annotation(n)
        n = n + 1

import json
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from PIL import Image, ImageDraw


def is_float(str_val):
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
                if is_float(line_str[0]):
                    line_float = [float(x) for x in line_str]
                    p.append(line_float)
    pa = np.array(p)
    return pa


# transformation matrix converts from sensorA->sensorB to sensorB->sensorA
def inv_trans(T):
    rotation = np.linalg.inv(T[0:3, 0:3])  # rotation matrix

    translation = T[0:3, 3]
    translation = -1 * np.dot(rotation, translation.T)
    translation = np.reshape(translation, (3, 1))
    Q = np.hstack((rotation, translation))

    # # test if it is truly a roation matrix
    # d = np.linalg.det(rotation)
    # t = np.transpose(rotation)
    # o = np.dot(rotation, t)
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
    # # test if it is truly a roation matrix
    # d = np.linalg.det(rotation)
    # t = np.transpose(rotation)
    # o = np.dot(rotation, t)
    return rot_matrix


# read calibration file and transform inversely
def get_calibration(filename):
    filename = filename[:6] + '.json'
    with open(calib_dir + filename, mode='r') as f:
        data = json.load(f)
    T_fromLidar = np.array(data['sensors'][1]['calib_data']['T_to_ref_COS'])
    T_fromCamera = np.array(data['sensors'][2]['calib_data']['T_to_ref_COS'])
    K = np.array(data['sensors'][2]['calib_data']['K'])

    T_toLidar = inv_trans(T_fromLidar)
    T_toCamera = inv_trans(T_fromCamera)
    return T_toLidar, T_toCamera, K


def get_objects(filename):
    filename = filename[:6] + '.json'
    with open(groundtruth_data_dir + filename, mode='r') as f:
        data = json.load(f)
    objects_info = data['objects']
    objects = []
    classids = []

    for p in objects_info:
        center = np.array(p['center3d'])
        dimension = np.array(p['dimension3d'])
        w = dimension[0]
        l = dimension[1]
        h = dimension[2]
        orientation = np.array(p['orientation_quat'])
        classids.append(p['classname'])

        x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        # rotate and translate 3d bounding box
        R = quat_to_rotation(orientation)
        # case 1: rotate + translate
        bbox = np.vstack([x_corners, y_corners, z_corners])
        bbox = np.dot(R, bbox)
        bbox = bbox + center[:, np.newaxis]

        # case 2: translate + rotate
        # bbox = np.vstack([x_corners, y_corners, z_corners]) + center[:,np.newaxis]
        # bbox = np.dot(R, bbox)

        bbox = np.transpose(bbox)
        objects.append(bbox)
    return objects, classids


def plot_2Dbox_on_pcl(ax, points, classid):
    for k in range(0, 3):
        ax.plot(points[k:k + 2, 0], points[k:k + 2, 1], 'r-')

    ax.plot([points[3, 0], points[0, 0]], [points[3, 1], points[0, 1]], 'r-')
    return 0


def get_objects_lidar(objects, T_toLidar):
    objects_lidar = []
    for obj in objects:
        obj_lidar = np.dot(T_toLidar[0:3, 0:3], np.transpose(obj))
        T = T_toLidar[0:3, 3]
        obj_lidar = obj_lidar + T[:, np.newaxis]
        obj_lidar = np.transpose(obj_lidar)
        objects_lidar.append(obj_lidar)
    return objects_lidar


def get_objects_2Dimage(objects, T_toCamera, K):
    objects_2Dimage = []
    for obj in objects:
        obj_camera = np.dot(T_toCamera[0:3, 0:3], np.transpose(obj))
        T = T_toCamera[0:3, 3]
        obj_camera = obj_camera + T[:, np.newaxis]
        obj_image = np.dot(K, obj_camera)
        obj_image = obj_image / obj_image[2, :]
        obj_image = np.delete(obj_image, 2, 0)
        objects_2Dimage.append(obj_image)
    return objects_2Dimage


def plot_3Dbox_on_image(dx, qs, classid, color, linewidth=3):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
    """
    qs = qs.astype(np.int32)
    qs = np.transpose(qs)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        dx.line([(qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1])], color, width=linewidth)
        i, j = k + 4, (k + 1) % 4 + 4
        dx.line([(qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1])], color, width=linewidth)
        i, j = k, k + 4
        dx.line([(qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1])], color, width=linewidth)
    return 0


def plot_annotation(filename):
    # get radar/lidar pcl data
    radar_pcl = read_file(filename, 'radar')
    lidar_pcl = read_file(filename, 'lidar')

    # get transform matrix from calibration file
    T_toLidar, T_toCamera, K = get_calibration(filename)

    # get ground truth objects info
    objects, classids = get_objects(filename)

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    plt.get_current_fig_manager().full_screen_toggle()

    # plot radar pcl on x-y dimension
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(radar_pcl[:, 0], radar_pcl[:, 1], c='darkblue', s=1, alpha=0.5)
    ax.set_title('Radar Point Cloud 2D Visualization:' + filename[:6] + '.txt')
    ax.set_xlim(0, 100)  # (-100, 100)
    ax.set_ylim(-50, 50)  # (0, 100)

    # add 2D annotation on radar pcl plot
    for obj, classid in zip(objects, classids):
        plot_2Dbox_on_pcl(ax, obj, classid)

    # plot lidar pcl on x-y dimension
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(lidar_pcl[:, 0], lidar_pcl[:, 1], c='darkblue', s=1, alpha=0.5)
    ax.set_title('Lidar Point Cloud 2D Visualization:' + filename[:6] + '.txt')
    ax.set_xlim(-10, 100)
    ax.set_ylim(-50, 50)

    # get ground truth objects in lidar coordinator system
    objects_lidar = get_objects_lidar(objects, T_toLidar)

    # add 2D annotation on lidar pcl
    for obj, classid in zip(objects_lidar, classids):
        plot_2Dbox_on_pcl(ax, obj, classid)

    # plot camera image
    ax = fig.add_subplot(gs[1, :])
    image_path = camera_data_dir + filename[:6] + '.jpg'
    ax.set_title('Camera Image:' + filename[:6] + '.jpg')
    camera_image = Image.open(image_path)
    box_draw = ImageDraw.Draw(camera_image)

    # get ground truth objects in camera image 2D coordinator system
    objects_2Dimage = get_objects_2Dimage(objects, T_toCamera, K)

    # plot 3D annotation on camera image
    number_of_colors = len(objects_2Dimage)
    colorlist = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    for obj, classid, n in zip(objects_2Dimage, classids, range(0, len(objects_2Dimage))):
        plot_3Dbox_on_image(box_draw, obj, classid, colorlist[n])
    ax.imshow(camera_image)

    plt.show()
    return 0


# main
root_dir = os.environ['AOD_HOME']
groundtruth_data_dir = root_dir + 'groundtruth_obj3d/'
calib_dir = root_dir + 'calibration/'
radar_data_dir = root_dir + 'radar_6455/'
lidar_data_dir = root_dir + 'lidar_vlp16/'
camera_data_dir = root_dir + 'camera_front/'

files = os.listdir(radar_data_dir)
files.sort()
# plot the first two files
n = 0
files = files[n:n + 2]
radar_pcl_set = []
lidar_pcl_set = []

for file in files:
    if not os.path.isdir(file):
        plot_annotation(file)
        n = n + 1

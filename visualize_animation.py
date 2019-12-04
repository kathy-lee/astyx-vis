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


def isFloat(str_val):
    try:
        float(str_val)
        return True
    except ValueError:
        return False

# read sensor data file: radar,lidar
def read_file(filename,filetype):

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

def plot_2D_animation():

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()

    ax = fig.add_subplot(gs[0, 0])
    frames_radar = ax.scatter([], [], c='darkblue', s=1, alpha=0.5)
    title1 = ax.set_title('Radar Point Cloud')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(-100, 100)
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(gs[0, 1])
    frames_lidar = ax.scatter([], [], c='darkblue', s=1, alpha=0.5)
    title2 = ax.set_title('Lidar Point Cloud')
    ax.set_xlim(-130, 130)
    ax.set_ylim(-130, 130)

    ax = fig.add_subplot(gs[1,:])
    frames_camera = ax.imshow(np.zeros((618,2048)))
    # add dynamic bbox
    rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    title3 = ax.set_title('Camera Image')

    def update(n):
        p1 = radar_pcl_set[n]
        print(p1.shape)
        #frames_radar.set_offsets(p[:,0:2])
        p1 = np.flip(p1[:,0:2], 1)
        frames_radar.set_offsets(p1)

        p2 = lidar_pcl_set[n]
        print(p2.shape)
        #frames_lidar.set_offsets(p[:,0:2])
        p2 = np.flip(p2[:,0:2], 1)
        frames_lidar.set_offsets(p2)

        image_path = camera_data_dir + str(n).zfill(6) + '.jpg'
        print(image_path)
        image = img.imread(image_path)
        # add dynamic bouding box
        frames_camera.set_data(image)
        rect = patches.Rectangle((50, 50+n*50), 40, 30, linewidth=1, edgecolor='r', facecolor='none')

        title1.set_text('Radar Point Cloud 2D Visualization at time={}'.format(n))
        title2.set_text('Lidar Point Cloud 2D Visualization at time={}'.format(n))
        title3.set_text('Camera Image at time={}'.format(n))

        return frames_radar,frames_lidar,frames_camera,rect

    ani = animation.FuncAnimation(fig, update, len(lidar_pcl_set), interval=2000, repeat=False)

    plt.show()
    return 0


root_dir = os.environ['AOD_HOME']
truth_data_dir = root_dir + 'groundtruth_obj3d/'
calib_dir = root_dir + 'calibration/'
radar_data_dir = root_dir + 'radar_6455/'
lidar_data_dir = root_dir + 'lidar_vlp16/'
camera_data_dir = root_dir + 'camera_front/'


files = os.listdir(radar_data_dir)
files.sort()

radar_max_x=0
radar_min_x=0
radar_max_y=0
radar_min_y=0
radar_pcl_set = []
for file in files:
    if not os.path.isdir(file):
        radar_pcl = read_file(file,'radar')
        radar_pcl_set.append(radar_pcl)
        # if radar_max_x < np.amax(radar_pcl[:,0]):
        #     radar_max_x = np.amax(radar_pcl[:,0])
        # if radar_min_x > np.amin(radar_pcl[:,0]):
        #     radar_min_x = np.amin(radar_pcl[:,0])
        # if radar_max_y < np.amax(radar_pcl[:,1]):
        #     radar_max_y = np.amax(radar_pcl[:,1])
        # if radar_min_y > np.amin(radar_pcl[:,1]):
        #     radar_min_y = np.amin(radar_pcl[:,1])

files = os.listdir(lidar_data_dir)
files.sort()

lidar_max_x=0
lidar_min_x=0
lidar_max_y=0
lidar_min_y=0
lidar_pcl_set = []
for file in files:
    if not os.path.isdir(file):
        lidar_pcl = read_file(file,'lidar')
        lidar_pcl_set.append(lidar_pcl)
        # if lidar_max_x < np.amax(lidar_pcl[:,0]):
        #     lidar_max_x = np.amax(lidar_pcl[:,0])
        # if lidar_min_x > np.amin(lidar_pcl[:,0]):
        #     lidar_min_x = np.amin(lidar_pcl[:,0])
        # if lidar_max_y < np.amax(lidar_pcl[:,1]):
        #     lidar_max_y = np.amax(lidar_pcl[:,1])
        # if lidar_min_y > np.amin(lidar_pcl[:,1]):
        #     lidar_min_y = np.amin(lidar_pcl[:,1])

print(radar_min_x,radar_max_x)
print(radar_min_y,radar_max_y)
print(lidar_min_x,lidar_max_x)
print(lidar_min_y,lidar_max_y)

#plot_3D_animation()
plot_2D_animation()



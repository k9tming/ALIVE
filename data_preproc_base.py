import os
import re
from datetime import datetime
import scipy.io
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import  namedtuple
from datetime import datetime, timedelta
from tqdm import tqdm

MATCH = namedtuple('match', [ 'curr_frame_sig_list', 'timestamp',
                              'infra_file','infra_index',
                              'gt_pos','gt_index','cam_ref',
                              'imu_data',
                              'closest_imu_index' ,  ])

def calculate_hip_positions(LASI, RASI, LPSI, RPSI, vertical_offset):
            # Pelvic center
            # print(LASI)
            # print(RASI)
            PC = (LASI + RASI) / 2.0

            # Pelvic coordinate system
            x_axis = (LASI - RASI) / np.linalg.norm(LASI - RASI)  # Width direction
            z_axis = np.cross((LASI - RASI), (LPSI - RPSI))  # Normal vector to pelvic plane
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)  # Right-hand rule

            # Pelvic width (half for lateral offset)
            lateral_offset = np.linalg.norm(LASI - RASI) / 2.0

            # Local hip positions
            Lhip = PC - lateral_offset * x_axis - vertical_offset * z_axis
            Rhip = PC + lateral_offset * x_axis - vertical_offset * z_axis

            return Lhip, Rhip
class Match:
    def __init__(self,
                curr_frame_sig_list,
                timestamp,
                infra_file,  
                infra_index, 
                gt_pos, gt_index, 
                cam_ref,
                imu_data,
                closest_imu_index ):
        self.infra_file = infra_file   
        self.infra_index = infra_index   
        self.curr_frame_sig_list = curr_frame_sig_list
        self.gt_pos = gt_pos
        self.cam_ref = cam_ref
        self.gt_index = gt_index
        self.imu_data = imu_data
        self.closest_imu_index = closest_imu_index
        self.timestamp = timestamp
# Function to parse timestamp from the filename
def parse_timestamp(filename):
    match = re.search(r'(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})', filename)
    if match:
        return datetime.strptime(" ".join(match.groups()), "%d %H %M %S %f")
    return None

class DataFrame:
    def __init__(self, waist_path, wasit_image, infra_path, infra_image , vio , ground_truth):
        self.waist_path = waist_path
        self.wasit_image = wasit_image
        self.infra_path = infra_path
        self.infra_image = infra_image
        self.vio = vio 
        self.ground_truth = ground_truth

class BaseData:
    
    def read_gt_trc(self, root_path):
        sub_gt_dir = root_path + "/gt"
        for filename in os.listdir(sub_gt_dir):
            if filename.endswith(".trc"):
                csv_file_path = sub_gt_dir + "/" + filename
                self.gt_data = pd.read_csv(csv_file_path, delim_whitespace=True , skiprows=[0,1,2,3])
                self.gt_data_max_rows = self.gt_data.shape[0]
        return
    
    def read_imu_csv(self, root_path):
        root_dir = root_path 
        for filename in os.listdir(root_dir):
            if filename.endswith("imu.csv"):
                csv_file_path = root_dir + "/" + filename
                self.imu_data = pd.read_csv(csv_file_path, delim_whitespace=False)
                self.imu_data_max_rows = self.imu_data.shape[0]
        return

    def read_camera_ref(self, root_path):
        root_dir = root_path 
        for filename in os.listdir(root_dir):
            if filename.endswith("X3D.npy"):
                npy_file_path = root_dir + "/" + filename
                self.cam_refs = np.load(npy_file_path)

    def read_gt_pos(self,index):
        
        LASI = [self.gt_data['X1'].iloc[index] , self.gt_data['Y1'].iloc[index] ,  self.gt_data['Z1'].iloc[index] ]
        RASI = [self.gt_data['X2'].iloc[index] , self.gt_data['Y2'].iloc[index] ,  self.gt_data['Z2'].iloc[index] ]

        LPSI = [self.gt_data['X3'].iloc[index] , self.gt_data['Y3'].iloc[index] ,  self.gt_data['Z3'].iloc[index] ]
        RPSI = [self.gt_data['X4'].iloc[index] , self.gt_data['Y4'].iloc[index] ,  self.gt_data['Z4'].iloc[index] ]

        LKNE = [self.gt_data['X6'].iloc[index] , self.gt_data['Y6'].iloc[index] ,  self.gt_data['Z6'].iloc[index] ]
        RKNE = [self.gt_data['X12'].iloc[index], self.gt_data['Y12'].iloc[index],  self.gt_data['Z12'].iloc[index] ]

        RTIB = [self.gt_data['X13'].iloc[index], self.gt_data['Y13'].iloc[index] , self.gt_data['Z13'].iloc[index] ]
        LTIB = [self.gt_data['X7'].iloc[index] , self.gt_data['Y7'].iloc[index] ,  self.gt_data['Z7'].iloc[index] ]

        RANK = [self.gt_data['X14'].iloc[index], self.gt_data['Y14'].iloc[index] , self.gt_data['Z14'].iloc[index] ]
        LANK = [self.gt_data['X8'].iloc[index] , self.gt_data['Y8'].iloc[index] ,  self.gt_data['Z8'].iloc[index] ]

        RTOE = [self.gt_data['X16'].iloc[index], self.gt_data['Y16'].iloc[index] , self.gt_data['Z16'].iloc[index] ]
        LTOE = [self.gt_data['X10'].iloc[index], self.gt_data['Y10'].iloc[index] , self.gt_data['Z10'].iloc[index] ]

        # cell_value = df.iloc[9560, 0]  
        # print(cell_value==None)
        # print(cell_value=='nan')
        # print(cell_value)
        # # print(df[0])
        # is_empty = pd.isna(cell_value)
        # print("Cell is empty:" if is_empty else "Cell is not empty.")
        gt_pose = {
        "LASI" : np.array(LASI),
        "RASI" : np.array(RASI),
        "LPSI" : np.array(LPSI),
        "RPSI" : np.array(RPSI),
        "LKNE" : np.array(LKNE),
        "RKNE" : np.array(RKNE),
        "LANK" : np.array(LANK),
        "RANK" : np.array(RANK),
        "LTOE" : np.array(LTOE),
        "RTOE" : np.array(RTOE),
        }

        

        midpoint = np.mean([gt_pose['LASI'], gt_pose['RASI'],gt_pose['LPSI'], gt_pose['RPSI']], axis=0)
        vertical_offset = 0.1  # Distance from pelvic center to hips (e.g., 100 mm)
       

        RHIP, LHIP = calculate_hip_positions(gt_pose['LASI'], gt_pose['RASI'], gt_pose['LPSI'], gt_pose['RPSI'], vertical_offset)
        gt_pose['LHIP'] = LHIP
        gt_pose['RHIP'] = RHIP
        # Transform each point relative to the midpoint
        relative_pose_data = {key: np.array(value) - midpoint for key, value in gt_pose.items()}
        
     
        relative_pose_data["MIDP"] = [0., 0., 0.]
        return relative_pose_data

    def find_emg_file_and_extract_timestamp(self, directory):
    # Iterate over each file in the specified directory
        for filename in os.listdir(directory):
            # Check if the file starts with "EMG" and ends with ".mat"
            if filename.startswith("EMG") and filename.endswith(".mat"):
                # Define a regex pattern to match the timestamp in the filename
                match = re.search(r"EMG_R(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})", filename)
                if match:
                    return match.string ,datetime.strptime(" ".join(match.groups()), "%d %H %M %S %f")
        # Return None if no matching file is found
        return None, None


    def __init__(self, root_path=None,output_path=None):
        """
        Initialize the BaseData class with imu, depth, rgb, and emg channels.
        
        :param imu: IMU data (e.g., a list or numpy array).
        :param depth: Depth data (e.g., a 2D or 3D array).
        :param rgb: RGB data (e.g., a 3D array).
        :param emg: EMG data (e.g., a list or numpy array).
        """
        # self.field =  namedtuple('match', [ 'waist_file', 
        #                                     'closest_infra_file', 'min_time_diff_infra', 
        #                                     'closest_depth_file', 'min_time_diff_depth',
        #                                     'curr_frame_sig_list', 
        #                                     'gt_pos','gt_index',
        #                                      'closest_imu_index' , 'timestamp' ])
        self.field =  Match

        self.root_path = root_path
        # self.waist_cam_files = sorted(os.listdir(self.root_path + "/waist_cam"))
        self.infra_cam_files = sorted(os.listdir(self.root_path + "/infra_cam"))
        # self.depth_files = sorted(os.listdir(self.root_path + "/depth"))

        self.read_gt_trc(self.root_path)
        self.read_imu_csv(self.root_path)
        self.read_camera_ref(self.root_path)

        self.emg_file_name , self.emg_gt_start_time = self.find_emg_file_and_extract_timestamp(self.root_path)

        emg_mat_data = scipy.io.loadmat(self.root_path + "/"+ self.emg_file_name)
        self.emg_max_index = len(emg_mat_data['EMG_R'])

        # Create a dictionary to store the parsed timestamps
        # waist_timestamps = {f: parse_timestamp(f) for f in self.waist_cam_files}
        infra_timestamps = {f: parse_timestamp(f) for f in self.infra_cam_files}
        # depth_timestamps = {f: parse_timestamp(f) for f in self.depth_files}

        imu_timestamps = [ parse_timestamp(f) for f in self.imu_data['Timestamp']]
        # print(waist_timestamps)        
        # Find closest infra image for each waist image
        self.matches = []
        self.start_time =  parse_timestamp(self.infra_cam_files[0])

        for emg_index in tqdm(range(self.emg_max_index)):

            curr_frame_sig_list = emg_mat_data['EMG_R'][emg_index]
            emg_time = self.emg_gt_start_time + timedelta( milliseconds = emg_index * 1000 / 1111)
            closest_infra_file = None
            closest_infra_index = None
            gt_indx = None
            gt_pos = None
            imu_data = None
            closest_imu_index = None
            min_time_diff_infra = float("inf")
            min_time_diff_imu = float("inf")

            infra_index = 0
            for infra_file, infra_time in infra_timestamps.items():
                time_diff = abs((infra_time - emg_time).total_seconds() * 1000)  # Difference in ms
                if time_diff < 0.45 and time_diff < min_time_diff_infra:
                    min_time_diff_infra = time_diff
                    closest_infra_file = infra_file
                    closest_infra_index = infra_index
                infra_index += 1
            
            imu_index = 0
            for imu_time in imu_timestamps:
                time_diff = abs((imu_time - emg_time).total_seconds() * 1000)  # Difference in ms
                if time_diff < 0.45 and time_diff < min_time_diff_imu:
                    min_time_diff_imu = time_diff
                    closest_imu_index = imu_index
                    # print("imu",closest_imu_index) 
                    # print("time_diff",time_diff)
                imu_index += 1

            nearest_gt_index = int( (emg_index * 1000 / 1111) * 100 /1000)
            if  abs(((self.emg_gt_start_time + timedelta( milliseconds = nearest_gt_index-1 * 1000 / 100)) - emg_time).total_seconds() * 1000) <0.45:
                gt_indx = nearest_gt_index -1
            if  abs(((self.emg_gt_start_time + timedelta( milliseconds = nearest_gt_index * 1000 / 100)) - emg_time).total_seconds() * 1000) <0.45:
                gt_indx = nearest_gt_index 

            if gt_indx is not None:
                gt_pos = self.read_gt_pos(gt_indx)
                # print(gt_indx)
                # print(emg_time)
                # print(closest_infra_file)
            
            imu_data = self.get_imu_data(closest_imu_index)

            if closest_infra_index is not None:
                camera_ref = self.cam_refs[closest_infra_index]
            else:
                camera_ref = None

            self.matches.append( self.field(curr_frame_sig_list,
                                            emg_time,
                                            closest_infra_file,
                                            closest_infra_index, 
                                            gt_pos, 
                                            gt_indx,
                                            camera_ref,
                                            imu_data,
                                            closest_imu_index   ))
            
        if output_path is not None:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump( self.matches, f)
                

    def get_img_index(self):
        return self.waist_cam_files

    def get_one_frame(self, index):
        return self.matches[index]
    
    def get_one_frame_imgs(self, index):
        # waist_img_path = self.matches[index].waist_file
        # Read the image using OpenCV
        # waist_image = cv2.imread(self.root_path + "/waist_cam"+"/"+ waist_img_path)  # Replace with your image file path
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        # waist_image_rgb = cv2.cvtColor(waist_image, cv2.COLOR_BGR2RGB)
        
        infra_img_path = self.matches[index].infra_file
        infra_img = cv2.imread(self.root_path + "/infra_cam"+"/"+ infra_img_path)  # Replace with your image file path
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        infra_img_rgb = cv2.cvtColor(infra_img, cv2.COLOR_BGR2RGB)

        depth_img_path = self.matches[index].closest_depth_file

        return [None , infra_img]
    
    def get_all_frame(self):
        return self.matches

    def get_one_frame_imu_idx(self, index):
        clst_id =  self.matches[index].closest_imu_index
        return clst_id

    def get_imu_data(self, id):
        if id is None:
            return None
        
        key_list = ['Timestamp', 'Translation_x', 'Translation_y', 'Translation_z', 'Rotation_x',	'Rotation_y', 'Rotation_z', 'Rotation_w']
        # Velocity_x	Velocity_y	Velocity_z	Acceleration_x	Acceleration_y	Acceleration_z
        # imu_field = namedtuple('imu_msg',['timestamp', 'angular_velocity', 'linear_acceleration'])
        col = self.imu_data.iloc[id]
        # print(col)
        # am = np.array([ col['Accel X'], col['Accel Y'], col['Accel Z'] ])
        # wm = np.array([ col['Gyro X'], col['Gyro Y'], col['Gyro Z'] ])

        return np.array([ col['Translation_x'], col['Translation_y'], col['Translation_z'], 
                          col['Rotation_x'], col['Rotation_y'], col['Rotation_z'], col['Rotation_w']
                        ])
        # print(imu_field( (parse_timestamp(col['Timestamp'])-self.start_time).total_seconds() * 1000, wm, am))
        # return imu_field( (parse_timestamp(col['Timestamp'])-self.start_time).total_seconds() , wm, am),col['Translation_z']

    def get_rgb_by_index():
        pass

    def parse(self, line):
        """
        line: (timestamp [ns],
        w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],  
        a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2])
        """
        line = [float(_) for _ in line.strip().split(',')]

        timestamp = line[0] * self.scaler
        wm = np.array(line[1:4])
        am = np.array(line[4:7])
        return self.field(timestamp, wm, am)
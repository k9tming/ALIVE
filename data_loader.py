import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import random
# prediction  (N (camera frames) * 17 * 3 )
# gt  (100 frames  * 17 * 3 )


SLIDE_WINDOW_SIZE = 800


class EMGDataset(Dataset):
    def __init__(self, inputs, targets, reference_inputs, imu_inputs ):
        """
        self defienition of a custom dataset class for loading sliding window generated input and target sequences.

        Args:
            inputs (np.ndarray):  (num_samples, input_length, num_channels)。 N * 120 * 8
            targets (np.ndarray): (num_samples, target_length, num_channels)。 pts_dim * pts_num * 1
        """
        self.referenece_inputs = torch.tensor(reference_inputs, dtype=torch.float32)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.imu_data = torch.tensor(imu_inputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
    
        return self.inputs[idx], self.targets[idx],self.referenece_inputs[idx], self.imu_data[idx]

def slide_window_collect( data,slide_window_list, window_size = SLIDE_WINDOW_SIZE+1):
    slide_window_list.append(data)
    if len(slide_window_list) > window_size:
        slide_window_list.pop(0) 
    if len(slide_window_list) < window_size:
        return slide_window_list , False
    return slide_window_list, True

def dataloader_generator (path, batch_size=256,shuffle=True):

    directory =  path

    key_list = ['MIDP','RHIP','RKNE', 'RANK' ,'LHIP', 'LKNE',  'LANK']
    channel_list = [8,10,11,13,14,15]
    targets = []
    EMG_inputs = []
    Reference_inputs = []
    IMU_inputs = []
    invalid_cnt = 0
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):  # 只处理 .pkl 文件
            filepath = os.path.join(directory, filename)
            print(filepath)
            with open(filepath, 'rb') as f:
                restored_match = pickle.load(f)
            # restored_matches.append(restored_match)
        else:
            continue
        slide_window_list = []
        has_reference_in_window = False
        for i in restored_match:
            slide_window_list,valid_window = slide_window_collect(i, slide_window_list)

            if i.gt_pos == None:
                continue
            if np.isnan(np.array(list(i.gt_pos.values()))).any():
                continue
            for j in slide_window_list:
                if j.cam_ref is not None:
                    has_reference_in_window = True
                    break
            
            if has_reference_in_window and valid_window:
                data_to_collect = slide_window_list[:SLIDE_WINDOW_SIZE]
                ref = []
                for j in data_to_collect:
                    if j.cam_ref is not None:
                        # if  random.random()<0.85:  # 输出0到1之间的浮点数
                        #     j.cam_ref = j.cam_ref
                        # else:
                        #     j.cam_ref = np.zeros((17,3))
                        ref.append( j.cam_ref)
                    else:
                        ref.append(np.zeros((17,3)))
                        # print("ref empty", np.zeros((7,3)))
                ref = np.array(ref)

                # print(ref.shape)
                # if np.all(ref[:-400,:,:] == 0):
                #     continue
                # else:
                ref[-400:,:,:] = 0 

                EMG_inputs.append(np.array( [item.curr_frame_sig_list[channel_list] for item in data_to_collect])) 
                # if (ref[:210,:,:] == 0).all:
                           
                IMU_input = []
                for k in data_to_collect:
                    # print(k)
                    if k.imu_data is not None:
                        # print(k.imu_data.shape)
                        IMU_input.append(k.imu_data)
                    else: 
                        IMU_input.append(np.zeros(7))

                IMU_inputs.append(np.array(IMU_input))
                Reference_inputs.append(ref)
                target = []
                for key in key_list:
                    target.append(i.gt_pos[key])
                targets.append(np.array(target))


    EMG_array = np.stack(EMG_inputs, axis=0)
    targets_array = np.stack(targets, axis=0)
    reference_array = np.stack(Reference_inputs, axis=0)
    imu_array = np.stack(IMU_inputs, axis=0)

    # reference_array [:,:,:,2] = 0
    imu_array[:,:,:3]  = imu_array[:,:,:3]-  imu_array[:,0:1,:3]
    

    for i in range(6):
        EMG_mean = EMG_array[:,:,i].mean()
        EMG_std = EMG_array[:,:,i].std()
        print(EMG_mean,EMG_std)
        EMG_array[:,:,i] = (EMG_array[:,:,i] - EMG_mean) / EMG_std

    print(f"Loaded {(EMG_array.shape)} matches")
    print(f"Loaded {(targets_array.shape)} matches")
    print(f"Loaded {(reference_array.shape)} matches")
    print(f"Loaded {(imu_array.shape)} matches")
    
    dataset = EMGDataset(EMG_array, targets_array,reference_array ,imu_array)
    
    print(f"Dataset size: {len(dataset)}")

    # batch_size = 32 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  
    return dataloader


# time_stamp, data

def interpolation_pos_by_time( time_query, ref_pos_start ,ts_start ,ref_pos2 ,ts2):
    """"time_query:   datetime
    ref_pos1: 3d position  17*3
    ts1: datetime
    ref_pos2: 3d position 17 *3
    ts2: datetime
    """
     # Calculate the time difference ratios
    total_time_diff = (ts2 - ts_start).total_seconds()
    query_time_diff = (time_query - ts_start).total_seconds()
    
    # Calculate the interpolation ratio
    ratio = query_time_diff / total_time_diff
    
    # Perform linear interpolation
    interpolated_pos = ref_pos_start + ratio * (ref_pos2 - ref_pos_start)
    return interpolated_pos

import torch
import os
from model import LSTMModel,LSTMModel_IMU ,NeuroPose
from data_loader import dataloader_generator
import numpy as np

MODEL_PATH = "/your_path_to/models/model_epoch_30.pt"
RESULTS_PATH = "/your_path_to/EMG/1222/res.npy"
GT_PATH = "/your_path_to/EMG/1222/gt.npy"
DATASET_PATH = "/your_path_to/EMG/test_data_preprocd/"

# infer function for testing
def infer(model, dataloader, device='cpu'):
    """
    Perform inference on the model using the provided dataloader.
    Args:
        model: The trained model to be used for inference.
        dataloader: DataLoader providing the input data.
        device: Device to run the inference on ('cpu' or 'cuda').
    Returns:
        results: List of inference results.
        gt: List of ground truth values.
    """
    model.to(device)
    results = []
    gt=[]
    n = 0 
    time_acc = 0
    with torch.no_grad():  # 关闭梯度计算
        for batch_inputs,gt_, batch_reference ,batch_imu in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_reference = batch_reference.to(device)
            batch_imu = batch_imu.to(device)
            
            # 检查是否存在 NaN
            if torch.isnan(batch_inputs).any() or torch.isnan(batch_reference).any():
                print("NaN detected in input data. Skipping this batch.")
                continue
            
            # 开始计时
            # start_time = time.time()

            # 模型前向推理
            outputs = model(batch_inputs.zero_(), batch_reference,batch_imu.zero_())
            # outputs = model(batch_inputs, batch_reference, batch_imu )
            
            # 结束计时
            # end_time = time.time()
            # n += 1 
            # 计算推理时间
            # inference_time = end_time - start_time
            # time_acc += inference_time
            # if n > 2000:
            #    print(f"推理耗时: {time_acc/n}秒")

            # 将结果转换为numpy格式方便后续处理
       
            outputs = outputs.view(7,3)
            gt_ = gt_.view(7,3)
            # print(outputs.shape)
            # print(gt_.shape)
            results.append(outputs.cpu().numpy())
            gt.append(gt_.cpu().numpy())
    return results,gt





def main():
    model_path = MODEL_PATH

    # 模型实例化
    input_size = 128  # EMG 通道数
    hidden_size = 256
    output_size = 21  # 预测未来的 EMG 通道值


    # load the model
    model = NeuroPose()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    inference_dataloader = dataloader_generator(path=DATASET_PATH, batch_size=1)

    # infer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_results, gt = infer(model, inference_dataloader, device)

    np.save(RESULTS_PATH, inference_results, allow_pickle=True, fix_imports=True)
    np.save(GT_PATH, gt, allow_pickle=True, fix_imports=True)
    scores = 0
    from evaluate import calculate_pa_mpjpe
    # 处理推理结果
    for i, result in enumerate(inference_results):
        # print(result )
        # print(gt[i])
        print(calculate_pa_mpjpe(result,gt[i])[0])
        scores += (calculate_pa_mpjpe(result,gt[i])[0])

    avg = scores / len(inference_results)

    print("average score:", avg)
    return




if __name__ == "__main__":
    main()




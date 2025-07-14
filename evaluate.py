import numpy as np

def procrustes_alignment(pred_joints, gt_joints):
    """
    对预测的关节点和真实关节点进行 Procrustes 对齐
    :param pred_joints: 模型预测的关节点坐标，形状 (N, 3)
    :param gt_joints: 真实关节点坐标，形状 (N, 3)
    :return: 对齐后的预测关节点
    """
    # 去中心化
    pred_mean = np.mean(pred_joints, axis=0)
    gt_mean = np.mean(gt_joints, axis=0)
    pred_centered = pred_joints - pred_mean
    gt_centered = gt_joints - gt_mean
    
    # 计算旋转矩阵
    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 防止旋转矩阵出现反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算缩放因子
    scale = np.trace(H @ R) / np.sum(pred_centered ** 2)
    
    # 对齐
    # print("S:,",scale,"R:",R,"M:",gt_mean)
    pred_aligned = scale * pred_centered @ R + gt_mean
    return pred_aligned,[scale,R,gt_mean]

def calculate_pa_mpjpe(pred_joints, gt_joints):
    """
    计算 PA-MPJPE
    :param pred_joints: 模型预测的关节点坐标，形状 (N, 3)
    :param gt_joints: 真实关节点坐标，形状 (N, 3)
    :return: PA-MPJPE 值
    """
    # 对预测关节点进行对齐
    pred_aligned,params = procrustes_alignment(pred_joints, gt_joints)
    # 计算对齐后的 MPJPE
    errors = np.linalg.norm(pred_aligned - gt_joints, axis=1)
    pa_mpjpe = np.mean(errors)
    return pa_mpjpe,params


def calculate_mpjpe(pred_joints, gt_joints):
    """
    计算 PA-MPJPE
    :param pred_joints: 模型预测的关节点坐标，形状 (N, 3)
    :param gt_joints: 真实关节点坐标，形状 (N, 3)
    :return: PA-MPJPE 值
    """
    # 计算对齐后的 MPJPE
    errors = np.linalg.norm(pred_joints - gt_joints, axis=1)
    pa_mpjpe = np.mean(errors)
    return pa_mpjpe
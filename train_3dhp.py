import os
import random
import argparse
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.data.const import H36M_TO_MPI
from lib.data.mpi3dhp_dataset import MPI3DHP
from lib.data.mpi3dhp_dataset_v2 import MPI3DHPTest
from lib.loss.pose3d import (
    loss_mpjpe,
    n_mpjpe,
    loss_velocity,
    loss_limb_var,
    loss_limb_gt,
    loss_angle,
    loss_angle_velocity,
)
from lib.utils.model import load_model, save_checkpoint
from lib.utils.config import get_config
from lib.utils.evaluate import AverageMeter
from lib.utils.data import denormalize, get_variable, flip_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file to use")
    parser.add_argument(
        "--checkpoint", type=str, metavar="PATH", help="checkpoint directory"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="checkpoint file name (default: none)",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training."
    )
    parser.add_argument("--num_cpus", default=4, type=int, help="Number of CPU cores")
    parser.add_argument(
        "--log_name",
        type=str,
        default="blockpose",
        metavar="NAME",
        help="save name (default: blockpose)",
    )
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--keepon", action="store_true", help="continue training")
    parser.add_argument("--eval_only", action="store_true", help="evaluation only")
    opts = parser.parse_args()
    return opts


def mpi_to_h36m(pose):
    MPI_TO_H36M = {v: k for k, v in H36M_TO_MPI.items()}
    joints_map = [MPI_TO_H36M[i] for i in range(17)]
    return pose[..., joints_map, :]


def h36m_to_mpi(pose):
    joints_map = [H36M_TO_MPI[i] for i in range(17)]
    return pose[..., joints_map, :]


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 14:15, :]
            else:
                y[..., 2] = (
                    y[..., 2] - y[:, 0:1, 14:15, 2]
                )  # Place the depth of first frame root to be 0
        x = mpi_to_h36m(x)
        pred = model(x)  # (N, T, 17, 3)
        pred = h36m_to_mpi(pred)

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = (
            loss_3d_pos
            + args.lambda_scale * loss_3d_scale
            + args.lambda_3d_velocity * loss_3d_velocity
            + args.lambda_lv * loss_lv
            + args.lambda_lg * loss_lg
            + args.lambda_a * loss_a
            + args.lambda_av * loss_av
        )

        losses["3d_pose"].update(loss_3d_pos.item(), batch_size)
        losses["3d_scale"].update(loss_3d_scale.item(), batch_size)
        losses["3d_velocity"].update(loss_3d_velocity.item(), batch_size)
        losses["lv"].update(loss_lv.item(), batch_size)
        losses["lg"].update(loss_lg.item(), batch_size)
        losses["angle"].update(loss_a.item(), batch_size)
        losses["angle_velocity"].update(loss_av.item(), batch_size)
        losses["total"].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


# def evaluate(args, model, test_loader, device):
#     model.eval()
#     joints_left = [5, 6, 7, 11, 12, 13]
#     joints_right = [2, 3, 4, 8, 9, 10]

#     data_inference = {}
#     error_sum_test = AverageMeter()

#     for data in tqdm(test_loader, 0):
#         x, y, valid, seq = data
#         batch_size = x.shape[0]

#         x, y, valid = x.to(device), y.to(device), valid.to(device)
#         if args.flip:
#             x_flip = flip_data(x, joints_left, joints_right)
#             pred = model(x)
#             pred_flip = model(x_flip)
#             pred_flip = flip_data(pred_flip, joints_left, joints_right)  # Flip back
#             pred = (pred + pred_flip) / 2
#         else:
#             pred = model(x)

#         if args.root_rel:
#             # pred[:, :, 14, :] = 0
#             y = y - y[..., 14:15, :]
#         else:
#             y[:, 0, 14, 2] = 0

#         y = y * torch.tensor(args.scale).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, y.size(1), 17, 3).to(device)
#         pred = denormalize(pred, seq)
#         pred = pred - pred[..., 14:15, :] # Root-relative prediction
#         inference = pred

#         # bool_valid = valid.unsqueeze(-1).unsqueeze(-1).bool()
#         # pred = pred * bool_valid
#         # y = y * bool_valid
#         joint_error_test = loss_mpjpe(pred, y).item()

#         for seq_cnt in range(len(seq)):
#             seq_name = seq[seq_cnt]
#             temp_infer = inference[seq_cnt].permute(2, 1, 0).cpu().numpy()
#             if seq_name in data_inference:
#                 data_inference[seq_name] = np.concatenate((data_inference[seq_name], temp_infer), axis=2)
#             else:
#                 data_inference[seq_name] = temp_infer

#         error_sum_test.update(joint_error_test, batch_size)

#     for seq_name in data_inference.keys():
#         data_inference[seq_name] = data_inference[seq_name][:, :, None, :]

#     print(f'Protocol #1 Error (MPJPE): {error_sum_test.avg:.2f} mm')
#     return error_sum_test.avg, data_inference


def input_augmentation(input_2D, model, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]
    input_2D = input_2D_non_flip

    input_2D_flip = mpi_to_h36m(input_2D_flip)
    input_2D_non_flip = mpi_to_h36m(input_2D_non_flip)

    output_3D_flip = model(input_2D_flip)
    output_3D_flip = h36m_to_mpi(output_3D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[
        :, :, joints_right + joints_left, :
    ]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_non_flip = h36m_to_mpi(output_3D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    return input_2D, output_3D


def evaluate(args, model, test_loader, device):
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}
    error_sum_test = AverageMeter()

    for data in tqdm(test_loader):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        gt_3D, input_2D, scale, bb_box = (
            gt_3D.to(torch.float32).to(device),
            input_2D.to(torch.float32).to(device),
            scale.to(torch.float32).to(device),
            bb_box.to(torch.float32).to(device),
        )
        # [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable(
        #     "test", [input_2D, gt_3D, batch_cam, scale, bb_box]
        # )
        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, 17, 3).to(device)

        input_2D, output_3D = input_augmentation(
            input_2D, model, joints_left, joints_right
        )

        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
            1, output_3D.size(1), 17, 3
        )
        pad = (args.num_frames - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)

        pred_out[..., 14, :] = 0
        pred_out = denormalize(pred_out, seq)
        pred_out = pred_out - pred_out[..., 14:15, :]  # Root-relative prediction
        inference_out = (
            pred_out + out_target[..., 14:15, :]
        )  # final inference (for PCK and AUC) is not root relative
        out_target = out_target - out_target[..., 14:15, :]  # Root-relative prediction
        joint_error_test = loss_mpjpe(pred_out, out_target).item()

        for seq_cnt in range(len(seq)):
            seq_name = seq[seq_cnt]
            if seq_name in data_inference:
                data_inference[seq_name] = np.concatenate(
                    (
                        data_inference[seq_name],
                        inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy(),
                    ),
                    axis=2,
                )
            else:
                data_inference[seq_name] = (
                    inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()
                )

        error_sum_test.update(joint_error_test, N)

    for seq_name in data_inference.keys():
        data_inference[seq_name] = data_inference[seq_name][:, :, None, :]

    print(f"Protocol #1 Error (MPJPE): {error_sum_test.avg:.2f} mm")

    return error_sum_test.avg, data_inference


def save_data_inference(path, data_inference, latest):
    if latest:
        mat_path = os.path.join(path, "inference_data.mat")
    else:
        mat_path = os.path.join(path, "inference_data_best.mat")
    scio.savemat(mat_path, data_inference)


def train(args, opts):
    print(args)
    if not os.path.exists(opts.checkpoint):
        os.makedirs(opts.checkpoint)

    train_dataset = MPI3DHP(args, train=True)
    # test_dataset = MPI3DHP(args, train=False)
    test_dataset = MPI3DHPTest(args, train=False)
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": opts.num_cpus - 1,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model = load_model(args)
    model = model.to(device)
    n_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters: {}".format(n_params))

    lr = args.learning_rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    lr_decay = args.lr_decay
    min_mpjpe = float("inf")

    stt_epoch = 0
    if opts.checkpoint:
        checkpoint_path = os.path.join(
            opts.checkpoint,
            opts.checkpoint_file if opts.checkpoint_file else "last.pth",
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            if opts.keepon:
                stt_epoch = checkpoint["epoch"]
            if opts.resume:
                lr = checkpoint["lr"]
                optimizer.load_state_dict(checkpoint["optimizer"])
                min_mpjpe = checkpoint["min_mpjpe"]
            print("Checkpoint loaded from {}".format(checkpoint_path))
        else:
            print("No checkpoint found at {}".format(checkpoint_path))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if not opts.eval_only:
        if not os.path.exists("runs"):
            os.makedirs("runs")
        writer = SummaryWriter(log_dir=os.path.join("runs", opts.log_name))

    checkpoint_path_last = os.path.join(opts.checkpoint, "last.pth")
    checkpoint_path_best = os.path.join(opts.checkpoint, "best.pth")

    print("INFO: Training")
    for epoch in range(stt_epoch, args.epochs):
        if opts.eval_only:
            mpjpe, inference = evaluate(args, model, test_loader, device)
            save_data_inference(opts.checkpoint, inference, latest=False)
            exit()

        print("Epoch: {}".format(epoch + 1))
        meters = {
            "3d_pose": AverageMeter(),
            "3d_scale": AverageMeter(),
            "3d_velocity": AverageMeter(),
            "lv": AverageMeter(),
            "lg": AverageMeter(),
            "angle": AverageMeter(),
            "angle_velocity": AverageMeter(),
            "total": AverageMeter(),
        }

        train_one_epoch(args, model, train_loader, optimizer, device, meters)
        mpjpe, inference = evaluate(args, model, test_loader, device)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(
                checkpoint_path_best,
                epoch,
                lr,
                optimizer,
                model,
                min_mpjpe,
                args.config_name,
            )
            save_data_inference(opts.checkpoint, inference, latest=False)
        save_checkpoint(
            checkpoint_path_last,
            epoch,
            lr,
            optimizer,
            model,
            min_mpjpe,
            args.config_name,
        )
        save_data_inference(opts.checkpoint, inference, latest=True)

        writer.add_scalar("loss/3d_pose", meters["3d_pose"].avg, epoch + 1)
        writer.add_scalar("loss/3d_scale", meters["3d_scale"].avg, epoch + 1)
        writer.add_scalar("loss/3d_velocity", meters["3d_velocity"].avg, epoch + 1)
        writer.add_scalar("loss/lv", meters["lv"].avg, epoch + 1)
        writer.add_scalar("loss/lg", meters["lg"].avg, epoch + 1)
        writer.add_scalar("loss/angle", meters["angle"].avg, epoch + 1)
        writer.add_scalar(
            "loss/angle_velocity", meters["angle_velocity"].avg, epoch + 1
        )
        writer.add_scalar("loss/total", meters["total"].avg, epoch + 1)
        writer.add_scalar("eval/mpjpe", mpjpe, epoch + 1)

        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_decay


def main():
    opts = parse_args()

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    args = get_config(opts.config)
    train(args, opts)


if __name__ == "__main__":
    main()


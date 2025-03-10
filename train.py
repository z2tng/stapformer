import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.data.const import (
    H36M_JOINT_TO_LABEL,
    H36M_UPPER_BODY_JOINTS,
    H36M_LOWER_BODY_JOINTS,
    H36M_1_DF,
    H36M_2_DF,
    H36M_3_DF,
)
from lib.data.h36m_dataset import MotionDataset3D
from lib.data.h36m_reader import H36mReader
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
from lib.utils.evaluate import (
    AverageMeter,
    mpjpe as calculate_mpjpe,
    p_mpjpe as calculate_p_mpjpe,
    jpe as calculate_jpe,
)
from lib.utils.data import flip_data


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


def train_one_epoch(args, model, train_loader, optimizer, device, meters):
    model.train()
    for data in tqdm(train_loader):
        x, y = data
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = (
                    y[..., 2] - y[:, 0:1, 0:1, 2]
                )  # Place the depth of first frame root to be 0
        pred = model(x)

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

        meters["3d_pose"].update(loss_3d_pos.item(), batch_size)
        meters["3d_scale"].update(loss_3d_scale.item(), batch_size)
        meters["3d_velocity"].update(loss_3d_velocity.item(), batch_size)
        meters["lv"].update(loss_lv.item(), batch_size)
        meters["lg"].update(loss_lg.item(), batch_size)
        meters["angle"].update(loss_a.item(), batch_size)
        meters["angle_velocity"].update(loss_av.item(), batch_size)
        meters["total"].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, datareader, device):
    results_all = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    with open("data/h36m_16x128_results.pkl", "wb") as f:
        np.save(f, results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset["test"]["action"])
    factors = np.array(datareader.dt_dataset["test"]["2.5d_factor"])
    gts = np.array(datareader.dt_dataset["test"]["joints_2.5d_image"])
    sources = np.array(datareader.dt_dataset["test"]["source"])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, args.num_joints))
    e2_all = np.zeros(num_test_frames)
    # acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    # results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset["test"]["action"]))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        # results_accelaration[action] = []
        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = [
        "s_09_act_05_subact_02",
        "s_09_act_10_subact_02",
        "s_09_act_13_subact_01",
    ]
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        # acc_err = calculate_acc_err(pred, gt)
        # acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            # acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            # results_accelaration[action].append(acc_err)
            for joint_idx in range(args.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
    final_result_procrustes = []
    final_result_joints = [[] for _ in range(args.num_joints)]
    # final_result_acceleration = []
    final_result = []

    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        # final_result_acceleration.append(np.mean(results_accelaration[action]))
        for joint_idx in range(args.num_joints):
            final_result_joints[joint_idx].append(
                np.mean(results_joints[joint_idx][action])
            )
    print("action_names:", action_names)
    print("P1 MPJPE:", final_result)
    print("P2 MPJPE:", final_result_procrustes)

    joint_errors = []
    for joint_idx in range(args.num_joints):
        joint_errors.append(np.mean(np.array(final_result_joints[joint_idx])))
    joint_errors = np.array(joint_errors)
    e1 = np.mean(np.array(final_result))
    assert (
        round(e1, 4) == round(np.mean(joint_errors), 4)
    ), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    # acceleration_error = np.mean(np.array(final_result_acceleration))
    e2 = np.mean(np.array(final_result_procrustes))
    print("Protocol #1 Error (MPJPE):", e1, "mm")
    # print('Acceleration error:', acceleration_error, 'mm/s^2')
    print("Protocol #2 Error (P-MPJPE):", e2, "mm")
    print("----------")
    return e1, e2, joint_errors


def train(args, opts):
    print(args)
    if not os.path.exists(opts.checkpoint):
        os.makedirs(opts.checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, "train")
    test_dataset = MotionDataset3D(args, args.subset_list, "test")
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": opts.num_cpus - 1,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)

    datareader = H36mReader(
        n_frames=args.num_frames,
        sample_stride=1,
        data_stride_train=args.num_frames // 3,
        data_stride_test=args.num_frames,
        dt_root="data/motion3d",
        dt_file=args.dt_file,
    )  # Used for H36m evaluation

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            evaluate(args, model, test_loader, datareader, device)
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
        p1, p2, joint_error = evaluate(args, model, test_loader, datareader, device)

        if p1 < min_mpjpe:
            min_mpjpe = p1
            save_checkpoint(
                checkpoint_path_best,
                epoch,
                lr,
                optimizer,
                model,
                min_mpjpe,
                args.config_name,
            )
        save_checkpoint(
            checkpoint_path_last,
            epoch,
            lr,
            optimizer,
            model,
            min_mpjpe,
            args.config_name,
        )

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
        writer.add_scalar("eval/p1", p1, epoch + 1)
        writer.add_scalar("eval/p2", p2, epoch + 1)
        writer.add_scalar(
            "eval_additional/upper_body_error",
            np.mean(joint_error[H36M_UPPER_BODY_JOINTS]),
            epoch + 1,
        )
        writer.add_scalar(
            "eval_additional/lower_body_error",
            np.mean(joint_error[H36M_UPPER_BODY_JOINTS]),
            epoch + 1,
        )
        writer.add_scalar(
            "eval_additional/1_DF_error", np.mean(joint_error[H36M_1_DF]), epoch + 1
        )
        writer.add_scalar(
            "eval_additional/2_DF_error", np.mean(joint_error[H36M_2_DF]), epoch + 1
        )
        writer.add_scalar(
            "eval_additional/3_DF_error", np.mean(joint_error[H36M_3_DF]), epoch + 1
        )

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
    # from lib.utils.data import read_pkl
    # dt_file = read_pkl('data/motion3d/h36m_sh_conf_cam_source_final.pkl')
    # print(dt_file['test'].keys())
    # print(len(dt_file['test']['action']))
    # print(dt_file['test']['joint3d_image'][:1])
    # print(dt_file['test']['joints_2.5d_image'][:1])
    # print(dt_file['test']['2.5d_factor'][:1])

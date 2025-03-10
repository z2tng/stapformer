import torch
import numpy as np


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def jpe(pred, trgt):
    assert pred.shape == trgt.shape
    return np.linalg.norm(pred - trgt, axis=len(trgt.shape) - 1)


def mpjpe(pred, trgt):
    assert pred.shape == trgt.shape
    return np.mean(np.linalg.norm(pred - trgt, axis=len(trgt.shape) - 1), axis=1)


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(
        np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1),
        axis=len(target.shape) - 2,
    )

def mpjpe_by_action_p1(predicted, target, action, mpjpe_action_meter):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist = torch.mean(
        torch.norm(predicted - target, dim=len(target.shape) - 1),
        dim=len(target.shape) - 2,
    )

    if len(set(list(action))) == 1:
        end_index = action[0].find(" ")
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        mpjpe_action_meter[action_name]["p1"].update(
            torch.mean(dist).item(), batch_num * frame_num
        )
    else:
        for i in range(batch_num):
            end_index = action[i].find(" ")
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            mpjpe_action_meter[action_name]["p1"].update(
                torch.mean(dist[i]).item(), frame_num
            )

    return mpjpe_action_meter


def mpjpe_by_action_p2(predicted, target, action, mpjpe_action_meter):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    
    if len(set(list(action))) == 1:
        end_index = action[0].find(" ")
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        mpjpe_action_meter[action_name]["p2"].update(np.mean(dist), num)
    else:
        for i in range(num):
            end_index = action[i].find(" ")
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            mpjpe_action_meter[action_name]["p2"].update(np.mean(dist), 1)

    return mpjpe_action_meter


def define_mpjpe_meters():
    actions = [
        "Directions",
        "Discussion",
        "Eating",
        "Greeting",
        "Phoning",
        "Photo",
        "Posing",
        "Purchases",
        "Sitting",
        "SittingDown",
        "Smoking",
        "Waiting",
        "WalkDog",
        "Walking",
        "WalkTogether",
    ]

    meters = {}
    meters.update({
        actions[i]: {"p1": AverageMeter(), "p2": AverageMeter()}
        for i in range(len(actions))
    })
    return meters


def print_mpjpe_action(mpjpe_action_meter, is_train):
    mean = {"p1": 0.0, "p2": 0.0}
    mean_meter = {"p1": AverageMeter(), "p2": AverageMeter()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in mpjpe_action_meter.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean["p1"] = value["p1"].avg * 1000.0
        mean_meter["p1"].update(mean["p1"], 1)

        mean["p2"] = value["p2"].avg * 1000.0
        mean_meter["p2"].update(mean["p2"], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean["p1"], mean["p2"]))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_meter["p1"].avg, mean_meter["p2"].avg))

    return mean_meter["p1"].avg, mean_meter["p2"].avg
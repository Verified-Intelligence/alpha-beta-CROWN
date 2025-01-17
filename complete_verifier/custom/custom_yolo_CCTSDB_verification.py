#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import onnx
import os
import onnxruntime
import glob
import pandas as pd
import torch
import torch.nn as nn
import onnx2pytorch
import arguments
from attack.attack_pgd import check_and_save_cex
from read_vnnlib import read_vnnlib


class RecoveredYOLO(nn.Module):
    def __init__(self, yolo_main):
        super(RecoveredYOLO, self).__init__()
        self.yolo_main = yolo_main

    def batch_wise_min(self, a, b):
        return torch.min(
            torch.cat([a.unsqueeze(dim=0), b.unsqueeze(dim=0)], dim=0),
            dim=0,
            keepdim=False,
        ).values

    def batch_wise_max(self, a, b):
        return torch.max(
            torch.cat([a.unsqueeze(dim=0), b.unsqueeze(dim=0)], dim=0),
            dim=0,
            keepdim=False,
        ).values

    def bbox_iou(self, box1, box2):
        # Get the coordinates of bounding boxes

        # @Chenan modify it to batch wise
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[:, 0],
            box1[:, 1],
            box1[:, 2],
            box1[:, 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[:, 0],
            box2[:, 1],
            box2[:, 2],
            box2[:, 3],
        )

        # Intersection area
        inter = (self.batch_wise_min(b1_x2, b2_x2) - self.batch_wise_max(b1_x1, b2_x1)).clamp(0) * (self.batch_wise_min(b1_y2, b2_y2) - self.batch_wise_max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = (w1 * h1 + 1e-16) + w2 * h2 - inter
        iou = inter / union

        return iou

    def eqaul(self, cls1, cls2):
        return cls1 == cls2

    def forward(self, x, gt_label, gt_bbox):
        out_reg_2, out_cls_2 = self.yolo_main(x)

        pred_cls = torch.argmax(out_cls_2, dim=1).detach()
        pred_bbox = out_reg_2.detach()

        cls_res = self.eqaul(pred_cls, gt_label.repeat(pred_cls.size(0))).clone().detach()
        reg_res = self.bbox_iou(gt_bbox.unsqueeze(dim=0).repeat(pred_bbox.size(0), 1), pred_bbox).clone().detach()

        final_out = cls_res.to(torch.long) * reg_res

        return final_out


def split_yolo_CCTSDB(onnx_path, extract_path):
    # vnncomp2023 cctsdb yolo extraction specs
    extract_specs = {
        "patch-1.onnx": (("364",), ("461", "463")),
        "patch-3.onnx": (
            ("input",),
            ("onnx::Gather_437", "onnx::ArgMax_439"),
        ),
    }
    basename = os.path.basename(onnx_path)
    # dirname = os.path.dirname(onnx_path)
    # extracted_model = None
    for filename, input_output in extract_specs.items():
        if filename in basename:
            onnx.utils.extract_model(
                input_path=onnx_path,
                output_path=extract_path,
                input_names=input_output[0],
                output_names=input_output[1],
                # check_model=True,
            )
            return


def customized_yolo_CCTSDB_loader(file_root, onnx_path, vnnlib_path):
    """
    Customized yolo_CCTSDB loader.
    We split the model for verification necessarily part only.
    """
    shape = (-1, 3, 64, 64)
    path = os.path.join(file_root, onnx_path[:-5] + "_split.onnx")
    if not os.path.exists(path):
        print(
            "Split yolo_CCTSDB model from:",
            os.path.join(file_root, onnx_path),
        )
        split_yolo_CCTSDB(os.path.join(file_root, onnx_path), path)
    else:
        print(f"Loaded split model from {path}")

    onnx_model = onnx.load(path)

    model_ori = onnx2pytorch.ConvertModel(
        onnx_model,
        experimental=True,
        quirks={
            "Reshape": {
                "fix_batch_size": True,
                "merge_batch_size_with_channel": True,
            },
            "Transpose": {
                "merge_batch_size_with_channel": True,
            },
        },
    )

    vnnlib = read_vnnlib(os.path.join(file_root, vnnlib_path))

    return model_ori, shape, vnnlib


@torch.no_grad()
def yolo_CCTSDB_verify(model_ori, vnnlib, onnx_path, test_mode=False):
    # load model and do verification here
    basename = os.path.basename(onnx_path)
    device = arguments.Config["general"]["device"] if not test_mode else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RecoveredYOLO(model_ori).to(device)

    # The vnnlib contain inputs_box_range, matrix and rhs of output constraint
    box, mat, rhs = vnnlib[0][0], vnnlib[0][1][0][0], vnnlib[0][1][0][1]
    box, mat, rhs = [torch.tensor(tmp, dtype=torch.float32, device=device) for tmp in [box, mat, rhs]]

    # The inputs_box_range is consisted of three parts: input image (1*3*64*64),
    # position_xs (1) and position_ys (1), gt_bbox (4) and gt_label (1)
    # we extract them here
    image_target_idx = [tmp for tmp in list(range(box.size(0))) if tmp not in [12288, 12289]]
    assert (box[image_target_idx, 0] == box[image_target_idx, 1]).all(), "image and target in the input should be equal!"
    x = box[:12288, 0]
    # get x from vnnlib and make follow operation batch-wise
    img = x.reshape(1, 3, 64, 64)
    position_xs = torch.arange(
        box[12288, 0].type(torch.int64).item(),
        box[12288, 1].type(torch.int64).item(),
    ).to(device)
    position_ys = torch.arange(
        box[12289, 0].type(torch.int64).item(),
        box[12289, 1].type(torch.int64).item(),
    ).to(device)
    grid_x, grid_y = torch.meshgrid(position_xs, position_ys, indexing="ij")
    targets = [box[12290:, 0]]

    gt_bbox = targets[0][2:6]
    gt_label = targets[0][1]

    # build imgs in (62*62, 3, 64, 64) that enumerate all possible perturbations
    combinations = torch.cat([grid_x.unsqueeze(dim=0), grid_y.unsqueeze(dim=0)], dim=0).transpose(0, 1).transpose(1, 2).reshape(-1, 2)
    width = int(basename.split(".onnx")[0][-1])  # decided by onnx_path
    height = int(basename.split(".onnx")[0][-1])
    imgs_ = []

    # apply the zero-patch
    for position_x, position_y in combinations:
        patch_mask = torch.ones_like(img)
        patch_mask[
            :,
            position_x : position_x + height,
            position_y : position_y + width,
            :,
        ] = 0
        imgs_.append((img * patch_mask).detach())
    imgs = torch.cat(imgs_, dim=0)

    # inference in batch-wise
    finalout = model(imgs, gt_label, gt_bbox)

    if ((mat * finalout) > rhs).all():
        return ("safe", finalout) if test_mode else "safe"
    else:
        if not test_mode:
            verified_status, verified_success = save_adv_example_yolo(vnnlib, box, mat, rhs, img, targets, combinations, finalout)
        return (verified_status, finalout) if test_mode else verified_status


def save_adv_example_yolo(vnnlib, box, mat, rhs, img, targets, combinations, finalout):
    adv_indices = ((mat * finalout) <= rhs).squeeze(dim=0).nonzero()
    # we pick the worst-case adv example
    adv_idx = adv_indices[(rhs - (mat * finalout)).squeeze(dim=0)[adv_indices].topk(1).indices[0]].squeeze(dim=0)
    attack_images = torch.cat([img.flatten(1), combinations[adv_idx], targets[0].unsqueeze(0)], dim=1)
    attack_output = finalout[adv_idx].unsqueeze(dim=0)
    return check_and_save_cex(attack_images, attack_output, vnnlib, arguments.Config["attack"]["cex_path"], "unsafe")


def yolo_CCTSDB_verify_onnx(model_ori, vnnlib, onnx_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    box, mat, rhs = vnnlib[0][0], vnnlib[0][1][0][0], vnnlib[0][1][0][1]
    box, mat, rhs = [torch.tensor(tmp, dtype=torch.float32, device=device) for tmp in [box, mat, rhs]]
    image_target_idx = [tmp for tmp in list(range(box.size(0))) if tmp not in [12288, 12289]]
    assert (box[image_target_idx, 0] == box[image_target_idx, 1]).all(), "image and target in the input should be equal!"
    x = box[:12288, 0]
    img = x.reshape(1, 3, 64, 64)
    position_xs = torch.arange(
        box[12288, 0].type(torch.int64).item(),
        box[12288, 1].type(torch.int64).item(),
    )
    position_ys = torch.arange(
        box[12289, 0].type(torch.int64).item(),
        box[12289, 1].type(torch.int64).item(),
    )
    grid_x, grid_y = torch.meshgrid(position_xs, position_ys, indexing="ij")
    targets = [box[12290:, 0]]

    gt_bbox = targets[0][2:6]
    gt_label = targets[0][1]
    len_x = len(position_xs)
    len_y = len(position_ys)
    batch = len_x * len_y
    combinations = torch.cat([grid_x.unsqueeze(dim=0), grid_y.unsqueeze(dim=0)], dim=0).transpose(0, 1).transpose(1, 2).reshape(-1, 2)
    batches = torch.cat(
        [
            img.repeat(batch, 1, 1, 1).flatten(start_dim=1),
            combinations.to(device),
            targets[0].unsqueeze(dim=0).repeat(batch, 1),
        ],
        dim=1,
    )
    finalout_ = []

    ort_session = onnxruntime.InferenceSession(onnx_path)
    inname = [input.name for input in ort_session.get_inputs()]
    outname = [output.name for output in ort_session.get_outputs()]

    for batch in range(batches.size(0)):
        img = batches[batch]
        org_ort_inputs = {inname[0]: img.cpu().numpy().astype("float32")}
        org_ort_outs = ort_session.run(outname, org_ort_inputs)
        finalout_.append(torch.tensor(org_ort_outs[0], device=img.device))
    finalout = torch.cat(finalout_)
    if ((mat * finalout) > rhs).all():
        status = "safe"
    else:
        status = "unsafe"
    return status, finalout


def remove_vnnlib_compiled_and_onnx_split_files():
    # delete previous compiled vnnlib files and split onnx files
    onnx_dir = "../../vnncomp2023_benchmarks/benchmarks/cctsdb_yolo/onnx"
    vnnlib_dir = "../../vnncomp2023_benchmarks/benchmarks/cctsdb_yolo/vnnlib"
    onnx_split_files = [path for path in glob.glob(os.path.join(onnx_dir, "*.onnx")) if "_split" in path]
    vnnlib_compiled_files = glob.glob(os.path.join(vnnlib_dir, "*.vnnlib.compiled"))
    [os.remove(file) for file in onnx_split_files + vnnlib_compiled_files]


def yolo_CCTSDB_test_compare_onnx_torch():
    benchmark_yolo_dir = "../../vnncomp2023_benchmarks/benchmarks/cctsdb_yolo"
    instances_df_path = "../../vnncomp2023_benchmarks/benchmarks/cctsdb_yolo/instances.csv"
    instances_df = pd.read_csv(instances_df_path, header=None, names=["onnx", "vnnlib", "timeout"])
    results = []
    for onnx_, vnnlib_ in zip(instances_df["onnx"].values, instances_df["vnnlib"].values):
        onnx_path = os.path.join(benchmark_yolo_dir, onnx_)
        vnnlib_path = os.path.join(benchmark_yolo_dir, vnnlib_)
        model_ori, shape, vnnlib = customized_yolo_CCTSDB_loader(file_root, onnx_path, vnnlib_path)
        torch_status, torch_out = yolo_CCTSDB_verify(model_ori, vnnlib, onnx_path, test_mode=True)
        onnx_status, onnx_out = yolo_CCTSDB_verify_onnx(model_ori, vnnlib, onnx_path)
        abs_max = torch.abs(torch_out - onnx_out).max().item()
        print(f"onnx path: {onnx_path}")
        print(f"vnnlib path: {vnnlib_path}")
        print(f"abs diff max: {abs_max}, same status: {onnx_status == torch_status}, onnx status: {onnx_status}, torch status: {torch_status}")
        results.append(torch.abs(torch_out - onnx_out).max().item())
        print()
        assert onnx_status == torch_status
        # assert abs_max < 1e-5
    # torch.save({"results": results}, "onnx_torch_compare.pt")


if __name__ == "__main__":
    file_root = ""
    onnx_path = "../../vnncomp2023_benchmarks/benchmarks/cctsdb_yolo/onnx/patch-3.onnx"
    vnnlib_path = "../../vnncomp2023_benchmarks/benchmarks/cctsdb_yolo/vnnlib/spec_onnx_patch-3_idx_02945_0.vnnlib"
    remove_vnnlib_compiled_and_onnx_split_files()
    model_ori, shape, vnnlib = customized_yolo_CCTSDB_loader(file_root, onnx_path, vnnlib_path)
    status, final_output = yolo_CCTSDB_verify(model_ori, vnnlib, onnx_path, test_mode=True)
    print(status, final_output)

    # Test outputs of onnx with pytorch conversion
    remove_vnnlib_compiled_and_onnx_split_files()
    yolo_CCTSDB_test_compare_onnx_torch()

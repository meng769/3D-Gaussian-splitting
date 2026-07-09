import os
import torch
from random import randint

from gaussian_renderer import network_gui
from utils.loss_utils import *
from gaussian_renderer import *
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import *


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
SPARSE_ADAM_AVAILABLE = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # === 初始化电磁信道解码网络 ===
    decoder = ChannelDecoder(em_feature_dim=gaussians.em_feature_dim, M=256, N=4, S=192).cuda()

    # 为解码网络设置独立的优化器
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    # ==========================================

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # === 将以下变量初始化放在循环外 ===
        ema_loss_for_log = 0.0
        ema_nmse_for_log = 0.0

        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1

        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # 1. 选取随机的接收点 (相当于原版的 Camera)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            rand_idx = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)

            # 2. 前向传播：调用电磁渲染函数
            render_pkg = ChannelDecoder(viewpoint_cam, gaussians, decoder)
            H_pred = render_pkg["H_pred"]

            # 为了适配高斯密度的自适应控制，需要提取空间中高斯球的可见性 (通过不透明度判断)
            visibility_filter = (gaussians.get_opacity > 0.001).squeeze()
            # 伪造一个占位半径，以兼容原版的修剪逻辑
            radii = torch.ones_like(visibility_filter, dtype=torch.float32) * 0.05

            # 3. 获取真实信道数据 Ground Truth
            H_gt = viewpoint_cam.gt_channel

            # 4. 计算物理损失
            # (由于复数运算，确保 nmse_loss 等函数内部使用了 torch.abs()**2 来处理复数范数)
            loss_nmse = nmse_loss(H_pred, H_gt)

            # 计算功率时延谱 (PDP) 损失 (对应任务书 C2)
            pdp_pred = compute_pdp(H_pred)
            pdp_gt = compute_pdp(H_gt)
            loss_pdp = cosine_similarity_loss(pdp_pred, pdp_gt)

            # 若需要 PAS 损失
            pas_pred = compute_pas(H_pred)
            pas_gt = compute_pas(H_gt)
            loss_pas = cosine_similarity_loss(pas_pred, pas_gt)

            # 综合损失
            loss = 0.2 * loss_nmse + 0.4 * loss_pdp   + 0.4 * loss_pas

            # 5. 反向传播
            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # 6. 日志与进度条打印
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_nmse_for_log = 0.4 * loss_nmse.item() + 0.6 * ema_nmse_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {"Total Loss": f"{ema_loss_for_log:.5f}", "NMSE": f"{ema_nmse_for_log:.5f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # 保存模型
                if (iteration in saving_iterations):
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    scene.save(iteration)
                    # 同时保存解码网络的权重
                    torch.save(decoder.state_dict(), scene.model_path + f"/decoder_chkpnt{iteration}.pth")

                # 7. 高斯分布自适应密度控制 (Densification)
                if iteration < opt.densify_until_iter:
                    # 记录用于裁剪的最大半径
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])

                    # 计算 3D 空间梯度的范数，用来决定哪些高斯球需要克隆或分裂
                    if gaussians.get_xyz.grad is not None:
                        spatial_grads = gaussians.get_xyz.grad
                        grad_norms = torch.norm(spatial_grads, dim=-1, keepdim=True)
                        gaussians.xyz_gradient_accum[visibility_filter] += grad_norms[visibility_filter]
                        gaussians.denom[visibility_filter] += 1

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # 执行克隆、分裂与裁剪
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                    size_threshold)

                    if iteration % opt.opacity_reset_interval == 0 or (
                            dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # 8. 优化器更新参数
                if iteration < opt.iterations:
                    # 更新高斯球的物理参数 (坐标、缩放、旋转、不透明度、电磁隐特征)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                    # 更新神经网络解码器参数
                    decoder_optimizer.step()
                    decoder_optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print(f"\n[ITER {iteration}] Saving Checkpoint")
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    torch.save(decoder.state_dict(), scene.model_path + f"/decoder_chkpnt{iteration}.pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    #parser.add_argument('--source_path', type=str, default='./data')

    if len(sys.argv) == 1:
        sys.argv.extend(["-s", "D:\Code_Integer\gaussian-splatting-main\data\Round1_Setup.json"])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")

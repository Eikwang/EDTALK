import argparse
import os
import multiprocessing as mp

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train',
                    help='running mode')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--data_path', type=str, help='path of faces')

# optimizer params
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd', 'adagrad', 'rmsprop'])
parser.add_argument('--adam_beta1', type=float, default=0.5,
                    help='value of adam beta 1')
parser.add_argument('--adam_beta2', type=float, default=0.999,
                    help='value of adam beta 2')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')

# training params
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs')
parser.add_argument('--summary_steps', type=int, default=600,
                    help='summary steps')

# dataset params
parser.add_argument('--image_size', type=int, default=64,
                    help='size of cropped images')
parser.add_argument('--ids', type=int, default=100,
                    help='number of identities for training')
parser.add_argument('--correction_ratio', type=float, default=0.5,
                    help='ratio of training pairs forced to target gaze (0,0)')

# evaluation dir
parser.add_argument('--log_dir', type=str, help='path of eval checkpoint')
parser.add_argument('--vgg_path', type=str, help='path of vgg model')

# inference params
parser.add_argument('--input_dir', type=str, help='input image sequence directory for inference')
parser.add_argument('--output_dir', type=str, help='output image sequence directory for inference')
parser.add_argument('--input_video', type=str, help='input video file for inference')
parser.add_argument('--output_video', type=str, help='output video file for inference')
parser.add_argument('--convergence_h', type=float, default=0.05,
                    help='convergence angle for inference')
parser.add_argument('--smooth_alpha', type=float, default=0.4,
                    help='EMA smoothing factor (0-1, lower=smoother)')

# test dir
parser.add_argument('--client_pictures_dir', type=str, help='path of eyes images')

params = parser.parse_args()

if __name__ == '__main__':

    if params.mode == 'inference':
        from src.inference import GazeRedirectInference

        if not params.log_dir:
            raise ValueError("log_dir is not specified!")
        ckpt = os.path.join(params.log_dir, 'current_model.ckpt')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        engine = GazeRedirectInference(
            checkpoint_path=ckpt,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            convergence_h=params.convergence_h,
            smooth_alpha=params.smooth_alpha,
            image_size=params.image_size
        )

        if params.input_dir:
            out = params.output_dir or os.path.join(params.input_dir + '_redirected')
            engine.process_image_sequence(params.input_dir, out)
        elif params.input_video:
            out = params.output_video or params.input_video.rsplit('.', 1)[0] + '_redirected.mp4'
            engine.process_video(params.input_video, out)
        else:
            raise ValueError("Specify --input_dir or --input_video for inference mode")

    else:
        from src.model import Model
        model = Model(params)

        if params.mode == 'train':
            conti = False
            if not os.path.exists(params.log_dir):
                os.mkdir(params.log_dir)
            elif os.path.exists(os.path.join(params.log_dir, 'current_model.ckpt')):
                print("log_dir contains checkpoint! Continue training")
                conti = True
            else:
                print("log_dir exists but no checkpoint found. Starting fresh training.")
            model.train(conti)

        elif params.mode == 'eval':
            if not params.log_dir:
                raise ValueError("log_dir is not specified!")
            model.eval()

        else:
            raise ValueError("mode must be 'train', 'eval', or 'inference'")
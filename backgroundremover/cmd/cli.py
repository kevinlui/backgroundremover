import argparse
import requests
import os
from distutils.util import strtobool
from .. import utilities
from ..bg import removeBG
from urllib.parse import urlparse


def main():
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-m",
        "--model",
        default="u2net",
        type=str,
        choices=model_choices,
        help="The model name, u2net, u2netp, u2net_human_seg",
    )

    ap.add_argument(
        "-a",
        "--alpha-matting",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="When true use alpha matting cutout.",
    )

    ap.add_argument(
        "-af",
        "--alpha-matting-foreground-threshold",
        default=240,
        type=int,
        help="The trimap foreground threshold.",
    )

    ap.add_argument(
        "-ab",
        "--alpha-matting-background-threshold",
        default=10,
        type=int,
        help="The trimap background threshold.",
    )

    ap.add_argument(
        "-ae",
        "--alpha-matting-erode-size",
        default=10,
        type=int,
        help="Size of element used for the erosion.",
    )

    ap.add_argument(
        "-az",
        "--alpha-matting-base-size",
        default=1000,
        type=int,
        help="The image base size.",
    )
    ap.add_argument(
        "-wn",
        "--workernodes",
        default=1,
        type=int,
        help="Number of parallel workers"
    )

    ap.add_argument(
        "-gb",
        "--gpubatchsize",
        default=2,
        type=int,
        help="GPU batchsize"
    )

    ap.add_argument(
        "-fr",
        "--framerate",
        default=-1,
        type=int,
        help="Override the frame rate"
    )

    ap.add_argument(
        "-fl",
        "--framelimit",
        default=-1,
        type=int,
        help="Limit the number of frames to process for quick testing.",
    )
    ap.add_argument(
        "-mk",
        "--mattekey",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Output the Matte key file",
    )
    ap.add_argument(
        "-tv",
        "--transparentvideo",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Output transparent video format mov",
    )

    ap.add_argument(
        "-tov",
        "--transparentvideoovervideo",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Overlay transparent video over another video",
    )
    ap.add_argument(
        "-toi",
        "--transparentvideooverimage",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Overlay transparent video over another image",
    )
    ap.add_argument(
        "-tg",
        "--transparentgif",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Make transparent gif from video",
    )
    ap.add_argument(
        "-tgwb",
        "--transparentgifwithbackground",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: bool(strtobool(x)),
        help="Make transparent background overlay a background image",
    )

    ap.add_argument(
        "-i",
        "--input",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to the input video or image.",
    )

    ap.add_argument(
        "-url",
        "--url",
        nargs="?",
        default="-",
        type=MyUrlType,
        help="URL to the image.",
    )

    ap.add_argument(
        "-bi",
        "--backgroundimage",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to background image.",
    )

    ap.add_argument(
        "-bv",
        "--backgroundvideo",
        nargs="?",
        default="-",
        type=argparse.FileType("rb"),
        help="Path to background video.",
    )

    ap.add_argument(
        "-o",
        "--output",
        nargs="?",
        default="-",
        type=argparse.FileType("wb"),
        help="Path to the output",
    )

    args = ap.parse_args()
    """"
    if args.input.name.rsplit('.', 1)[1] in ['mp4', 'mov', 'webm', 'ogg', 'gif']:
        if args.mattekey:
            print("utilities.matte_key")
            utilities.matte_key(os.path.abspath(args.output.name), os.path.abspath(args.input.name),
                                worker_nodes=args.workernodes,
                                gpu_batchsize=args.gpubatchsize,
                                model_name=args.model,
                                frame_limit=args.framelimit,
                                framerate=args.framerate)
        elif args.transparentvideo:
            print("utilities.transparentvideo")
            utilities.transparentvideo(os.path.abspath(args.output.name), os.path.abspath(args.input.name),
                                       worker_nodes=args.workernodes,
                                       gpu_batchsize=args.gpubatchsize,
                                       model_name=args.model,
                                       frame_limit=args.framelimit,
                                       framerate=args.framerate)
        elif args.transparentvideoovervideo:
            print("utilities.transparentvideoovervideo")
            utilities.transparentvideoovervideo(os.path.abspath(args.output.name), os.path.abspath(args.backgroundvideo.name),
                                                os.path.abspath(args.input.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate)
        elif args.transparentvideooverimage:
            print("utilities.transparentvideooverimage")
            utilities.transparentvideooverimage(os.path.abspath(args.output.name), os.path.abspath(args.backgroundimage.name),
                                                os.path.abspath(args.input.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate)
        elif args.transparentgif:
            print("utilities.transparentgif")
            utilities.transparentgif(os.path.abspath(args.output.name), os.path.abspath(args.input.name),
                                     worker_nodes=args.workernodes,
                                     gpu_batchsize=args.gpubatchsize,
                                     model_name=args.model,
                                     frame_limit=args.framelimit,
                                     framerate=args.framerate)
        elif args.transparentgifwithbackground:
            print("utilities.transparentgifwithbackground")
            utilities.transparentgifwithbackground(os.path.abspath(args.output.name), os.path.abspath(args.backgroundimage.name), os.path.abspath(args.input.name),
                                                   worker_nodes=args.workernodes,
                                                   gpu_batchsize=args.gpubatchsize,
                                                   model_name=args.model,
                                                   frame_limit=args.framelimit,
                                                   framerate=args.framerate)

    else:
    print("args.url: ", args.url)
    # print("args.input.name: ", args.input.name)
    print("args.output.name: ", args.output.name)
    print("args.model: ", args.model)
    print("args.alpha_matting: ", args.alpha_matting)
    print("alpha_matting_foreground_threshold: ", args.alpha_matting_foreground_threshold)
    print("alpha_matting_background_threshold: ", args.alpha_matting_background_threshold)
    print("alpha_matting_erode_size: ", args.alpha_matting_erode_size)
    print("alpha_matting_base_size: ", args.alpha_matting_base_size)
    """

    wr = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)

    """
    # file input code
    rd = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
    wr(
        args.output,
        removeBG(
            rd(args.input),
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=args.alpha_matting_background_threshold,
            alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
            alpha_matting_base_size=args.alpha_matting_base_size,
        ),
    )
    """

    # url arg code
    rd = lambda i: requests.get(args.url).content
    wr(
        args.output,
        removeBG(
            rd(args.url),
            model_name=args.model,
            alpha_matting=args.alpha_matting,
            alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=args.alpha_matting_background_threshold,
            alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
            alpha_matting_base_size=args.alpha_matting_base_size,
        ),
    )

    

def MyUrlType(arg):
    url = urlparse(arg)
    if all((url.scheme, url.netloc)):  # possibly other sections?
        return arg  # return url object, or arg str
    raise argparse.ArgumentTypeError('Invalid URL')


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()

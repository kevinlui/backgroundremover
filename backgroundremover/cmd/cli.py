import argparse
import requests
import os
from distutils.util import strtobool
from .. import utilities
from ..bg import removeBG
from urllib.parse import urlparse
from google.cloud import storage

# CLOUD_STORAGE_BUCKET_NAME = "ks-img"    # under the Google Cloud "Karsearch" project
CLOUD_STORAGE_BUCKET_NAME = "ks-thn"  # under the Firebase "karsearch-c1b75" project

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
        "--in_file",
        nargs="?",
        type=argparse.FileType("rb"),
        help="Path to the input video or image.",
    )

    ap.add_argument(
        "-url",
        "--url",
        nargs="?",
        type=myArgParseUrl,
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
        "--out_file",
        nargs="?",
        type=argparse.FileType("wb"),
        help="Path to the output file",
    )

    ap.add_argument(
        "-s",
        "--storage",
        nargs="?",
        type=str,
        help="Path under the Google Cloud Storage bucket",
    )

    args = ap.parse_args()

    if (args.url is not None):
        print("-url: %s" % args.url)
    if (args.in_file is not None):
        print("-in_file: %s" % args.in_file.name)
    if (args.out_file is not None):
        print("-out_file: %s" % args.out_file.name)
    if (args.storage is not None):
        print("-storage: %s" % args.storage)


    """"
    if args.in_file.name.rsplit('.', 1)[1] in ['mp4', 'mov', 'webm', 'ogg', 'gif']:
        if args.mattekey:
            print("utilities.matte_key")
            utilities.matte_key(os.path.abspath(args.out_file.name), os.path.abspath(args.in_file.name),
                                worker_nodes=args.workernodes,
                                gpu_batchsize=args.gpubatchsize,
                                model_name=args.model,
                                frame_limit=args.framelimit,
                                framerate=args.framerate)
        elif args.transparentvideo:
            print("utilities.transparentvideo")
            utilities.transparentvideo(os.path.abspath(args.out_file.name), os.path.abspath(args.in_file.name),
                                       worker_nodes=args.workernodes,
                                       gpu_batchsize=args.gpubatchsize,
                                       model_name=args.model,
                                       frame_limit=args.framelimit,
                                       framerate=args.framerate)
        elif args.transparentvideoovervideo:
            print("utilities.transparentvideoovervideo")
            utilities.transparentvideoovervideo(os.path.abspath(args.out_file.name), os.path.abspath(args.backgroundvideo.name),
                                                os.path.abspath(args.in_file.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate)
        elif args.transparentvideooverimage:
            print("utilities.transparentvideooverimage")
            utilities.transparentvideooverimage(os.path.abspath(args.out_file.name), os.path.abspath(args.backgroundimage.name),
                                                os.path.abspath(args.in_file.name),
                                                worker_nodes=args.workernodes,
                                                gpu_batchsize=args.gpubatchsize,
                                                model_name=args.model,
                                                frame_limit=args.framelimit,
                                                framerate=args.framerate)
        elif args.transparentgif:
            print("utilities.transparentgif")
            utilities.transparentgif(os.path.abspath(args.out_file.name), os.path.abspath(args.in_file.name),
                                     worker_nodes=args.workernodes,
                                     gpu_batchsize=args.gpubatchsize,
                                     model_name=args.model,
                                     frame_limit=args.framelimit,
                                     framerate=args.framerate)
        elif args.transparentgifwithbackground:
            print("utilities.transparentgifwithbackground")
            utilities.transparentgifwithbackground(os.path.abspath(args.out_file.name), os.path.abspath(args.backgroundimage.name), os.path.abspath(args.in_file.name),
                                                   worker_nodes=args.workernodes,
                                                   gpu_batchsize=args.gpubatchsize,
                                                   model_name=args.model,
                                                   frame_limit=args.framelimit,
                                                   framerate=args.framerate)

    else:
    print("args.url: ", args.url)
    # print("args.in_file.name: ", args.in_file.name)
    print("args.out_file.name: ", args.out_file.name)
    print("args.model: ", args.model)
    print("args.alpha_matting: ", args.alpha_matting)
    print("alpha_matting_foreground_threshold: ", args.alpha_matting_foreground_threshold)
    print("alpha_matting_background_threshold: ", args.alpha_matting_background_threshold)
    print("alpha_matting_erode_size: ", args.alpha_matting_erode_size)
    print("alpha_matting_base_size: ", args.alpha_matting_base_size)
    """

    # *********************************
    # - Input: file system
    # - Output: file system
    # *********************************
    if (args.in_file is not None and args.out_file is not None):
        rd = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
        wr = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)
        wr(
            args.out_file,
            removeBG(
                rd(args.in_file),
                model_name=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            ),
        )

    # *********************************
    # - Input: URL
    # - Output: file system
    # *********************************
    if (args.url is not None and args.out_file is not None):
        wr = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)
        wr(
            args.out_file,
            removeBG(
                requests.get(args.url).content, # make Http Requests on the URL 
                model_name=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            ),
        )


    # *********************************
    # - Input: URL
    # - Output: Google Cloud Storage
    # *********************************
    if (args.url is not None and args.storage is not None):
        upload_data_to_gcs(args.storage, 
            removeBG(
                requests.get(args.url).content, # make Http Requests on the URL 
                model_name=args.model,
                alpha_matting=args.alpha_matting,
                alpha_matting_foreground_threshold=args.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=args.alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=args.alpha_matting_erode_size,
                alpha_matting_base_size=args.alpha_matting_base_size,
            ).tobytes()
        )

    # TODO: can speed up utilizing hyper-threading on the I/O operations:
    #   https://towardsdatascience.com/demystifying-python-multiprocessing-and-multithreading-9b62f9875a27


def upload_data_to_gcs(target_key, data):
    try:
        # see https://cloud.google.com/storage/docs/uploading-objects-from-memory
        client = storage.Client()
        bucket = client.bucket(CLOUD_STORAGE_BUCKET_NAME)
        blob = bucket.blob(target_key)
        blob.upload_from_string(data, content_type='image/png')
        url = bucket.blob(target_key).public_url
        print("Uploaded storage Url = ", url)
        return url

    except Exception as e:
        print(e)

    return None


def myArgParseUrl(arg):
    url = urlparse(arg)
    if all((url.scheme, url.netloc)):  # possibly other sections?
        return arg  # return url object, or arg str
    raise argparse.ArgumentTypeError('Invalid URL argument')


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()

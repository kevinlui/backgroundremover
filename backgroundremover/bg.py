import io
import os
import typing
from PIL import Image, ImageOps
import matplotlib.image as img_matplotlib
import binascii
import scipy
import scipy.misc
import scipy.cluster
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd

from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
import math
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional
from hsh.library.hash import Hasher
from .u2net import detect, u2net
from . import utilities

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        hasher = Hasher()
        model = {
            'u2netp': (u2net.U2NETP,
                       'e4f636406ca4e2af789941e7f139ee2e',
                       '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                       'U2NET_PATH'),
            'u2net': (u2net.U2NET,
                      '09fb4e49b7f785c9f855baf94916840a',
                      '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                      'U2NET_PATH'),
            'u2net_human_seg': (u2net.U2NET,
                                '347c3d51b01528e5c6c071e3cff1cb55',
                                '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                                'U2NET_PATH')
        }[model_name]

        if model_name == "u2netp":
            net = u2net.U2NETP(3, 1)
            path = os.environ.get(
                "U2NETP_PATH",
                os.path.expanduser(os.path.join( "~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                or hasher.md5(path) != "e4f636406ca4e2af789941e7f139ee2e"
            ):
                utilities.downloadfiles_from_github(
                    path, model_name
                )

        elif model_name == "u2net":
            net = u2net.U2NET(3, 1)
            path = os.environ.get(
                "U2NET_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
            ):
                utilities.downloadfiles_from_github(
                    path, model_name
                )

        elif model_name == "u2net_human_seg":
            net = u2net.U2NET(3, 1)
            path = os.environ.get(
                "U2NET_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
            ):
                utilities.downloadfiles_from_github(
                    path, model_name
                )
        else:
            print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)

        net.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
        net.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
        net.eval()
        self.net = net

    def forward(self, block_input: torch.Tensor):
        image_data = block_input.permute(0, 3, 1, 2)
        original_shape = image_data.shape[2:]
        image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='bilinear')
        image_data = (image_data / 255 - 0.485) / 0.229
        out = self.net(image_data)[0][:, 0:1]
        ma = torch.max(out)
        mi = torch.min(out)
        out = (out - mi) / (ma - mi) * 255
        out = torch.nn.functional.interpolate(out, original_shape, mode='bilinear')
        out = out[:, 0]
        out = out.to(dtype=torch.uint8, device=torch.device('cpu'), non_blocking=True).detach()
        return out


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    print('alpha_matting_cutout')

    # size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    # Original code simply resize to original image size
    #cutout = cutout.resize(size, Image.LANCZOS)

    # Cutout to crop out transparent background, to maximize car size before scaling down to thumbnail
    # - https://stackoverflow.com/questions/1905421/crop-a-png-image-to-its-minimum-size
    cutout = cutout.crop(cutout.getbbox())
    # resize down to TN for KarSearch
    KS_THUMBNAIL_WIDTH = 96
    tnHeight = math.floor(KS_THUMBNAIL_WIDTH / cutout.width * cutout.height)
    cutout.thumbnail((KS_THUMBNAIL_WIDTH, tnHeight), Image.LANCZOS)

    return cutout

# https://stackoverflow.com/questions/71748559/given-an-image-crop-of-vehicle-how-to-find-the-colour-of-the-vehicle-with-image
def get_dominant_color_3(pil_img, palette_size=16):
    # Resize image to speed up processing
    img = pil_img.copy()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dominant_color = palette[palette_index*3:palette_index*3+3]

    print(dominant_color)

    return dominant_color

# https://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image
def get_dominant_colors_2(im):
    NUM_CLUSTERS = 5

    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')

    print('most frequent is %s (#%s)' % (peak, colour))


# https://www.geeksforgeeks.org/extract-dominant-colors-of-an-image-using-python/
def get_dominant_colors(pil_image):
    arr = img_matplotlib.pil_to_array(pil_image)
    r = []
    g = []
    b = []
    for row in arr:
        for temp_r, temp_g, temp_b, temp in row:    # temp sometimes exist
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
   
    image_df = pd.DataFrame({'red' : r,
                          'green' : g,
                          'blue' : b})
  
    image_df['scaled_color_red'] = whiten(image_df['red'])
    image_df['scaled_color_blue'] = whiten(image_df['blue'])
    image_df['scaled_color_green'] = whiten(image_df['green'])
  
    cluster_centers, _ = kmeans(image_df[['scaled_color_red',
                                        'scaled_color_blue',
                                        'scaled_color_green']], 3)
  
    dominant_colors = []
    
    red_std, green_std, blue_std = image_df[['red',
                                            'green',
                                            'blue']].std()
    
    for cluster_center in cluster_centers:
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            red_scaled * red_std / 255,
            green_scaled * green_std / 255,
            blue_scaled * blue_std / 255
        ))

    print([dominant_colors])


def naive_cutout(inputImg, mask):
    # print("inputImg w:, h:", inputImg.width, inputImg.height)
    empty = Image.new("RGBA", (inputImg.size), 0)   # need "RGBA" to have transparent background
    cutout = Image.composite(inputImg, empty, mask.resize(inputImg.size, Image.LANCZOS))
    # print("cutout w:, h:", cutout.width, cutout.height)

    # resize down to TN for KarSearch
    KS_THUMBNAIL_WIDTH = 96
    # Cutout to crop out transparent background, to maximize car size before scaling down to thumbnail
    # - https://stackoverflow.com/questions/1905421/crop-a-png-image-to-its-minimum-size
    cutout = cutout.crop(cutout.getbbox())
    # print("cropped-cutout w:, h:", cutout.width, cutout.height)
    tnHeight = math.floor(KS_THUMBNAIL_WIDTH / cutout.width * cutout.height)
    cutout.thumbnail((KS_THUMBNAIL_WIDTH, tnHeight), Image.Resampling.LANCZOS)
    # print("resized-cutout w:, h:", cutout.width, cutout.height)
    #get_dominant_color_3(cutout)

    #tnHeight = math.floor(KS_THUMBNAIL_WIDTH / inputImg.width * inputImg.height)
    #cutout = cutout.resize((KS_THUMBNAIL_WIDTH, tnHeight), Image.LANCZOS)

    return cutout


def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    if model_name == "u2net_human_seg":
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")


def removeBG(
    input_img_data,
    model_name="u2net",
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,
):
    model = get_model(model_name)
    inputImg = Image.open(io.BytesIO(input_img_data)).convert("RGB")

    #get_dominant_color_3(inputImg)

    # https://stackoverflow.com/questions/63947990/why-are-width-and-height-of-an-image-are-inverted-when-loading-using-pil-versus
    inputImg = ImageOps.exif_transpose(inputImg)
    mask = detect.predict(model, np.array(inputImg)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            inputImg,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(inputImg, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")

    return bio.getbuffer()


def iter_frames(path):
    return mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8")


@torch.no_grad()
def remove_many(image_data: typing.List[np.array], net: Net):
    image_data = np.stack(image_data)
    image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
    return net(image_data).numpy()

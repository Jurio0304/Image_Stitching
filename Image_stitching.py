#######################################
# Image stitching
# Jurio.
# 2023/12/21
#######################################

import os
import sys
import time

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from pathlib import Path

from stitching.blender import Blender
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.timelapser import Timelapser
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder

from Feature_matcher import *


################ Plot function####################
def plot_image(img, figsize_in_inches=(10, 10), save=None):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    if save:
        plt.savefig(save, dpi=300)


def plot_images(imgs, figsize_in_inches=(10, 10), save=None):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    if save:
        plt.savefig(save, dpi=300)


################ Load path ####################
def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]


################ Load images ####################
def load_images(img_path):
    images = Images.of(img_path)

    medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
    low_imgs = list(images.resize(Images.Resolution.LOW))
    final_imgs = list(images.resize(Images.Resolution.FINAL))

    return images, low_imgs, medium_imgs, final_imgs


################ Camera Estimation ##################
def camera_correction(features, matches):
    camera_estimator = CameraEstimator()
    camera_adjuster = CameraAdjuster()
    wave_corrector = WaveCorrector()

    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    cameras = wave_corrector.correct(cameras)

    return cameras


################ Warp images ####################
def warp_image(images, cameras, low_imgs, final_imgs):
    warper = Warper()
    warper.set_scale(cameras)

    low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM,
                                     Images.Resolution.LOW)  # since cameras were obtained on medium imgs

    warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
    warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
    low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

    final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

    warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
    warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

    # Timelapser
    timelapser = Timelapser('as_is')
    timelapser.initialize(final_corners, final_sizes)

    frame = []
    for img, corner in zip(warped_final_imgs, final_corners):
        timelapser.process_frame(img, corner)
        frame.append(timelapser.get_frame())

    return (warped_low_imgs, warped_low_masks, low_corners, low_sizes,
            warped_final_imgs, warped_final_masks, final_corners, final_sizes, frame)


################ Crop images ####################
def crop_image(images, warped_low_imgs, warped_low_masks, low_corners, low_sizes,
               warped_final_imgs, warped_final_masks, final_corners, final_sizes):
    cropper = Cropper()

    mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

    lir = cropper.estimate_largest_interior_rectangle(mask)

    low_corners = cropper.get_zero_center_corners(low_corners)
    rectangles = cropper.get_rectangles(low_corners, low_sizes)

    overlap = cropper.get_overlap(rectangles[1], lir)

    intersection = cropper.get_intersection(rectangles[1], overlap)

    cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

    cropped_low_masks = list(cropper.crop_images(warped_low_masks))
    cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
    low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

    lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)  # since lir was obtained on low imgs
    cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
    cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
    final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

    # Redo the timelapse with cropped Images:
    timelapser = Timelapser('as_is')
    timelapser.initialize(final_corners, final_sizes)

    frame = []
    for img, corner in zip(cropped_final_imgs, final_corners):
        timelapser.process_frame(img, corner)
        frame.append(timelapser.get_frame())

    return (cropped_low_imgs, cropped_low_masks, cropped_final_imgs,
            cropped_final_masks, final_corners, final_sizes, frame)


################ Seam Masks ####################
def seam(cropped_low_imgs, low_corners, cropped_low_masks, cropped_final_masks,
         cropped_final_imgs, final_corners):
    seam_finder = SeamFinder()

    seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
    seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]

    seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in
                        zip(cropped_final_imgs, seam_masks)]

    # Exposure Error Compensation
    compensator = ExposureErrorCompensator()

    compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

    compensated_imgs = [compensator.apply(idx, corner, img, mask)
                        for idx, (img, mask, corner)
                        in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]

    return seam_finder, seam_masks_plots, compensated_imgs, seam_masks


################ Main ####################
def stitching(img, detector='orb', crop=False, mask=None):
    """
    Stitching images
    :param img: image name
    :param detector: 'orb', 'sift'
    :param crop: crop image or not
    :param mask: mask image
    :return: None
    """
    # 0. check path
    img_path = get_image_paths(img)
    if not img_path:
        raise FileNotFoundError(f'Image path {img_path} not found.')

    result_path = f'./results'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if crop:
        if mask:
            save_path = f'{result_path}/{detector}/{img}_cropped_mask'
        else:
            save_path = f'{result_path}/{detector}/{img}_cropped'
    else:
        if mask:
            save_path = f'{result_path}/{detector}/{img}_mask'
        else:
            save_path = f'{result_path}/{detector}/{img}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. load
    images, low_imgs, medium_imgs, final_imgs = load_images(img_path)
    images_to_match = medium_imgs

    # 2. plot original images
    plot_images(images_to_match, (20, 20), save=f'{save_path}/1-original.png')

    # 3. print image size
    print(f'Original image size: {images_to_match[0].shape}')

    # 4. Feature detection: ORB, SIFT
    finder = FeatureDetector(detector=detector)
    if mask:
        mask_path = get_image_paths(mask)
        images_mask, _, feature_masks, _ = load_images(mask_path)
        feature_masks = [images_mask.to_binary(mask) for mask in feature_masks]
        features = [finder.detect_with_masks(images_to_match, feature_masks)][0]
    else:
        features = [finder.detect_features(img) for img in images_to_match]
    key_points_img = []
    for i in range(len(images_to_match)):
        key_points_img.append(finder.draw_keypoints(images_to_match[i], features[i]))

    plot_images(key_points_img, (20, 20), save=f'{save_path}/2-key_points.png')

    # 5. Feature matching: homography
    matcher = FeatureMatcher()
    matches = matcher.match_features(features)

    print(matcher.get_confidence_matrix(matches))

    # 6. plot matching
    all_relevant_matches = matcher.draw_matches_matrix(images_to_match, features, matches, conf_thresh=1,
                                                       inliers=True, matchColor=(0, 255, 0))

    for idx1, idx2, img in all_relevant_matches:
        print(f"Matches Image {idx1 + 1} to Image {idx2 + 1}")
        plot_image(img, (20, 10), save=f'{save_path}/3-matching.png')

    # 7. Camera Estimation, Adjustion and Correction
    cameras = camera_correction(features, matches)

    # 8. Warp images
    (warped_low_imgs, warped_low_masks, low_corners, low_sizes,
     warped_final_imgs, warped_final_masks, final_corners, final_sizes, frame) \
        = warp_image(images, cameras, low_imgs, final_imgs)

    plot_images(warped_low_imgs, (10, 10), save=f'{save_path}/4-warped_low_imgs.png')
    plot_images(warped_low_masks, (10, 10), save=f'{save_path}/4-warped_low_masks.png')
    plot_images(frame, (20, 10), save=f'{save_path}/4-warped_final_imgs.png')

    # 9. Crop images
    if crop:
        (cropped_low_imgs, cropped_low_masks, cropped_final_imgs,
         cropped_final_masks, final_corners, final_sizes, frame) = (
            crop_image(images, warped_low_imgs, warped_low_masks, low_corners, low_sizes,
                       warped_final_imgs, warped_final_masks, final_corners, final_sizes))

        plot_images(frame, (20, 10), save=f'{save_path}/5-cropped_final_imgs.png')
    else:
        cropped_low_imgs = warped_low_imgs
        cropped_low_masks = warped_low_masks
        cropped_final_imgs = warped_final_imgs
        cropped_final_masks = warped_final_masks

    # 10. Seam Masks
    seam_finder, seam_masks_plots, compensated_imgs, seam_masks = (
        seam(cropped_low_imgs, low_corners, cropped_low_masks,
             cropped_final_masks, cropped_final_imgs, final_corners))
    plot_images(seam_masks_plots, (15, 10), save=f'{save_path}/6-seam_masks.png')

    # 11. Matching result
    blender = Blender()
    blender.prepare(final_corners, final_sizes)
    for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
        blender.feed(img, mask, corner)
    panorama, _ = blender.blend()
    blended_seam_masks = seam_finder.blend_seam_masks(seam_masks, final_corners, final_sizes)

    plot_image(panorama, (20, 20), save=f'{save_path}/7-matched_result.png')
    plot_image(seam_finder.draw_seam_lines(panorama, blended_seam_masks, linesize=3), (15, 10),
               save=f'{save_path}/8-seam_lines.png')
    plot_image(seam_finder.draw_seam_polygons(panorama, blended_seam_masks), (15, 10),
               save=f'{save_path}/9-seam_polygons.png')

    # 12. Done
    print('Done!')


if __name__ == "__main__":
    ################ Images ####################
    # 'building', 'door', 'sportfield' are from test data
    # 'cat', 'bridge' are photos taken in Hangzhou

    images_list = ['cat', 'bridge', 'building', 'sportfield', 'door', 'buda']
    # images_list = ['barc', 'mask_barc']

    ################ Image stitching ##################
    # detector: 'orb', 'sift'
    det = 'orb'

    for img in images_list:
        print(f"Start stitching {img}...\n")
        t1 = time.time()

        if img == 'mask_barc':
            stitching('barc', detector=det, mask='mask_barc')
            t2 = time.time()
            print(f"{img}_mask cost: {t2 - t1:.3f} s\n")

            stitching('barc', detector=det, crop=True, mask='mask_barc')
            t3 = time.time()
            print(f"{img}_cropped_mask cost: {t3 - t2:.3f} s\n")
        else:
            stitching(img, detector=det)
            t2 = time.time()
            print(f"{img} cost: {t2 - t1:.3f} s\n")

            stitching(img, detector=det, crop=True)
            t3 = time.time()
            print(f"{img}_cropped cost: {t3 - t2:.3f} s\n")

        print(f"Stitching {img} done!")
        print("#############################################\n")

    ################ done ##################

    sys.exit(0)

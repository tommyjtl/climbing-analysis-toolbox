import torch
import cv2
import numpy as np
from tqdm import tqdm
import os


def tensor_to_np(img):
    """
    Convert a torch tensor image (CHW) to a normalized numpy array (HWC, uint8).
    """
    if torch.is_tensor(img):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        return img_np
    return img  # fallback if not tensor


def add_text_to_image(
    image,
    text,
    position=(10, 60),
    font_scale=2.0,
    color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=3,
):
    """
    Overlay text on an image at a given position.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bg_pos = (position[0] + 2, position[1] + 2)
    cv2.putText(image, text, bg_pos, font, font_scale, bg_color, thickness, cv2.LINE_AA)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Adjust the brightness and contrast of an image.
    alpha: Contrast control (1.0 means no change).
    beta: Brightness control (0 means no change).
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def compute_image_contrast(image):
    """
    Compute a simple contrast measure of an image using std of grayscale intensities.
    """
    return np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def save_frame_as_jpg(frame, path):
    """
    Save a video frame as a JPEG file.
    """
    cv2.imwrite(path, frame)


def load_and_prepare_images(reference_path, frame, matcher, temp_frame_path):
    """
    Load reference and target images using matcher loader.
    """
    img_reference = matcher.load_image(reference_path)
    save_frame_as_jpg(frame, temp_frame_path)
    img_target = matcher.load_image(temp_frame_path)
    return img_reference, img_target


def invert_homography(H):
    """
    Invert a homography matrix if not None.
    """
    if H is not None:
        return np.linalg.inv(H)
    return None


def warp_image(img, H, shape):
    """
    Warp an image using a homography matrix to a given shape.
    """
    return cv2.warpPerspective(img, H, shape)


def create_white_canvas_like(img):
    """
    Create a white canvas with the same shape as the input image.
    """
    return np.ones_like(img) * 255


def combine_images_with_masks(R_prime, warped_img, warped_mask):
    """
    Combine reference and warped images using masks to handle overlap and black areas.
    """
    R_double_prime = cv2.bitwise_or(
        R_prime, warped_img, mask=cv2.bitwise_not(warped_mask[:, :, 0])
    )

    mask_black_R_double_prime = cv2.inRange(R_double_prime, (0, 0, 0), (0, 0, 0))
    mask_black_warped_img = cv2.inRange(warped_img, (0, 0, 0), (0, 0, 0))
    mask_non_black_R_double_prime = cv2.bitwise_not(mask_black_R_double_prime)
    mask_non_black_warped_img = cv2.bitwise_not(mask_black_warped_img)

    O = cv2.bitwise_or(
        cv2.bitwise_and(
            R_double_prime, R_double_prime, mask=mask_non_black_R_double_prime
        ),
        cv2.bitwise_and(warped_img, warped_img, mask=mask_black_R_double_prime),
    )
    O = cv2.bitwise_or(
        O,
        cv2.bitwise_and(warped_img, warped_img, mask=mask_non_black_warped_img),
    )
    O = cv2.bitwise_or(
        O,
        cv2.bitwise_and(R_double_prime, R_double_prime, mask=mask_black_warped_img),
    )

    return O


def normalize_and_convert_color(img, convert=True):
    """
    Normalize image to 8-bit and convert BGR to RGB.
    """
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    if convert == True:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_warped_frame_using_diff_H(img_reference_path, frame, matcher, parent_dir):
    """
    Warp a video frame to align with a reference image using feature matching and homography.
    """
    # if parent_dir is empty, use current directory
    if not parent_dir:
        parent_dir = os.getcwd()
    temp_frame_path = f"{parent_dir}/current_frame.jpg"
    img_reference, img_target = load_and_prepare_images(
        img_reference_path, frame, matcher, temp_frame_path
    )
    result = matcher(img_reference, img_target)
    H = result["H"]

    img0_np = tensor_to_np(img_reference)
    img1_np = tensor_to_np(img_target)

    if H is not None:
        H_inv = invert_homography(H)
        h, w = img0_np.shape[:2]
        warped_img1_np = warp_image(img1_np, H_inv, (w, h))
        white_canvas = create_white_canvas_like(img1_np)
        warped_mask = warp_image(white_canvas, H_inv, (w, h))
        R_prime = img0_np.copy()
        O = combine_images_with_masks(R_prime, warped_img1_np, warped_mask)
        O = normalize_and_convert_color(O)
        # cv2.imwrite(f"{parent_dir}/warped_frame.jpg", O)
        return O

    print("Homography matrix is None. Cannot warp the frame.")
    return np.zeros_like(frame)


def compute_homography_once(img_reference_path, target_image_path, matcher):
    """
    Compute homography between reference image and a single target image.
    """
    img_reference = matcher.load_image(img_reference_path)
    img_target = matcher.load_image(target_image_path)
    result = matcher(img_reference, img_target)
    H = result["H"]
    return H, img_reference, img_target


def warp_video_with_fixed_homography(
    img_reference_path,
    target_video_path,
    matcher,
    parent_dir,
    output_video_path,
    overlay_text=True,
):
    file_name = os.path.splitext(os.path.basename(target_video_path))[0]
    ref_img = cv2.imread(img_reference_path)
    ref_height, ref_width = ref_img.shape[:2]
    print(f"Reference image dimensions: {ref_width} x {ref_height}")

    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Target video dimensions: {video_width} x {video_height}")

    # Use output_video_path instead of saving to parent_dir
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ref_width, ref_height))

    # Extract the first frame from the video and save as temp image
    cap_first = cv2.VideoCapture(target_video_path)
    ret_first, first_frame = cap_first.read()
    if not ret_first:
        print("Error: Could not read the first frame from video.")
        return
    # if parent_dir is empty, use current directory
    if not parent_dir:
        parent_dir = os.getcwd()
    temp_first_frame_path = f"{parent_dir}/_temp_first_frame.jpg"
    cv2.imwrite(temp_first_frame_path, first_frame)
    cap_first.release()

    # Compute homography once using the first frame as target image
    H, img_reference, img_target = compute_homography_once(
        img_reference_path, temp_first_frame_path, matcher
    )
    if H is not None:
        H_inv = invert_homography(H)
    else:
        print("Homography matrix is None. Cannot warp the video.")
        return

    # Ensure img_reference is in BGR for consistency with OpenCV
    R_prime_bgr = tensor_to_np(img_reference)
    if R_prime_bgr.shape[2] == 3:
        R_prime_bgr = cv2.cvtColor(R_prime_bgr, cv2.COLOR_RGB2BGR)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        with tqdm(
            total=total_frames, desc="Processing Video (fixed H)", unit="frame"
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Finished reading the video.")
                    break
                img1_np = tensor_to_np(frame)

                # Add file name text to the top left corner of the target frame (after warped, not combined)
                if overlay_text:
                    img1_np_with_text = add_text_to_image(img1_np.copy(), file_name)
                else:
                    img1_np_with_text = img1_np.copy()

                warped_img1_np = warp_image(
                    img1_np_with_text, H_inv, (ref_width, ref_height)
                )
                white_canvas = create_white_canvas_like(img1_np)
                warped_mask = warp_image(white_canvas, H_inv, (ref_width, ref_height))
                # Use BGR reference for combining
                O = combine_images_with_masks(R_prime_bgr, warped_img1_np, warped_mask)
                O = normalize_and_convert_color(O, convert=False)  # keep BGR for OpenCV
                if O.shape[:2] != (ref_height, ref_width):
                    O = cv2.resize(O, (ref_width, ref_height))
                pbar.update(1)
                out.write(O)
            cv2.destroyAllWindows()
    cap.release()
    out.release()


def warp_video_with_per_frame_homography(
    img_reference_path,
    target_video_path,
    matcher,
    parent_dir,
    output_video_path,
    overlay_text=True,
):
    file_name = os.path.splitext(os.path.basename(target_video_path))[0]
    ref_img = cv2.imread(img_reference_path)
    ref_height, ref_width = ref_img.shape[:2]
    print(f"Reference image dimensions: {ref_width} x {ref_height}")

    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Target video dimensions: {video_width} x {video_height}")

    # Use output_video_path instead of saving to parent_dir
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (ref_width, ref_height))

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        with tqdm(
            total=total_frames, desc="Processing Video (per-frame H)", unit="frame"
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Finished reading the video.")
                    break

                # Add file name text to the top left corner of the target frame (after warped, not combined)
                if overlay_text:
                    frame_with_text = add_text_to_image(frame.copy(), file_name)
                else:
                    frame_with_text = frame.copy()

                warped_frame = get_warped_frame_using_diff_H(
                    img_reference_path, frame_with_text, matcher, parent_dir
                )
                if warped_frame.shape[:2] != (ref_height, ref_width):
                    warped_frame = cv2.resize(warped_frame, (ref_width, ref_height))
                pbar.update(1)
                out.write(warped_frame)
            cv2.destroyAllWindows()
    cap.release()
    out.release()

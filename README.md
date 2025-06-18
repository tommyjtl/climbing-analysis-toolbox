# Climbing Analysis Toolbox 

A set of computer vision tools for analyzing your climbing videos.

## Getting Started

```shell
# (Optional) Create an virtual env
python -m venv PATH_TO_YOUR_VENV
source PATH_TO_YOUR_VENV
```

```shell
# Install prerequisites
python -m pip install -r requirements.txt
```

## Catalogue

1. Warping Video for Scene Matching [goto](#1️⃣-warping-video-for-scene-matching)
2. Drawing Trajectories for Body Movements [goto](#2️⃣-drawing-trajectories-for-body-movements)

---

### 1️⃣ Warping Video for Scene Matching

![](./examples/screenshots/warp-dynamic.png)

Sometimes, to analyze our sequences for a climb, we typically have multiple sessions. During those sessions, we might have the camera placed at different locations, thus pointing from different angles towards the climb we are projecting. This tool helps you transform videos so that they match a reference image that corresponds to the whole picture of your climb. Reasons for doing this are: 

1. It is better for using tools that involve 2D/3D pose estimation
2. It is easier to see how your body moves with respect to similar angles. Note that, right now, it is impossible to strictly match a video to the scene of a base image if their camera angles and positions differ by a large amount; some area might be off from base scene.

To warp a video to match a reference scene, we extract the features between two frames, and then a homography matrix is extracted for the image transformation. By default, we use a per-frame homography matrix, but that also means we have to compute $H$ for each frame of the input video if the input video is moving. If the camera of your input video is not moving, we can reduce the processing time by only comparing the first frame of the video and the base scene. This reduces the computation time for the matcher we are using, so only image transformation is involved for the entire warping process. We call the first scenario `dynamic` and the second scenario `fixed`, as you can set with the `type` option.

```shell
# Warp a video with moving camera (per-frame homography matrix for the transformation)
python src/cruxes/warp_video.py \
--src_video_path "examples/videos/warp-dynamic-input.mp4" \
--ref_img "examples/videos/warp-dynamic-ref.jpg"
# by default the type of warping is `dynamic`
```

<details>
    <summary> 🎬 Example Resulting Video </summary>
    <video width="480" controls>
        <source src="examples/videos/warp-dynamic-result.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</details>

```shell
# Warp a video with fixed camera (first-frame homography matrix for the transformation)
python src/cruxes/warp_video.py \
--src_video_path "examples/videos/warp-fixed-input.mp4" \
--ref_img "examples/videos/warp-fixed-ref.jpg" \
--type "fixed"
```

<details>
    <summary> 🎬 Example Resulting Video </summary>
    <video width="480" controls>
        <source src="examples/videos/warp-fixed-result.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</details>

> If you can't see the example resulting video, go to the [example](./examples/videos/) folder.

---

### 2️⃣ Drawing Trajectories for Body Movements

> It is recommended to apply this script to a video with fixed camera position, i.e., camera is not being moved.

![](./examples/screenshots/body-trajectories.png)

There is a couple of settings you can adjust inside the script for `extract_pose_and_draw_trajectory()`:

| Argument | Description | 
| - | - |
| `track_point`  | Points of interest on the estimated pose you want to track. A velocity vector arrow will be drawn to indicate how fast each point is moving with respect to its 3D position |
| `overlay_trajectory`  | Whether to overlay a half-transparent mask on top of the original video. Note that if this is set to `True`, the velocity vector arrow that corresponds to each track point will be removed. |
| `draw_pose`  | Whether to draw pose skeleton or not |
| `kalman_settings`  | Whether to apply Kalman filter to smooth out the trajectory (not the pose itself) |
| `trajectory_png_path`  | Whether to generate a `.png` file for the trajectory with black background |

Then, run the command as follows:

```shell
python src/cruxes/body_trajectory.py \
--video_path "examples/videos/body-trajectory-input.mp4"
```

The generated video will then be located inside of the `output` folder.

<details>
    <summary> 🎬 Example Resulting Video </summary>
    <video width="480" controls>
        <source src="examples/videos/body-trajectory-result.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
   
</details>

> If you can't see the example resulting video, go to the [example](./examples/videos/) folder.

## To-do

- [ ] Migrate to PyPI for easier installation and use.
- [ ] Add a server backend to allow API request for specific functionality.
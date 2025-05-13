# Climbing Analysis Toolbox 

A set of computer vision tools for analyzing your climbing videos.

```bash
python -m pip install -r requirements.txt
```

## Drawing Trajectories for Body Movements

> No documentaion at the moment. Submit an issue or ask ChatGPT if you are having issue running the script.

```bash
python tools/body_trajectory.py --video_path \
    "../videos/may 9 yellow tag circuit/footage/IMG_1722_converted_resized.mp4"
```

Some reference notes:

- ["How can I use smoothing techniques to remove jitter in pose estimation?"](https://stackoverflow.com/questions/52450681/how-can-i-use-smoothing-techniques-to-remove-jitter-in-pose-estimation) on StackOverflow
- ["Savitzky–Golay filter"](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) on Wikipedia
- ["Kalman filter"](https://en.wikipedia.org/wiki/Kalman_filter) on Wikipedia
- Papers
    - ["Temporal Smoothing for 3D Human Pose Estimation and Localization for Occluded People"](https://arxiv.org/abs/2011.00250)
    - ["SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos (ECCV 2022)"](https://ailingzeng.site/smoothnet)
    - ["Fast 3D Pose Estimation With Out-of-Sequence Measurements"](https://dellaert.github.io/files/Ranganathan07iros.pdf)
    - ["Towards Robust and Smooth 3D Multi-Person Pose Estimation from Monocular Videos in the Wild"](https://www.youtube.com/watch?v=yrQ3ZU4zB6Q), also see [[1]](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Towards_Robust_and_Smooth_3D_Multi-Person_Pose_Estimation_from_Monocular_ICCV_2023_paper.pdf)

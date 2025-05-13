import os
import sys
from termcolor import colored


def get_output_path(input_video_path, output_video_path, output_prefix: str) -> str:
    if output_video_path is None:
        # store the output video in `output` folder in the current directory,
        # if the folder doesn't exist, create it
        # extract the file name from the video path
        file_name = os.path.basename(input_video_path)

        # append a prefix to the file name
        file_name = f"{output_prefix}_{file_name}"
        output_path = os.path.join("output", file_name)

        # check if the output folder exists, if not, create it
        if not os.path.exists("output"):
            os.makedirs("output")
            output_path = os.path.join("output", file_name)

        print(
            colored(
                f"Output video will be saved to {output_path}",
                "green",
                attrs=["bold"],
            )
        )

        return output_path
    else:
        # check if the output path specified is valid
        if not os.path.exists(os.path.dirname(output_video_path)):
            raise ValueError(
                f"Output path {output_video_path} does not exist. Please specify a valid path."
            )

        return output_video_path

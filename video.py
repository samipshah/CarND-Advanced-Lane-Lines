from moviepy.editor import ImageSequenceClip
import argparse
import glob


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    files = glob.glob(args.image_folder + "/*.jpg")
    files = sorted(files, key=lambda x: int(x.split("/")[-1].split('.')[0]))
    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(files, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()

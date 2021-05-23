import imageio
import os


def generate_gif(image_paths, gif_path, duration=0.35):
    frames = []
    for image_path in image_paths:
        frames.append(imageio.imread(image_path))

    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)
    print('done')


def main():
    image_folder = "images"
    gif_path = "./images/counting.gif"

    image_paths = []
    files = os.listdir(image_folder)
    print(files)
    for file in files:
        image_path = os.path.join(image_folder, file)
        image_paths.append(image_path)

    duration = 1
    generate_gif(image_paths, gif_path, duration)


if __name__ == "__main__":
    main()

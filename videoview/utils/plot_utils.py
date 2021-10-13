from ..imports import *
from matplotlib.animation import ArtistAnimation,FFMpegWriter

def im_convert(
    tensor,
    denormalize=False,
    denormalize_mean=(0.485, 0.456, 0.406),
    denormalize_std=(0.229, 0.224, 0.225),
):
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    if denormalize:
        image = image * np.array(denormalize_std) + np.array(denormalize_mean)
    image = image.clip(0, 1)
    return image



def save_animation(tensors,filename="./animation"):
    fig, ax = plt.subplots()
    images = []
    for image in tensors:
        images.append([plt.imshow(np.transpose(image.squeeze().numpy(), (1, 2, 0)))])
    anim = ArtistAnimation(fig, images, interval=100, repeat=True)
    writer = FFMpegWriter(fps=5)
    anim.save(f'{filename}.mp4', writer=writer)
    anim.save(f'{filename}.gif', writer='pillow')
    plt.show()
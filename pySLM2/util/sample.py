import os
import PIL
import numpy as np

__all__ = ["number_image"]

_font_path = os.path.join(__file__, "NotoMono-Regular.ttf")


def number_image(i, Nx, Ny):
    img = PIL.Image.new('1', (Nx, Ny))
    draw = PIL.ImageDraw.Draw(img)
    fnt_default = PIL.ImageFont.truetype(_font_path, size=Ny // 2)
    draw.text((0, Ny // 4), str(i), (1), font=fnt_default, align='center')
    return np.array(img, dtype=np.bool)

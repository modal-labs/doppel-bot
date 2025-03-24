from io import BytesIO

import requests
from PIL import Image


def overlay_disguise(avatar_url: str) -> bytes:
    avatar_response = requests.get(avatar_url)
    avatar_img = Image.open(BytesIO(avatar_response.content))

    # Load and resize disguise overlay
    disguise = Image.open("/disguise.png")
    disguise = disguise.resize(avatar_img.size)
    disguise = disguise.convert("RGBA")

    # Composite images
    avatar_img.paste(disguise, (0, 0), disguise)

    # Save to bytes
    img_byte_arr = BytesIO()
    avatar_img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr

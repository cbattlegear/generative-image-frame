"""
This is a simple script to display generated images on a small frame.
This can be run easily on a raspberry pi, but you do need to get the
API keys from their respective services and put them into a config.yaml
"""

#TODO Figure out out to turn script into Debian/Ubuntu service
#TODO Image uploads to blob/delete
#TODO Add the ability to quickly add categories/types of images

import io
import random
import warnings
import datetime
from base64 import b64decode
from PIL import Image
import pygame
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import openai
import yaml

landscape_types = [
    "mountains",
    "hills",
    "plains",
    "plateau",
    "canyon",
    "lakes",
    "stream",
    "river",
    "forest",
    "ocean",
    "field",
    "urban",
    "city",
    "street",
    "desert",
    "backcountry", # DANG
    "galactic", # DANG
    "fjord", # DANG
    "glacier", # DANG
    "scabland", # DANG
    "jungle", # DANG
    "dunes"
]

timeofday = [
    "sunrise",
    "noon",
    "morning",
    "sunset",
    "evening",
    "dawn", # DANG
    "dusk", # DANG
    "night",
    "midnight"
]

weather = [
    "sunny",
    "raining",
    "snowy",
    "cloudy",
    "blizzard",
    "hot",
    "cold",
    "humid",
    "waves"
]

style = [
    "enhance",
    "anime",
    "photographic",
    "digital art",
    "comic book",
    "fantasy art",
    "analog film",
    "neon punk",
    "isometic",
    "cubist", # DANG
    "impressionist", # DANG
    "pencil sketch", # DANG
    "bob ross", # DANG
    "holographic", # DANG
    "80s sci-fi film poster", # DANG
    "low poly"
]

def get_prompt():
    """Generate the prompt based on the lists above"""
    prompt_list = ["landscape"]
    prompt_list.append(random.choice(landscape_types))
    prompt_list.append(random.choice(timeofday))
    prompt_list.append(random.choice(weather))
    prompt_list.append(random.choice(style))
    return " ".join(prompt_list)

def size_image(img_edit, prompt, gen_width, gen_height, screen_width, screen_height):
    """Resize/crop the image to fit the resolution of the screen"""
    # if the image is larger than the screen we are going to resize to fit
    # (either width or height) then crop the opposite
    # If the screen is larger than the image, we will just resize up and crop to fit
    # realized my logic worked either way... so just one branch

    width_ratio = screen_width/float(gen_width)
    height_ratio = screen_height/float(gen_height)
    resize_width = 0
    resize_height = 0
    # Here we are making sure if we resize that we will have to crop instead of have stretching
    if gen_height * width_ratio >= screen_height:
        resize_width = int(screen_width)
        resize_height = int(gen_height * width_ratio)
    else:
        resize_width = int(gen_width * height_ratio)
        resize_height = int(screen_height)

    #resize to calculated width/height from above
    size = (resize_width, resize_height)
    img_edit.resize(size, Image.LANCZOS)

    w_diff_center = (resize_width - screen_width)/2
    h_diff_center = (resize_height - screen_height)/2

    left = w_diff_center
    right = resize_width-w_diff_center
    top = h_diff_center
    bottom = resize_height-h_diff_center
    img_save = img_edit.crop((left, top, right, bottom))
    img_save.save(
        f"images/{int(datetime.datetime.now().timestamp())}-{prompt.replace(' ', '-')}.png")
    return pygame.image.frombytes(img_save.tobytes(), (screen_width, screen_height), "RGB")

def generate_openai(screen_width, screen_height):
    """Generate an image with OpenAI DALL-E 2 APIs"""
    gen_width = 1024
    gen_height = 1024
    prompt = get_prompt()

    image_response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=str(gen_width)+"x"+str(gen_height),
        response_format="b64_json"
    )

    # Make our image into an object
    img_edit = Image.open(io.BytesIO(b64decode(image_response.data[0].b64_json)))

    return size_image(img_edit, prompt, gen_width, gen_height, screen_width, screen_height)

def generate_stable_diff(screen_width, screen_height, stability_api):
    """Generate an image with stable diffusion APIs"""
    # Set up our initial generation parameters.
    gen_width = 896
    gen_height = 512

    prompt = get_prompt()
    answers = stability_api.generate(
        prompt=prompt,
        seed=int(random.random() * 1000000000),
        steps=15,
        cfg_scale=8.0,
        width=gen_width,
        height=gen_height,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img_edit = Image.open(io.BytesIO(artifact.binary))

    return size_image(img_edit, prompt, gen_width, gen_height, screen_width, screen_height)

def main():
    """
    Our main loop, the primary use here is setting up the display and opening the config
    We are also handling timing with pygame events
    """
    with open('config.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=config['api']['stability_key'], # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-xl-beta-v2-2-2", # Set the engine
    )

    openai.api_key = config['api']['openai_key']
    # Properly handling pygame with pylint
    # pylint: disable=no-member
    pygame.init()
    # pylint: enable=no-member
    info_object = pygame.display.Info()

    screen_width = info_object.current_w
    screen_height = info_object.current_h

    flags = pygame.FULLSCREEN
    window_surface = pygame.display.set_mode((screen_width, screen_height), flags)
    pygame.mouse.set_visible(False)

    new_image = True
    use_openai = False
    loop_count = 0

    change_picture = config['settings']['change_time'] # minutes

    start_time = config['settings']['start_hour']
    end_time = config['settings']['end_hour']

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # pylint: disable=no-member
                pygame.quit()
                # pylint: enable=no-member
        if new_image:
            if use_openai:
                img_display = generate_openai(screen_width, screen_height)
                use_openai = False
            else:
                img_display = generate_stable_diff(screen_width, screen_height, stability_api)
                use_openai = True
            window_surface.blit(img_display, (0, 0)) #Replace (0, 0) with desired coordinates
            pygame.display.flip()
            new_image = False
        else:
            loop_count = loop_count + 1
            if loop_count >= (change_picture * 60):
                current_time = datetime.datetime.now()
                if(current_time.hour >= start_time and current_time.hour < end_time):
                    pygame.transform.gaussian_blur(window_surface, 20, dest_surface=window_surface)
                    pygame.display.update()
                    new_image = True
                loop_count = 0
            else:
                pygame.time.wait(1000)

if __name__=="__main__":
    main()

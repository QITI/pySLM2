from pySLM2.util import LuxbeamController

luxbeam = LuxbeamController()
luxbeam.initialize()

# generate two images
image_1 = luxbeam.number_image(1)
image_2 = luxbeam.number_image(2)

luxbeam.load_multiple([image_1, image_2])

print("Use trigger (software or external) to switch between images")
while True:
    input("Press Enter to fire software trigger...")
    luxbeam.fire_software_trigger()
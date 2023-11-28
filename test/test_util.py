import pySLM2
import pySLM2.util
import random
import pytest
import time


def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False


@pytest.mark.luxbeam
def test_luxbeam_load_single(luxbeam_ip):
    luxbeam = pySLM2.util.LuxbeamController(ip=luxbeam_ip)
    luxbeam.initialize()
    number = random.randint(100,999)
    luxbeam.load_single(luxbeam.number_image(number))

    assert yesno("Is number {0} displayed on the DMD".format(number))


@pytest.mark.alp
def test_alp_Nx(alp_version):
    alp = pySLM2.util.ALPController(version=alp_version)
    alp.initialize()

    assert alp.Nx == 2560
    alp.close()

@pytest.mark.alp
def test_alp_Ny(alp_version):
    alp = pySLM2.util.ALPController(version=alp_version)
    alp.initialize()

    assert alp.Ny == 1600
    alp.close()

@pytest.mark.alp
def test_alp_load_single(alp_version):
    alp = pySLM2.util.ALPController(version=alp_version)
    alp.initialize()

    number = random.randint(1, 100)

    alp.load_single(alp.number_image(number))

    assert yesno("Is number {0} displayed on the DMD".format(number))

    # time.sleep(10)
    alp.close()

@pytest.mark.alp
def test_alp_load_multiple():
    NUM_IMAGES = 2
    PICTURE_TIME = 2000000

    alp = pySLM2.util.ALPController()
    alp.initialize()

    number_lst = [random.randint(1, 100) for _ in range(NUM_IMAGES)]
    number_img = [alp.number_image(num) for num in number_lst]
    alp.load_multiple(number_img, picture_time=PICTURE_TIME)

    for number in number_lst:
        input('manually trigger the next image')
        assert yesno("Is number {0} displayed on the DMD".format(number))

    alp.close()

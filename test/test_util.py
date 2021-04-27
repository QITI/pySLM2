import pySLM2
import pySLM2.util
import random
import pytest

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


import os


def pytest_addoption(parser):
    parser.addoption('--volledig', action='store_true', default=False,
                     help='gebruik de volledige database in plaats van de kleine testdatabase')


def pytest_configure(config):
    if config.getoption('--volledig'):
        os.environ['SCHAATS_VOLLEDIG'] = '1'

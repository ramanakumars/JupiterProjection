import os
from bs4 import BeautifulSoup
import requests
import fnmatch
import tqdm

BASEURL = 'https://naif.jpl.nasa.gov/pub/naif/'
KERNEL_DATAFOLDER = './kernels/'


def fetch_kernels_from_https(path, pattern):
    with requests.get(path) as response:
        soup = BeautifulSoup(response.text, 'html.parser')
    kernels_all = [a['href'] for a in soup.find('pre').find_all('a')]
    base_path = path.replace(BASEURL, '')
    return [f"{base_path}/{kernel}" for kernel in fnmatch.filter(kernels_all, pattern)]


def check_and_download_kernels(kernels, KERNEL_DATAFOLDER):
    kernel_fnames = []
    for kernel in kernels:
        if not os.path.exists(os.path.join(KERNEL_DATAFOLDER, kernel)):
            download_kernel(kernel, KERNEL_DATAFOLDER)
        kernel_fnames.append(os.path.join(KERNEL_DATAFOLDER, kernel))

    return kernel_fnames


def download_kernel(kernel, KERNEL_DATAFOLDER):
    link = os.path.join(BASEURL, kernel)
    file_name = os.path.join(KERNEL_DATAFOLDER, kernel)
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            total_length = int(total_length)
            with tqdm.tqdm(total=total_length, unit='B', unit_scale=True, unit_divisor=1024, dynamic_ncols=True, ascii=True, desc=f'Downloading {kernel}') as pbar:
                for data in response.iter_content(chunk_size=4096):
                    f.write(data)
                    pbar.update(len(data))


def get_kernels(KERNEL_DATAFOLDER, target='JUPITER'):
    if not os.path.exists(KERNEL_DATAFOLDER):
        os.mkdir(KERNEL_DATAFOLDER)

    for folder in ['generic_kernels/pck/', 'generic_kernels/spk/planets/', 'generic_kernels/spk/satellites/', 'generic_kernels/fk/planets/', 'generic_kernels/lsk/']:
        if not os.path.exists(os.path.join(KERNEL_DATAFOLDER, folder)):
            os.makedirs(os.path.join(KERNEL_DATAFOLDER, folder))

    pcks = fetch_kernels_from_https(BASEURL + "generic_kernels/pck/", "pck00011*.tpc")
    pcks.extend(fetch_kernels_from_https(BASEURL + "generic_kernels/pck/", "earth_latest_high_prec.bpc"))
    spks1 = fetch_kernels_from_https(BASEURL + "generic_kernels/spk/planets/", "de441*.bsp")
    spks2 = fetch_kernels_from_https(BASEURL + "generic_kernels/spk/satellites/", f"{target[:3].lower()}*.bsp")
    fks = fetch_kernels_from_https(BASEURL + "generic_kernels/fk/planets/", "*.tf")
    lsks = fetch_kernels_from_https(BASEURL + "generic_kernels/lsk/", "latest_leapseconds.tls")

    kernels = [*pcks, *spks1, spks2[-1], *fks, *lsks]

    return check_and_download_kernels(kernels, KERNEL_DATAFOLDER)

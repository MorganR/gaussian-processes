import requests
import numpy as np
import re
import base64

num_images = 10

img_size = 50
scale = 0.05
url = 'http://casjobs.sdss.org/ImgCutoutDR7/ImgCutout.asmx/GetJpeg'

objid, spiral, elliptical, uncertain, nvote = np.loadtxt('galaxy-info-2000.csv', skiprows=2, usecols=[0, 3, 4, 5, 6], delimiter=',', unpack=True, dtype=np.int64)
ra, dec, cs_debiased, el_debiased, p_merge, p_edge, p_acw, p_cw, p_disk, petroRad_g, petroRad_i, petroRad_r = np.loadtxt('galaxy-info-2000.csv', skiprows=2, usecols=[1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], delimiter=',', unpack=True)

i = -1
for n in range(0, num_images):
    i += 1
    while spiral[i] == 0 or cs_debiased[i] < 0.85:
       i += 1

    payload = {
        'ra_': ra[i],
        'dec_': dec[i],
        'scale_': scale*petroRad_i[i],
        'width_': img_size,
        'height_': img_size,
        'opt_': ''# ,
        # 'query_': '',
        # 'imgtype_': '',
        # 'imgfield_': ''
    }
    print(objid[i], payload)
    result = requests.get(url, params=payload)
    parser = re.compile(r'<base64Binary xmlns="[^"]+">([^<]+)</base64Binary>')
    parsed_result = parser.search(str(result.content))
    encoded_img = parsed_result.group(1)
    img = base64.b64decode(encoded_img)
    img_name = 'spirals/{}.jpg'.format(objid[i])
    with open(img_name, 'wb') as f:
        f.write(img)

import requests
import numpy as np
import re
import base64
from PIL import Image
import io

num_images = 2000

img_size = 64
scale = 0.05
url = 'http://casjobs.sdss.org/ImgCutoutDR7/ImgCutout.asmx/GetJpeg'
base_files = ['spirals', 'ellipticals', 'uncertains', 'spiral-edges']
files = ('2000-'+bf for bf in base_files)

for bf,f in zip(base_files, files):
    objid, spiral, elliptical, uncertain, nvote = np.loadtxt(
        f+'.csv', skiprows=2, usecols=[0, 3, 4, 5, 6], delimiter=',', unpack=True, dtype=np.int64)
    ra, dec, cs_debiased, el_debiased, p_merge, p_edge, p_acw, p_cw, p_disk, petroRad_g, petroRad_i, petroRad_r = np.loadtxt(
        f+'.csv', skiprows=2, usecols=[1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], delimiter=',', unpack=True)

    max_images = min(num_images, objid.size)
    train_images = int(max_images/2)
    test_images = max_images - train_images

    byte_filenames = [
        str(train_images)+'-train-'+bf+'.ubyte',
        str(test_images)+'-test-'+bf+'.ubyte']
    folders = ['train-'+bf, 'test-'+bf]
    
    start_i = 0
    for byte_filename, num_per_file, folder in zip(byte_filenames, [train_images, test_images], folders):
        with open(byte_filename, 'wb') as image_byte_data:
            # print number of images
            image_byte_data.write((num_per_file).to_bytes(4, byteorder='big'))
            # print num image rows
            image_byte_data.write((img_size).to_bytes(4, byteorder='big'))
            # print num image cols
            image_byte_data.write((img_size).to_bytes(4, byteorder='big'))

            for i in range(start_i, num_per_file + start_i):
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
                result = requests.get(url, params=payload)
                parser = re.compile(r'<base64Binary xmlns="[^"]+">([^<]+)</base64Binary>')
                parsed_result = parser.search(str(result.content))
                encoded_img = parsed_result.group(1)
                img = base64.b64decode(encoded_img)
                img_base_name = folder+'/{}'.format(objid[i])
                img_name = img_base_name+'.jpg'
                with open(img_name, 'wb') as img_file:
                    img_file.write(img)
                im = Image.open(img_name)
                # convert to black and white
                im_bw = im.convert('L')
                im_bw.save(img_base_name+'-bw.jpg')
                np_array = np.array(im_bw.getdata(), dtype=np.uint8)
                image_byte_data.write(np_array.tobytes())
            start_i = num_per_file

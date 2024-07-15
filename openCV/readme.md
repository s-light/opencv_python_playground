# examples for OpenCV

source: `pfandautomat`
- detect and read barcode
- check if barcode is in `valid_list`
- toggle output pin


## How to control IO lines
https://gpiozero.readthedocs.io/en/stable/recipes.html

### do not use the grove shield
https://github.com/Seeed-Studio/grove.py/issues/73


## samba share
https://www.raspberrypi.com/documentation/computers/remote-access.html#samba-smbcifs
(i did not get this to work)
→ better option to setup sftp?!
and then use [sftp extension](https://github.com/Natizyskunk/vscode-sftp) for vscodium
(this allows upload on save)

## motion-detection

https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
https://learnopencv.com/moving-object-detection-with-opencv/

some first tests can be seen in the `laptop` sub folder.

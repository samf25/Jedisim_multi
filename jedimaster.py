#! /usr/bin/env python
# last modified 8 July 2013
import os
import sys
import subprocess
import math
import re
import shutil
import copy

# parse command line inputs
if (len(sys.argv) != 3):
    print("Usage: jedimaster config")
    sys.exit(1)

print("Config file: %s" % sys.argv[1])

#get threadcount
threads = sys.argv[2]

# parse config file
config_file = open(sys.argv[1], 'r')
config = {}
string_regex = re.compile('"(.*?)"')
value_regex = re.compile('[^ |\t]*')
for line in config_file:
    if not line.startswith("#"):
        temp = line.split("=")
        if temp[1].startswith("\""):
            config[temp[0]] = string_regex.findall(temp[1])[0]
        else:
            config[temp[0]] = value_regex.findall(temp[1])[0]


config_record = copy.deepcopy(config)

# make the filenames from the config parameters
config['HST_image'] = config['output_folder'] + \
    config['prefix']+config['HST_image']
config['HST_convolved_image'] = config['output_folder'] + \
    config['prefix']+config['HST_convolved_image']
config['LSST_convolved_image'] = config['output_folder'] + \
    config['prefix']+config['LSST_convolved_image']
config['LSST_convolved_noise_image'] = config['output_folder'] + \
    config['prefix']+config['LSST_convolved_noise_image']
config['catalog_file'] = config['output_folder'] + \
    config['prefix'] + config['catalog_file']
config['dislist_file'] = config['output_folder'] + \
    config['prefix'] + config['dislist_file']
config['distortedlist_file'] = config['output_folder'] + \
    config['prefix'] + config['distortedlist_file']
config['convolvedlist_file'] = config['output_folder'] + \
    config['prefix'] + config['convolvedlist_file']

# if the output folder doesn't exist, create it
if (not os.path.exists(config['output_folder'])):
    os.makedirs(config['output_folder'])

# make folder for convolved images
convolved_path = "%sconvolved/" % config['output_folder']
if not os.path.exists(convolved_path):
    os.makedirs(convolved_path)

# function to run a program and write output to the shell

def run_process(
    name,
    args,
):
    print(
        "------------------------------------------------------------------------"
    )
    print("Running: %s\nCommand:" % name)
    for arg in args:
        print(arg),
    print("")
    print(
        "------------------------------------------------------------------------"
    )
    process = subprocess.Popen(args)

    process.communicate()
    if process.returncode != 0:
        print("Error: %s did not terminate correctly. Return code: %i." %
              (name, process.returncode))
        sys.exit(1)


# make the catalog of galaxies
run_process("jedicatalog_s", ["./jedicatalog_s", sys.argv[1]])

# make postage stamp images that fit the catalog parameters
#run_process(
#    "jeditransform_s",
#    ['./jeditransform_s', config['catalog_file'], config['dislist_file'], threads
#])

# lens the galaxies one at a time boo
#run_process("jedidistort_s", [
#    './jedidistort_s', config['nx'], config['ny'], config['dislist_file'],
#    config['lenses_file'], config['pix_scale'], config['lens_z'], threads
#])

# Transform and Distort
run_process("jeditranstort_s", [
    './jeditranstort_s', config['catalog_file'], config['lenses_file'], threads,
    config['nx'], config['ny'], config['pix_scale'], config['lens_z'], 
    config['HST_image'], sys.argv[1]
])

# combine the lensed galaxies onto one large image
#run_process("jedipaste (make unconvolved image)", [
#    './jedipaste', config['nx'], config['ny'], config['distortedlist_file'],
#    config['HST_image']
#])

# convonlve the large image with the PSF
# this creates one image for each band of the image
run_process("jediconvolve", [
    './jediconvolve', config['HST_image'], config['psf_file'], convolved_path
])
#
# combine each band into a single image
run_process("jedipaste (make convolved image)", [
    './jedipaste', config['nx'], config['ny'], config['convolvedlist_file'],
    config['HST_convolved_image']
])
#
# scale the image down from HST to LSST scale and trim the edges
run_process("jedirescale", [
    './jedirescale', config['HST_convolved_image'], config['pix_scale'],
    config['final_pix_scale'], config['x_trim'], config['y_trim'],
    config['LSST_convolved_image']
])
#
# simulate exposure time and add Poisson noise
run_process("jedinoise", [
    './jedinoise', config['LSST_convolved_image'], config['exp_time'],
    config['noise_mean'], config['LSST_convolved_noise_image']
])
#
'''
# make the folder for the 90_ simulations
# this is an ugly, but fast way to do this. almost all of the next section is redundant
# if this gets any more complicated, it should be completely rewritten

pre = "90_"
config_record['90_prefix'] = pre + config_record['prefix']
config_record['90_output_folder'] = pre + config_record['output_folder']
config['90_output_folder'] = config_record['90_output_folder']

config['90_HST_image'] = config_record['90_output_folder'] + \
    config_record['90_prefix']+config_record['HST_image']
config['90_HST_convolved_image'] = config_record['90_output_folder'] + \
    config_record['90_prefix']+config_record['HST_convolved_image']
config['90_LSST_convolved_image'] = config_record['90_output_folder'] + \
    config_record['90_prefix']+config_record['LSST_convolved_image']
config['90_LSST_convolved_noise_image'] = config_record['90_output_folder'] + \
    config_record['90_prefix']+config_record['LSST_convolved_noise_image']
config['90_catalog_file'] = config_record['90_output_folder'] + \
    config_record['90_prefix'] + config_record['catalog_file']
config['90_dislist_file'] = config_record['90_output_folder'] + \
    config_record['90_prefix'] + config_record['dislist_file']
config['90_distortedlist_file'] = config_record['90_output_folder'] + \
    config_record['90_prefix'] + config_record['distortedlist_file']
config['90_convolvedlist_file'] = config_record['90_output_folder'] + \
    config_record['90_prefix'] + config_record['convolvedlist_file']


# if the output folder doesn't exist, create it
if (not os.path.exists(config['90_output_folder'])):
    os.makedirs(config['90_output_folder'])

# make folder for convolved images
convolved_path = "%sconvolved/" % (config['90_output_folder'])
if not os.path.exists(convolved_path):
    os.makedirs(convolved_path)

# make folders because cfitsio is bad at it
for x in range(0, int(math.ceil(float(config['num_galaxies']) / 1000))):
    postage_path = "%sstamp_%i" % (config['90_output_folder'], x)
    distorted_path = "%sdistorted_%i" % (config['90_output_folder'], x)
    if not os.path.exists(postage_path):
        os.makedirs(postage_path)
    if not os.path.exists(distorted_path):
        os.makedirs(distorted_path)

old_catalog_file = open(config['catalog_file'], 'r')
catalog_file = open(config['90_catalog_file'], 'w')
for old_line in old_catalog_file:
    l = old_line.split("\t")
    angle = float(l[3]) + 90
    angle -= 360 * (int(angle) / 360)
    l[3] = str(angle)
    l[-1] = l[-1].replace(config['output_folder'], config['90_output_folder'])
    l[-2] = l[-2].replace(config['output_folder'], config['90_output_folder'])
    line = "\t".join(l)
    catalog_file.write(line)
old_catalog_file.close()
catalog_file.close()

old_convolvedlist_file = open(config['convolvedlist_file'], 'r')
convolvedlist_file = open(config['90_convolvedlist_file'], 'w')
for old_line in old_convolvedlist_file:
    line = old_line.replace(config['output_folder'],
                            config['90_output_folder'])
    convolvedlist_file.write(line)
old_convolvedlist_file.close()
convolvedlist_file.close()

old_distortedlist_file = open(config['distortedlist_file'], 'r')
distortedlist_file = open(config['90_distortedlist_file'], 'w')
for old_line in old_distortedlist_file:
    line = old_line.replace(config['output_folder'],
                            config['90_output_folder'])
    distortedlist_file.write(line)
old_distortedlist_file.close()
distortedlist_file.close()

# make postage stamp images that fit the catalog parameters
run_process(
    "jeditransform",
    ['./jeditransform', config['90_catalog_file'], config['90_dislist_file']])

# lens the galaxies one at a time
run_process("jedidistort", [
    './jedidistort', config['nx'], config['ny'], config['90_dislist_file'],
    config['lenses_file'], config['pix_scale'], config['lens_z']
])

# combine the lensed galaxies onto one large image
run_process("jedipaste (make unconvolved image)", [
    './jedipaste', config['nx'], config['ny'], config['90_distortedlist_file'],
    config['90_HST_image']
])

# convonlve the large image with the PSF
# this creates one image for each band of the image
run_process("jediconvolve", ['./jediconvolve', config['90_HST_image'], config['psf_file'], convolved_path])

# combine each band into a single image
run_process("jedipaste (make convolved image)", ['./jedipaste', config['nx'], config['ny'], config['90_convolvedlist_file'], config['90_HST_convolved_image']])

# scale the image down from HST to LSST scale and trim the edges
run_process("jedirescale", ['./jedirescale', config['90_HST_convolved_image'], config['pix_scale'], config['final_pix_scale'], config['x_trim'], config['y_trim'], config['90_LSST_convolved_image']])

# simulate exposure time and add Poisson noise
run_process("jedinoise", ['./jedinoise', config['90_LSST_convolved_image'], config['exp_time'], config['noise_mean'], config['90_LSST_convolved_noise_image']])
'''
print("jedisim successful! Exiting.")

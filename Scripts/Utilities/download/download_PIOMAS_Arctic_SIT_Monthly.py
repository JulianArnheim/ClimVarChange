"""
Retrieve the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) assimilative ice-ocean model
monthly Sea Ice Thickness (SIT) from the University of Washington Polar Science Center via SFTP

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
    Author : Julian B Arnheim
"""

### Import modules
import gzip
import os
import paramiko
import shutil
import time

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data', 'raw')

### Variable and years
var = 'SIT'
year_min, year_max = 1979, 2025

print(f'\n---- Download PIOMAS Monthly Sea Ice Thickness (SIT) ----\n')

### SFTP server and credentials
host = 'pscfiles.apl.uw.edu'
username, password, remote_dir = 'anonymous', 'anonymous', '/pscfiles/zhang/PIOMAS/data/v2.1/heff/'

### Set up SSH client and SFTP session
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=username, password=password)
sftp = ssh.open_sftp()

### Download and decompress
print(f'Connecting to PIOMAS SFTP server to download monthly SIT ({year_min}–{year_max}) ...')
for year in range(year_min, year_max + 1):

    ### Construct filenames for compressed (.gz) and decompressed (.heff) files
    gz_filename = f'heff.H{year}.gz'
    remote_path = remote_dir + gz_filename
    local_gz_path = os.path.join(data_dir, gz_filename)
    local_heff_path = os.path.join(data_dir, f'heff.H{year}')

    try:
        ### Download the compressed .gz file from the SFTP server
        sftp.get(remote_path, local_gz_path)
        print(f'Downloaded {gz_filename} -> {local_gz_path}')

        ### Decompress the .gz file into a binary .heff file
        with gzip.open(local_gz_path, 'rb') as f_in:
            with open(local_heff_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f'Decompressed {gz_filename} -> {local_heff_path}\n')
        os.remove(local_gz_path)

    except FileNotFoundError:
        print(f'{gz_filename} not found on server\n')

### Close connections
sftp.close()
ssh.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')
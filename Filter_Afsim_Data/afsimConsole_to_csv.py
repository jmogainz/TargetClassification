"""
Script to extract data from simulation logs and create a CSV file
"""
import sys
import os
import pandas as pd

# check if cmd line arg path exists
path = None
if len(sys.argv) > 1:
    if os.path.exists(sys.argv[1]):
        path = sys.argv[1]

if path:
    with open(path, 'r') as f:
        lines = f.readlines()
        count = 0
        store = []
        for line in lines:
            if 'Irradiance' in line and count == 0:
                beam_intensity = float(line.split(':')[2].strip())
                count += 1
            if 'Spot Size' in line and count == 1:
                beam_spread = float(line.split(':')[2].strip())
                count += 1
            if 'Range' in line and count == 2:
                range = int(line.split(':')[2].strip())
                count += 1
            if count == 3:
                count = 0
                # append range, beam_intensity, beam_spread to csv file
                store.append([range, beam_intensity, beam_spread])

        df = pd.DataFrame(store, columns=['Range', 'Beam Intensity', 'Beam Spread'])
        df = df.drop_duplicates(keep='first')
        df = df.sort_values(by=['Range'])
        df.to_csv('beam_data.dat', index=False)
else:
    print('Please provide a valid path to afsim console log as cmd line arg')
    sys.exit(1)



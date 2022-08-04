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
        store1 = []
        store2 = []
        tg_met = False
        min = 1994.5
        for line in lines:
            if 'Irradiance' in line and count == 0:
                beam_intensity = float(line.split(':')[2].strip())
                count += 1
            if 'Spot Size' in line and count == 1:
                beam_spread = float(line.split(':')[2].strip())
                count += 1
            if 'Range' in line and count == 2:
                range = float(line.split(':')[2].strip())
                if range == min:
                    tg_met = True
                count += 1
            if count == 3:
                count = 0
                if not tg_met:
                    store1.append([range, beam_intensity, beam_spread])
                else:
                    store2.append([range, beam_intensity, beam_spread])
                

        df = pd.DataFrame(store1, columns=['Range', 'Beam Intensity', 'Beam Spread'])
        df = df.drop_duplicates(keep='first')
        df = df.sort_values(by=['Range'])
        df.to_csv('beam_data_south.dat', index=False)

        df2 = pd.DataFrame(store2, columns=['Range', 'Beam Intensity', 'Beam Spread'])
        df2 = df2.drop_duplicates(keep='first')
        df2 = df2.sort_values(by=['Range'])
        df2.to_csv('beam_data_north.dat', index=False)

    # make plot of first column as x axis and second column as y axis using df
    import matplotlib.pyplot as plt

    # plt.plot(df['Range'], df['Beam Intensity'], 'r-')
    # plt.ylabel('Beam Intensity')
    # plt.xlabel('Range (m)')
    # plt.show()
    # plt.plot(df['Range'], df['Beam Spread'], 'b-')
    # plt.xlabel('Range (m)')
    # plt.ylabel('Beam Spread (cm)')
    # plt.show()

    plt.plot(df['Range'], df['Beam Spread'], 'b-')
    plt.plot(df2['Range'], df2['Beam Spread'], 'g-')
    plt.legend(['South', 'North'])
    plt.xlabel('Range (m)')
    plt.ylabel('Beam Spread (cm)')
    plt.show()

    
else:
    print('Please provide a valid path to afsim console log as cmd line arg')
    sys.exit(1)



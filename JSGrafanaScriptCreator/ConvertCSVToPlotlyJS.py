"""
Read in csv and convert to plotlyjs script
"""

import tkinter as tk
import tkinter.filedialog as fd
import os
import pandas as pd
import io

def get_path(wildcard):
    root = tk.Tk()
    root.withdraw()
    file_path = \
        fd.askopenfilename(parent=None, defaultextension='.csv',
                            initialdir=os.getcwd(),
                            title="Choose Classification Data Analyzer CSV File",
                            filetypes=[("JSON OR YAML Config", wildcard)])
    root.update()
    root.destroy()
    return file_path

csv = get_path("*.csv")
filename = os.path.basename(csv)
df = pd.read_csv(csv)

# configurables
colors = ['#1f77b4',  # muted blue
          '#ff7f0e',  # safety orange
          '#2ca02c',  # cooked asparagus green
          '#d62728',  # brick red
          '#9467bd',  # muted purple
          '#8c564b',  # chestnut brown
          '#e377c2',  # raspberry yogurt pink
          '#7f7f7f',  # middle gray
          '#bcbd22',  # curry yellow-green
          '#17becf']  # blue-teal
line_width = 2


with open(f"GuardianGrafanaPlot_{filename}.js", "w") as f:
    buff = io.StringIO()
    buff.write("data = [\n")
    cols = df.columns.tolist()
    time = df['Time'].tolist()

    track = 0
    truth_pred_pair = []
    for i in range(1, len(cols)):
        col = df[cols[i]].tolist()

        # plotly requires that there be one more class value before track is dropped
        for j in range(len(col)):
            try:
                if (col[j] != "_") and (col[j+1] == "_"):
                    col[j+1] = col[j]
                    break
            except:
                pass

        color = colors[2] if i % 2 == 0 else colors[3]
        track += 1 if i % 2 != 0 else 0    
        
        temp_data = ("   {\n"
                "     type: \'scatter\',\n"
                "     mode: \'lines\',\n"
               f"     title: \'{cols[i]}\',\n" 
               f"     x: {time},\n"
               f"     y: {col},\n"
               f"     legendgroup: \'g{track}\',\n"
               f"     xaxis: \'x{track}\',\n"
               f"     yaxis: \'y{track}\',\n"
               f"     name: \'{cols[i]}\',\n"
               f"     line: {{color: \'{color}\', width: {line_width}, shape: \'hv\'}},\n")

        truth_pred_pair.append(temp_data)

        if len(truth_pred_pair) == 2:
            buff.write(truth_pred_pair[1]) # truth needs to be written first
            buff.write("     showlegend: false,\n"
                       "     hoverinfo: \'skip\',\n")
            buff.write("   }\n") if i == len(cols) else buff.write("   },\n")
            buff.write(truth_pred_pair[0])
            buff.write("   }\n") if i == len(cols) else buff.write("   },\n")
            buff.write(truth_pred_pair[1]) # truth needs to be written again for displaying
            buff.write("   }\n") if i == len(cols) else buff.write("   },\n")
            truth_pred_pair.clear()

    # these legend size calculations were tested extensively and seem to work
    # do not change
    height = len(cols)//2 * 200
    buff.write("]\n\n")
    buff.write("layout = {\n"
               "   grid: {\n"
              f"      rows: {len(cols) - 1}, columns: 1, pattern: \'independent\',\n"
              f"      ygap: .5,\n"
               "   },\n"
               "   margin: {l: 160},\n"
              f"   height: {height},\n"
               "   legend: {\n"
               "      yanchor: \'top\',\n"
              f"      tracegroupgap: {height / (len(cols) * 1.575)},\n"
               "      font: {\n"
               "         color: \'black\',\n"
               "      },\n"
               "   },\n")

    # define each consecutive axis
    for i in range(2, len(cols), 2):
        # extract number from column name
        track = ''
        for z in cols[i]:
            if z.isdigit():
                track += z
        track = int(track)

        axis = i // 2

        if axis == 1: axis = '' # 'xaxis' represents the first axis in plotly
        buff.write(f"   xaxis{axis}: {{\n"
                    "      autorange: false,\n"
                   f"      range: [{time[0]}, {time[-1]}],\n"
                    "      type: \'linear\',\n")
        if axis != len(cols) // 2:
            buff.write("      showticklabels: false,\n")
        buff.write("   },\n")

        buff.write(f"   yaxis{axis}: {{\n"
                    "      title: {\n" 
                    "         font: {\n"
                    "            color: \'black\',\n"
                    "         },\n"
                   f"         text: \'Track {track}\',\n"
                    "      },\n"
                    "      tickfont: {\n"
                    "         color: \'black\',\n"
                    "      },\n"
                    "   },\n")

    buff.write("};\n\n")
    buff.write("return {data: data, layout: layout, config: {displayModeBar: false}}")

    # replace all '_' with nothing
    f.write(buff.getvalue().replace('\'_\'', ''))
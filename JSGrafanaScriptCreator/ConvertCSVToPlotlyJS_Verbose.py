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
    xaxes_list = []
    track = 1
    for i in range(1, len(cols)):
        col = df[cols[i]].tolist()
        time = df['Time'].tolist()
        
        for j in range(len(col)):
            try:
                if (col[j] != "_") and (col[j+1] == "_"):
                    col[j+1] = col[j]
                    break
            except:
                pass
            
        color = colors[2] if i % 2 == 0 else colors[3]
        
        buff.write("   {\n"
                "     type: \'scatter\',\n"
                "     mode: \'lines\',\n"
               f"     title: \'{cols[i]}\',\n" 
               f"     x: {time},\n"
               f"     y: {col},\n"
               f"     xaxis: \'x{i}\',\n"
               f"     yaxis: \'y{i}\',\n"
               f"     name: \'{cols[i]}\',\n"
               f"     line: {{color: \'{color}\', width: {line_width}, shape: \'hv\'}}\n")
        buff.write("   }\n") if i == len(cols) else buff.write("   },\n")

        xaxes_list.append(f"x{i}")
    
    buff.write("]\n\n")
    buff.write("layout = {\n"
               "   grid: {\n"
              f"      rows: {len(cols) - 1}, columns: 1, pattern: \'independent\',\n"
              f"      ygap: .5,\n"
               "   },\n"
               "   margin: {l: 150},\n")

    buff.write("   xaxis: {\n"
               "      visible: false,\n"
               "   },\n")
    track = 1
    for i in range(1, len(cols)):
        # slice cols[i] string from the first integer to the end
        col = cols[i][cols[i].find(' '):]
        buff.write(f"   yaxis{i}: {{\n")
        if i % 2 == 0:
            # buff.write(f"      title: \'T\', ")
            track += 1
        else:
            buff.write(f"      title: \'ST{track}\', ")
        buff.write(f"anchor: \'y{i}\'}},\n")
        if i % 2 != 0:
            buff.write(f"   xaxis{i}: {{\n"
                   f"      anchor: \'x{i}\',\n")
            buff.write(f"      showticklabels: false,\n")
            buff.write("   },\n")
                    
    buff.write("};\n\n")
    buff.write("return {data: data, layout: layout, config: {displayModeBar: false}}")

    # replace all '_' with nothing
    f.write(buff.getvalue().replace('\'_\'', ''))
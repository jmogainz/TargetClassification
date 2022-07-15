"""
Remove all files in directory that are version 3 and version 1
"""
import os
import pandas as pd

guardian_dir = "Missile_Capture-Classification-Guard_Missile"

def remove_version_1_and_3(guardian_dir):
    for file in os.listdir(os.path.join(os.getcwd(), guardian_dir)):
        if file.endswith("3.csv"):
            os.remove(os.path.join(os.getcwd(), guardian_dir, file))
        if file.endswith("1.csv"):
            os.remove(os.path.join(os.getcwd(), guardian_dir, file))


if __name__ == "__main__":
    remove_version_1_and_3(guardian_dir)
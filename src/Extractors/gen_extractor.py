# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:02:14 2023

@author: Konstantinos
"""


import numpy as np
import h5py
import os
import argparse

## File structure is
# box, cycle, time, mpi, rank0 ... rank99.
# This iterates over all the ranks
def extractor(filename):
    '''
    Loads the file, extracts X,Y,Z and Density. 
    
    Parameters
    ----------
    f : str, 
        hdf5 file name. Contains the data
    
    Returns
    -------
    X : np.array, float64
        X - coordinate
    Y : np.array, float64
        Y - coordinate.
    Z : np.array, float64
        Z - coordinate.
    Den : np.array, float64
        Density.
    
    '''
    # Read File
    f = h5py.File(filename, "r")
    # HDF5 are dicts, get the keys.
    keys = f.keys() 
    # List to store the length of each rank
    lengths = []
    # List with keys that don't hold relevant data
    not_ranks = ['Box', 'Cycle', 'Time', 'mpi']
    
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # Store the length of the dataset
            lengths.append(len(f[key]['X']))
    
    # Use lists for clarity
    X = []
    # Iterate over ranks
    for key in keys:
        if key in not_ranks:
            # Skip whatever is not a mpi rank
            continue
        else:
            # For some reason, having the collumns into variables is way faster.
            x_data = f[key]['Dissipation']
            for i in range(len(x_data)):
                X.append(x_data[i])

    # Close the file
    f.close()
    return X 

def main():
    parser = argparse.ArgumentParser(
        description="Parses simulation features for file extraction"
    )

    # Add arguments to the parser
    parser.add_argument(
        "-m", "--mass",
        type = float,
        help = "Mass of the star",
        required = True
    )

    parser.add_argument(
        "-r", "--radius",
        type = float,
        help = "Radius of the star",
        required = True
    )

    parser.add_argument(
        "-b", "--blackhole",
        type = str,
        help = "Mass of the Black Hole",
        required = True
    )

    parser.add_argument(
        "-n", "--name",
        type = str,
        help = 'Name of the directory to save at',
        required = True
    )
    
    parser.add_argument(
        "-f", "--first",
        type = int,
        help = 'First snapshot to extract',
        required = True,
    )

    parser.add_argument(
        "-l", "--last",
        type = int,
        help = 'Last snapshot to extract',
        required = True,
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    simname = args.name
    m = args.mass
    r = args.radius
    mbh = float(args.blackhole)

    fixes = np.arange(args.first, args.last + 1)
    realpre = '/data1/kilmetisk/TDE/'
    for fix in fixes:
        star = 'half'
        snapshot = f'{realpre}{simname}/snap_{fix}/snap_{fix}.h5'
        pre = f'{realpre}{simname}/snap_{fix}/'
        suf = f'_{fix}'
        
        try:
            X = extractor(snapshot)
        except FileNotFoundError:
            continue
        print('Did ', fix)
        #%% Save to another file.
        np.save(pre + 'Diss' + suf, X)   
if __name__ == "__main__":
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
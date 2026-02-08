import os
import h5py

def store_results(file_path, params, trajectories, fks):
    if not os.path.exists(file_path):
        # Create empty file with space for 0 entries, allowing future expansion
        with h5py.File(file_path, "w") as f:
            f.create_dataset("params", shape=(0, params.shape[1]), maxshape=(None, params.shape[1]), dtype='float32', compression="gzip")
            f.create_dataset("trajectories", shape=(0, *trajectories.shape[1:]), maxshape=(None, *trajectories.shape[1:]), dtype='float32', compression="gzip")
            f.create_dataset("fks", shape=(0, *fks.shape[1:]), maxshape=(None,  *fks.shape[1:]), dtype='float32', compression="gzip")
           
    with h5py.File(file_path, "a") as f:
        params_ds = f["params"]
        traj_ds = f["trajectories"]
        fk_ds = f["fks"]
        
        # Extend the datasets
        cur_size=params_ds.shape[0]
        new_size = cur_size + params.shape[0]
        params_ds.resize((new_size, params.shape[1]))
        traj_ds.resize((new_size, *trajectories.shape[1:]))
        fk_ds.resize((new_size, *fks.shape[1:]))

        # Append the new data
        params_ds[cur_size:,:] = params
        traj_ds[cur_size:,:,:] = trajectories
        fk_ds[cur_size:,:,:] = fks
        print("Added results.")

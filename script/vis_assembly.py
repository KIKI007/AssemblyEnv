import os
import numpy as np
import polyscope as ps
from AssemblyEnv.geometry import AssemblyChecker, AssemblyGUI
import time
import itertools
from AssemblyEnv import DATA_DIR
def gui(viewer):
    ps.init()
    ps.set_navigation_style("turntable")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.look_at((0., -5., 3.5), (0., 0., 0.))

    viewer.render()
    viewer.update_status(status)

    ps.set_user_callback(viewer.interface)

    ps.show()

if __name__ == "__main__":
    viewer = AssemblyGUI()
    viewer.load_from_file(DATA_DIR + "/block/dome.obj")
    status = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 2])
    gui(viewer)
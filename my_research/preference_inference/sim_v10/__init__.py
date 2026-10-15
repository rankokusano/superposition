"""
v10 copy of the original paper's simulator (root simulation/), with a
switchable landmark layout (sim_v10/layout.py). Import as a package so it
never collides with the root `simulation` package:

    sys.path.insert(0, '/work/my_research/preference_inference')
    from sim_v10 import creator, layout
    env = creator.create_environment(cfg.environment)
    env.init(); env.off_display()
    env.set_layout(layout.SWAP_RG)   # per episode; default 'original'
"""

import torch


path = "/users/ashah3/data-gdk/reai_group5/summer_experiments/skyline_hier_multi_npy_goal_16_saved_models/checkpoint_skyline_hier_multi_npy_goal_16_2020-05-29_13:29:25.366978_1650000.tar"
checkpoint = torch.load(path)
print(checkpoint)

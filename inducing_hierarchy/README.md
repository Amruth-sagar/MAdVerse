# Inducing Hierarchy readme 
These codes are designed for the task of inducing hierarchical structure in other ads data. In our case we tested on a small subset of Pitt's ad dataset and small subset of our multilingual newspaper dataset. 

### :file_folder: Sampled Dataset for inducing hierarchy on other ads dataset
We have inferenced our hierarchical classifier on a subset of Pitt's ad dataset and a subset of our multilingual dataset, whose results are shown in the paper.
The Sampled dataset can be downloaded from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishabh_s_students_iiit_ac_in/Etk7J95oiO9Hlnd87shjj2kBOZEbLLOEf764g2S4z6TK9A?e=C8gVdg)


1. To get our sampled data from original dataset of Pitt's ad and Madverse
- Download the original dataset from the links provided in the paper and place it in separate folders(say pitt_ad and madverse).
- Sample the data using the following command:
```python sample_adrianna_and dedup.py --adriana_path pitt_ad --madverse_path madverse --sampling_seed 1 --num_images 1000 --sim_threshold 0.9 --output_path sampled_data```

2. The annotation files is present in ```annotations``` folder, which contains the hierarchical annotations for the sampled data. Use these files and follow the steps mentioned in the [README](../README.md) to induce hierarchy on your dataset.
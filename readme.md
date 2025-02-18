# MedForge: Building Medical Foundation Models Like Open Source Software Development
The official implementation of MedForge. We have released the implementation of our methodology, the complete code will be updated soon.
![The overview of MedForge: (a) Feature branch development. Branch contributor should commit its branch plugin module and distilled data, then push them to MedForge. MedForge will merge the feature branch with the main branch. In our experiments, we adopt LoRA module as the plugin module. (b) Merging stage. Branch contributors can asynchronously commit and push their branch plugin modules and the distilled datasets to the main branch. Forge items of the main branch will be updated to equip the main branch model with new capabilities.](./img/overview.png)
## code description
distill_dm: generate distilled branch datasets.

train_lora: train branch loras on branch datasets for subsequent merging process.

model_fusion: the implementation of MedForge-Fusion.

model_mixture: the implementation of MedForge-Mixture.

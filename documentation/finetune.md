## Finetuning instructions



# Symbolically linking data to your data directory

If you don't want to modify the dataloader, you can symbolically 

```bash
ln -s <your_data_directory> <medvae_installed_directory>/medvae/data 
```

# Inference after finetuning


# Helpful tips

There may be a warning for the state_dict to be included in a weight, just go ahead and create a dictionary that can handle that. It will just need a 'state_dict' key.
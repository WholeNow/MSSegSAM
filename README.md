# MSSegSAM

This project is a specific implementation of the [finestSAM](https://github.com/WholeNow/finestSAM) repository, designed for the segmentation of Multiple Sclerosis (MS) lesions in MRI images.

### Dataset Creation
This project includes a specific script to convert MRI images (NIfTI format) into the COCO format required for training.
For detailed instructions on how to use the dataset creation tool, please refer to the [Dataset Creation Documentation](prep_data/README.md).

## To-Do List

- [x] Create the preprocessing script
- [x] Preprocess and merge datasets
- [x] Create script to convert dataset to COCO / Create script to convert dataset directly to tensors -- (extract only slices with lesions)
- [x] Test function
- [x] LoRA Layers (Adapter implementation)
- [x] Added a function to create the bounding boxes for training (suggestion on line 258 [finestSAM/data/dataset.py](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/data/dataset.py))
- [ ] Brightness normalization / other types of normalization of the dataset
- [ ] Prompt selection model (e.g. model for selecting points/boxes of the lesions)
- [ ] Support slices with and without sclerosis
- [ ] Integrate T1 and T2 modalities and union/intersection model
- [ ] Improve the evaluation methods.
- [ ] Add Gradient Accumulation support.
- [ ] Add support for more SAM variants.
- [ ] Implement additional data augmentation techniques.

## License
The model is licensed under the [Apache 2.0 license](https://github.com/WholeNow/MSSegSAM/blob/main/LICENSE.txt).